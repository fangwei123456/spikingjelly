import numpy as np
import pytest

from spikingjelly.datasets import utils


@pytest.mark.parametrize(
    ("timestamps", "frames_num", "left", "right"),
    [
        ([0, 9], 3, [0, 1, 1], [1, 1, 2]),
        ([4, 4], 3, [0, 0, 0], [0, 0, 2]),
        ([0, 1, 2], 5, [0, 0, 0, 0, 0], [0, 0, 0, 0, 3]),
        ([5], 1, [0], [1]),
        ([0, 3, 6, 9], 3, [0, 1, 2], [1, 2, 4]),
        ([0, 3, 9], 3, [0, 1, 2], [1, 2, 3]),
        ([0, 0, 1, 9], 1, [0], [4]),
    ],
)
def test_time_segmentation_preserves_events_with_empty_bins(
    timestamps: list[int], frames_num: int, left: list[int], right: list[int]
) -> None:
    """Preserve event counts when time bins are empty or have zero width."""
    events_t = np.array(timestamps)
    actual_left, actual_right = utils.cal_fixed_frames_number_segment_index(
        events_t, split_by="time", frames_num=frames_num
    )

    np.testing.assert_array_equal(actual_left, left)
    np.testing.assert_array_equal(actual_right, right)

    events = {
        "t": events_t,
        "x": np.zeros(len(timestamps), dtype=int),
        "y": np.zeros(len(timestamps), dtype=int),
        "p": np.zeros(len(timestamps), dtype=int),
    }
    frames = utils.integrate_events_by_fixed_frames_number(
        events, split_by="time", frames_num=frames_num, H=1, W=1
    )

    assert frames.shape == (frames_num, 2, 1, 1)
    np.testing.assert_array_equal(frames[:, 0, 0, 0], np.array(right) - left)
    np.testing.assert_array_equal(frames[:, 1], 0)
    assert frames.sum() == len(timestamps)


@pytest.mark.parametrize("dtype", [">u4", np.uint32, np.int32, np.int64, np.float64])
def test_time_integration_preserves_dense_frame_values(dtype) -> None:
    relative_t = np.repeat(np.arange(17), 3)
    events_t = (relative_t + 1_000_000).astype(dtype)
    rng = np.random.default_rng(794)
    events = {
        "t": events_t,
        "x": rng.integers(0, 4, size=events_t.size),
        "y": rng.integers(0, 3, size=events_t.size),
        "p": rng.integers(0, 2, size=events_t.size),
    }
    # Independent per-event reference, without interval-index searches.
    frame_indices = np.minimum(relative_t // 4, 3)
    expected = np.zeros((4, 2, 3, 4))
    np.add.at(expected, (frame_indices, events["p"], events["y"], events["x"]), 1)

    frames = utils.integrate_events_by_fixed_frames_number(
        events, split_by="time", frames_num=4, H=3, W=4
    )

    np.testing.assert_array_equal(frames, expected)
    assert frames.dtype == expected.dtype
    np.testing.assert_array_equal(events_t, (relative_t + 1_000_000).astype(dtype))
