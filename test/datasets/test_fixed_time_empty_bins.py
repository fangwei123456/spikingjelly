import numpy as np
import pytest

from spikingjelly.datasets import shd, utils


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


@pytest.mark.parametrize(
    ("timestamps", "frames_num", "left", "right"),
    [
        ([0.0, 0.05, 0.12, 0.47, 0.53, 0.6], 6, [0, 2, 3, 3, 3, 4], [2, 3, 3, 3, 4, 6]),
        ([0.0, 0.9], 3, [0, 1, 1], [1, 1, 2]),
        ([0.0, 0.25, 0.5, 0.9], 2, [0, 2], [2, 4]),
    ],
)
def test_shd_time_segmentation_preserves_events_with_empty_bins(
    timestamps: list[float], frames_num: int, left: list[int], right: list[int]
) -> None:
    """SHD/SSC time splitting keeps empty intervals instead of raising."""
    events_t = np.array(timestamps, dtype=np.float32)
    actual_left, actual_right = shd._cal_fixed_frames_number_segment_index(
        events_t, split_by="time", frames_num=frames_num
    )

    np.testing.assert_array_equal(actual_left, left)
    np.testing.assert_array_equal(actual_right, right)

    units = np.arange(len(timestamps), dtype=np.uint16)
    frames = shd._integrate_events_by_fixed_frames_number(
        {"t": events_t, "x": units}, split_by="time", frames_num=frames_num, W=8
    )
    expected = np.zeros((frames_num, 8))
    for index, (j_l, j_r) in enumerate(zip(left, right)):
        expected[index, units[j_l:j_r]] = 1

    np.testing.assert_array_equal(frames, expected)
    assert frames.sum() == len(timestamps)


def test_shd_time_segmentation_partitions_events_on_rounded_boundaries() -> None:
    """Events on a frame boundary are counted once, despite rounding in ``t + dt``."""
    events_t = np.array(
        [1.3306491374969482, 1.381555199623108, 1.432461142539978, 1.4833672046661377],
        dtype=np.float32,
    )
    left, right = shd._cal_fixed_frames_number_segment_index(
        events_t, split_by="time", frames_num=3
    )

    np.testing.assert_array_equal(left, [0, 1, 2])
    np.testing.assert_array_equal(right, [1, 2, 4])
    frames = shd._integrate_events_by_fixed_frames_number(
        {"t": events_t, "x": np.zeros(4, dtype=np.uint16)},
        split_by="time",
        frames_num=3,
        W=1,
    )
    assert frames.sum() == events_t.size


@pytest.mark.parametrize(
    ("dtype", "timestamps"),
    [
        (
            np.float64,
            [
                70.91828603166262,
                892.9182860316627,
                1714.9182860316625,
                2536.918286031663,
            ],
        ),
        (
            np.float32,
            [
                58.24595260620117,
                1247.2459716796875,
                2436.245849609375,
                3625.245849609375,
            ],
        ),
    ],
)
def test_time_segmentation_partitions_float_events_on_boundaries(
    dtype, timestamps: list[float]
) -> None:
    """Frame j ends at t_0 + dt * (j + 1), so a boundary event is counted once."""
    events_t = np.array(timestamps, dtype=dtype)
    left, right = utils.cal_fixed_frames_number_segment_index(
        events_t, split_by="time", frames_num=3
    )

    np.testing.assert_array_equal(left, [0, 1, 2])
    np.testing.assert_array_equal(right, [1, 2, 4])
    events = {
        "t": events_t,
        "x": np.zeros(events_t.size, dtype=int),
        "y": np.zeros(events_t.size, dtype=int),
        "p": np.zeros(events_t.size, dtype=int),
    }
    frames = utils.integrate_events_by_fixed_frames_number(
        events, split_by="time", frames_num=3, H=1, W=1
    )
    assert frames.sum() == events_t.size
