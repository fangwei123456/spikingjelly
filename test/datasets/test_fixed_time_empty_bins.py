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
