import numpy as np
import pytest

from spikingjelly.datasets import utils


def _events(timestamps, dtype):
    """Construct a small camera sample with distinct positions/polarities."""
    count = len(timestamps)
    return {
        "t": np.asarray(timestamps, dtype=dtype),
        "x": np.arange(count) % 2,
        "y": np.arange(count) // 2,
        "p": np.arange(count) % 2,
    }


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize(
    "timestamps,frame_count",
    [
        ([100000, 105000, 110000], 3),
        ([100000, 100000, 100000], 1),
        ([100000, 101000, 105000, 109999], 2),
    ],
)
def test_fixed_duration_uses_half_open_time_bins(dtype, timestamps, frame_count):
    """Keep boundary and zero-span events in their own duration intervals."""
    events = _events(timestamps, dtype)
    original = {key: value.copy() for key, value in events.items()}
    expected = np.zeros((frame_count, 2, 2, 2))
    for timestamp, x, y, polarity in zip(
        timestamps, events["x"], events["y"], events["p"], strict=True
    ):
        frame = (timestamp - timestamps[0]) // 5000
        expected[frame, polarity, y, x] += 1

    actual = utils.integrate_events_by_fixed_duration(events, 5000, H=2, W=2)

    np.testing.assert_array_equal(actual, expected)
    for key in events:
        np.testing.assert_array_equal(events[key], original[key])


@pytest.mark.parametrize("timestamps,frame_count", [([10, 15, 20], 3), ([10], 1)])
def test_fixed_duration_file_preserves_frame_count(tmp_path, timestamps, frame_count):
    """Persist the corrected number of frames through the public file API."""
    events_file = tmp_path / "events.npz"
    np.savez(events_file, **_events(timestamps, np.int64))

    def load(path):
        """Load the sample without retaining an open archive."""
        with np.load(path) as archive:
            return dict(archive)

    count = utils.integrate_events_file_to_frames_file_by_fixed_duration(
        load, str(events_file), str(tmp_path), duration=5, H=2, W=2
    )

    assert count == frame_count
    with np.load(tmp_path / f"events_{frame_count}.npz") as archive:
        frames = archive["frames"]
    assert frames.shape == (frame_count, 2, 2, 2)
    assert frames.sum() == len(timestamps)
