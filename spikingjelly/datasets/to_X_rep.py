from dataclasses import dataclass
import numpy as np
import math
from typing import Callable, Optional, Tuple, Union,Any, List

# Code adapted from https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L431,
# https://github.com/neuromorphs/tonic/blob/develop/tonic/transforms.py,
# and https://github.com/neuromorphs/tonic/blob/develop/tonic/functional/to_bina_rep.py
'''
#使用举例说明：（Directions for use）
#在头部导入方法（import method)
from spikingjelly.datasets.to_X_rep import Compose,ToFrame,ToBinaRep,ToVoxelGrid
transform = Compose(
            [
                ToFrame(
                    sensor_size=None,
                    n_time_bins=self.T * self.tbin,
                    ),
                    ToBinaRep(n_frames=self.T, n_bits=self.tbin),
                    ]
                    )
        frames = transform(events)
'''
class Compose:
    """Composes several transforms together. This a literal copy of torchvision.transforms.Compose function for convenience.
    Parameters:
        transforms (list of ``Transform`` objects): list of transform(s) to compose.
                                                    Can combine Tonic, PyTorch Vision/Audio transforms.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Denoise(filter_time=10000),
        >>>     transforms.ToFrame(n_time_bins=3),
        >>> ])
    """

    def __init__(self, transforms: Callable):
        self.transforms = transforms

    def __call__(self, events):
        for t in self.transforms:
            events = t(events)
        return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

@dataclass(frozen=True)
class SliceByTimeBins:
    """
    Slices data and targets along fixed number of bins of time length time_duration / bin_count * (1 + overlap).
    This method is good if your recordings all have roughly the same time length and you want an equal
    number of bins for each recording. Targets are copied.
    Parameters:
        bin_count (int): number of bins
        overlap (float): overlap specified as a proportion of a bin, needs to be smaller than 1. An overlap of 0.1
                    signifies that the bin will be enlarged by 10%. Amount of bins stays the same.
    """

    bin_count: int
    overlap: float = 0

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        events = data
        assert "t" in events.dtype.names
        assert self.overlap < 1

        times = events["t"]
        time_window = (times[-1] - times[0]) // self.bin_count * (1 + self.overlap)
        stride = time_window * (1 - self.overlap)

        window_start_times = np.arange(self.bin_count) * stride + times[0]
        window_end_times = window_start_times + time_window
        indices_start = np.searchsorted(times, window_start_times)
        indices_end = np.searchsorted(times, window_end_times)
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets

def slice_events_by_time_bins(events: np.ndarray, bin_count: int, overlap: float = 0.0):
    return SliceByTimeBins(bin_count=bin_count, overlap=overlap).slice(events, None)[0]

@dataclass(frozen=True)
class SliceByEventCount:
    """
    Slices data and targets along a fixed number of events and overlap size.
    The number of bins depends on the amount of events in the recording.
    Targets are copied.
    Parameters:
        event_count (int): number of events for each bin
        overlap (int): overlap in number of events
        include_incomplete (bool): include the last incomplete slice that has fewer events
    """

    event_count: int
    overlap: int = 0
    include_incomplete: bool = False

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        n_events = len(data)
        event_count = min(self.event_count, n_events)

        stride = self.event_count - self.overlap
        if stride <= 0:
            raise Exception("Inferred stride <= 0")

        if self.include_incomplete:
            n_slices = int(np.ceil((n_events - event_count) / stride) + 1)
        else:
            n_slices = int(np.floor((n_events - event_count) / stride) + 1)

        indices_start = (np.arange(n_slices) * stride).astype(int)
        indices_end = indices_start + event_count
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets

def slice_events_by_count(
    events: np.ndarray,
    event_count: int,
    overlap: int = 0,
    include_incomplete: bool = False,
):
    return SliceByEventCount(
        event_count=event_count, overlap=overlap, include_incomplete=include_incomplete
    ).slice(events, None)[0]

def to_frame_numpy(
    events,
    sensor_size,
    time_window=None,
    event_count=None,
    n_time_bins=None,
    n_event_bins=None,
    overlap=0.0,
    include_incomplete=False,
):
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (event_count) or constant number of frames (n_time_bins / n_event_bins).
    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H,P]
        time_window (None): window length in us.
        event_count (None): number of events per frame.
        n_time_bins (None): fixed number of frames, sliced along time axis.
        n_event_bins (None): fixed number of frames, sliced along number of events in the recording.
        overlap (0.): overlap between frames defined either in time in us, number of events or number of bins.
        include_incomplete (False): if True, includes overhang slice when time_window or event_count is specified. Not valid for bin_count methods.
    Returns:
        numpy array with dimensions (TxPxHxW)
    """
    assert "x" and "t" and "p" in events.dtype.names

    if (
        not sum(
            param is not None
            for param in [time_window, event_count, n_time_bins, n_event_bins]
        )
        == 1
    ):
        raise ValueError(
            "Please assign a value to exactly one of the parameters time_window,"
            " event_count, n_time_bins or n_event_bins."
        )

    if not sensor_size:
        sensor_size_x = int(events["x"].max() + 1)
        sensor_size_p = len(np.unique(events["p"]))
        if "y" in events.dtype.names:
            sensor_size_y = int(events["y"].max() + 1)
            sensor_size = (sensor_size_x, sensor_size_y, sensor_size_p)
        else:
            sensor_size = (sensor_size_x, 1, sensor_size_p)

    # test for single polarity
    if sensor_size[2] == 1:
        events["p"] = 0

  
    if time_window:
        event_slices = slice_events_by_time(
            events, time_window, overlap=overlap, include_incomplete=include_incomplete
        )
    elif event_count:
        event_slices = slice_events_by_count(
            events, event_count, overlap=overlap, include_incomplete=include_incomplete
        )
    elif n_time_bins:
        event_slices = slice_events_by_time_bins(events, n_time_bins, overlap=overlap)
    elif n_event_bins:
        event_slices = slice_events_by_event_bins(events, n_event_bins, overlap=overlap)

    if "y" in events.dtype.names:
        frames = np.zeros((len(event_slices), *sensor_size[::-1]), dtype=np.int16)
        for i, event_slice in enumerate(event_slices):
            np.add.at(
                frames,
                (i, event_slice["p"].astype(int), event_slice["y"], event_slice["x"]),
                1,
            )
    else:
        frames = np.zeros(
            (len(event_slices), sensor_size[2], sensor_size[0]), dtype=np.int16
        )
        for i, event_slice in enumerate(event_slices):
            np.add.at(frames, (i, event_slice["p"].astype(int), event_slice["x"]), 1)
    return frames

@dataclass(frozen=True)
class ToFrame:
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (spike_count) or constant number of frames (n_time_bins / n_event_bins).
    All the events in one slice are added up in a frame for each polarity.
    You can set one of the first 4 parameters to choose the slicing method. Depending on which method you choose,
    overlap will assume different functionality, whether that might be temporal overlap, number of events
    or fraction of a bin. As a rule of thumb, here are some considerations if you are unsure which slicing
    method to choose:

    * If your recordings are of roughly the same length, a safe option is to set time_window. Bare in mind
      that the number of events can vary greatly from slice to slice, but will give you some consistency when
      training RNNs or other algorithms that have time steps.

    * If your recordings have roughly the same amount of activity / number of events and you are more interested
      in the spatial composition, then setting spike_count will give you frames that are visually more consistent.

    * The previous time_window and spike_count methods will likely result in a different amount of frames for each
      recording. If your training method benefits from consistent number of frames across a dataset (for easier
      batching for example), or you want a parameter that is easier to set than the exact window length or number
      of events per slice, consider fixing the number of frames by setting n_time_bins or n_event_bins. The two
      methods slightly differ with respect to how the slices are distributed across the recording. You can define
      an overlap between 0 and 1 to provide some robustness.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size. If omitted, the sensor size is calculated for that sample. However,
                    do use this feature sparingly as when not all pixels fire in a sample, this might cause issues with batching/
                    stacking tensors further down the line.
        time_window (float): time window length for one frame. Use the same time unit as timestamps in the event recordings.
                             Good if you want temporal consistency in your training, bad if you need some visual consistency
                             for every frame if the recording's activity is not consistent.
        spike_count (int): number of events per frame. Good for training CNNs which do not care about temporal consistency.
        n_time_bins (int): fixed number of frames, sliced along time axis. Good for generating a pre-determined number of
                           frames which might help with batching.
        n_event_bins (int): fixed number of frames, sliced along number of events in the recording. Good for generating a
                            pre-determined number of frames which might help with batching.
        overlap (float): overlap between frames defined either in time units, number of events or number of bins between 0 and 1.
        include_incomplete (bool): if True, includes overhang slice when time_window or spike_count is specified.
                                   Not valid for bin_count methods.

    Example:
        >>> from tonic.transforms import ToFrame
        >>> transform1 = ToFrame(time_window=10000, overlap=300, include_incomplete=True)
        >>> transform2 = ToFrame(spike_count=3000, overlap=100, include_incomplete=True)
        >>> transform3 = ToFrame(n_time_bins=100, overlap=0.1)
    """
    

    sensor_size: Optional[Tuple[int, int, int]]
    time_window: Optional[float] = None
    event_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False

    def __call__(self, events):
        return to_frame_numpy(
            events=events,
            sensor_size=self.sensor_size,
            time_window=self.time_window,
            event_count=self.event_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
        )

def to_bina_rep_numpy(
    event_frames: np.ndarray,
    n_frames: int = 1,
    n_bits: int = 8,
):
    """Representation that takes T*B binary event frames to produce a sequence of T frames of N-bit numbers.
    To do so, N binary frames are interpreted as a single frame of N-bit representation. Taken from the paper
    Barchid et al. 2022, Bina-Rep Event Frames: a Simple and Effective Representation for Event-based cameras
    https://arxiv.org/pdf/2202.13662.pdf
    Parameters:
        event_frames: numpy.ndarray of shape (T*BxPxHxW). The sequence of event frames.
        n_frames (int): the number T of bina-rep frames.
        n_bits (int): the number N of bits used in the N-bit representation.
    Returns:
        (numpy.ndarray) the sequence of bina-rep event frames with dimensions (TxPxHxW).
    """
    assert type(event_frames) == np.ndarray and len(event_frames.shape) == 4
    assert n_frames >= 1
    assert n_
    
    bits >= 2

    if event_frames.shape[0] != n_bits * n_frames:
        raise ValueError(
            "the input event_frames must have the right number of frames to the targeted"
            f"sequence of {n_frames} bina-rep event frames of {n_bits}-bit representation."
            f"Got: {event_frames.shape[0]} frames. Expected: {n_frames}x{n_bits}={n_bits * n_frames} frames."
        )

    event_frames = (event_frames > 0).astype(np.float32)  # get binary event_frames

    bina_rep_seq = np.zeros((n_frames, *event_frames.shape[1:]), dtype=np.float32)

    for i in range(n_frames):
        frames = event_frames[i * n_bits : (i + 1) * n_bits]
        bina_rep_frame = bina_rep(frames)
        bina_rep_seq[i] = bina_rep_frame

    return bina_rep_seq


def bina_rep(frames: np.ndarray) -> np.ndarray:
    """Computes one Bina-Rep frame from the sequence of N binary event-frames in parameter.
    Args:
        frames (numpy.ndarray): the sequence of N binary event frames used to compute the bina-rep frame. Shape=(NxPxHxW)
    Returns:
        numpy.ndarray: the resulting bina-rep event frame. Shape=(PxHxW)
    """
    mask = 2 ** np.arange(frames.shape[0] - 1, -1, -1, dtype=np.float32)
    arr_mask = [
        mask for _ in range(frames.shape[1] * frames.shape[2] * frames.shape[3])
    ]
    mask = np.stack(arr_mask, axis=-1)
    mask = np.reshape(mask, frames.shape)

    return np.sum(mask * frames, 0) / (2 ** mask.shape[0] - 1)

@dataclass(frozen=True)
class ToBinaRep:
    """Takes T*B binary event frames to produce a sequence of T frames of N-bit numbers.
    To do so, N binary frames are interpreted as a single frame of N-bit representation. Taken from the paper
    Barchid et al. 2022, Bina-Rep Event Frames: a Simple and Effective Representation for Event-based cameras
    https://arxiv.org/pdf/2202.13662.pdf
    Parameters:
        n_frames (int): the number T of bina-rep frames.
        n_bits (int): the number N of bits used in the N-bit representation.
    Example:
        >>> n_time_bins = n_frames * n_bits
        >>>
        >>> transforms.Compose([
        >>>     transforms.ToFrame(
        >>>         sensor_size=sensor_size,
        >>>         n_time_bins=n_time_bins,
        >>>     ),
        >>>     transforms.ToBinaRep(
        >>>         n_frames=n_frames,
        >>>         n_bits=n_bits,
        >>>     ),
        >>> ])
    """

    n_frames: Optional[int] = 1
    n_bits: Optional[int] = 8

    def __call__(self, event_frames):

        return to_bina_rep_numpy(event_frames, self.n_frames, self.n_bits)

def to_voxel_grid_numpy(events, sensor_size, n_time_bins=10):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical flow, depth, and egomotion
    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H].
        n_time_bins: number of bins in the temporal axis of the voxel grid.
    Returns:
        numpy array of n event volumes (n,w,h,t)
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2

    voxel_grid = np.zeros((n_time_bins, sensor_size[1], sensor_size[0]), float).ravel()

    # normalize the event timestamps so that they lie between 0 and n_time_bins
    ts = (
        n_time_bins
        * (events["t"].astype(float) - events["t"][0])
        / (events["t"][-1] - events["t"][0])
    )
    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    pols = events["p"]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(
        voxel_grid, (n_time_bins, 1, sensor_size[1], sensor_size[0])
    )

    return voxel_grid

@dataclass(frozen=True)
class ToVoxelGrid:
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning
    of optical flow, depth, and egomotion.
    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        n_time_bins (int): fixed number of time bins to slice the event sample into."""

    sensor_size: Tuple[int, int, int]
    n_time_bins: int

    def __call__(self, events):

        return to_voxel_grid_numpy(
            events.copy(), self.sensor_size, self.n_time_bins
        )
    
@dataclass(frozen=True)
class ToImage:
    """Counts up all events to a *single* image of size sensor_size. ToImage will typically
    be used in combination with SlicedDataset to cut a recording into smaller chunks that
    are then individually binned to frames.
    """

    sensor_size: Tuple[int, int, int]

    def __call__(self, events):

        frames = to_frame_numpy(
            events=events, sensor_size=self.sensor_size, event_count=len(events)
        )

        return frames.squeeze(0)
