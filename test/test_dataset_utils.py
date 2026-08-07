import struct

import numpy as np
import pytest
import torch

from spikingjelly.datasets import transform, utils


def _events():
    return {
        "t": np.array([0, 1, 2, 3]),
        "x": np.array([0, 0, 1, 1]),
        "y": np.array([0, 0, 1, 1]),
        "p": np.array([0, 0, 1, 1]),
    }


def test_integrate_events_segment_counts_repeated_positions_and_polarities():
    events = _events()

    frame = utils.integrate_events_segment_to_frame(
        events["x"], events["y"], events["p"], H=2, W=2
    )

    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = 2
    expected[1, 1, 1] = 2
    assert np.array_equal(frame, expected)


def test_fixed_frame_number_preserves_all_events_and_boundaries():
    frames = utils.integrate_events_by_fixed_frames_number(
        _events(), split_by="number", frames_num=3, H=2, W=2
    )

    assert frames.shape == (3, 2, 2, 2)
    assert frames.sum() == 4
    assert np.array_equal(frames.sum(axis=(1, 2, 3)), np.array([1, 1, 2]))


def test_fixed_time_segmentation_includes_the_last_boundary():
    left, right = utils.cal_fixed_frames_number_segment_index(
        np.array([0, 1, 2, 3]), split_by="time", frames_num=2
    )

    assert np.array_equal(left, np.array([0, 1]))
    assert np.array_equal(right, np.array([1, 4]))


def test_fixed_duration_preserves_all_events():
    events = _events()
    events["t"] = events["t"] + 10

    frames = utils.integrate_events_by_fixed_duration(events, duration=2, H=2, W=2)

    assert frames.shape == (2, 2, 2, 2)
    assert frames.sum() == 4
    assert np.array_equal(frames.sum(axis=(1, 2, 3)), np.array([2, 2]))


def test_event_file_integration_writes_frames_archive(tmp_path):
    events_file = tmp_path / "events.npz"
    np.savez(events_file, **_events())

    utils.integrate_events_file_to_frames_file_by_fixed_frames_number(
        lambda path: dict(np.load(path)),
        str(events_file),
        str(tmp_path),
        split_by="number",
        frames_num=2,
        H=2,
        W=2,
    )

    with np.load(events_file) as archive:
        frames = archive["frames"]
    assert frames.shape == (2, 2, 2, 2)
    assert frames.sum() == 4


def test_padding_collate_and_mask_share_sequence_length_contract():
    padded, labels, lengths = utils.pad_sequence_collate(
        [(np.ones((2, 3)), 4), (np.full((4, 3), 2), 7)]
    )

    assert padded.shape == (2, 4, 3)
    assert torch.equal(labels, torch.tensor([4, 7]))
    assert torch.equal(lengths, torch.tensor([2, 4]))
    assert torch.equal(
        utils.padded_sequence_mask(lengths),
        torch.tensor(
            [
                [True, True],
                [True, True],
                [False, True],
                [False, True],
            ]
        ),
    )


def test_fast_and_regular_class_splits_produce_the_same_indices():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(12), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    )

    regular = utils.split_to_train_test_set(0.5, dataset, num_classes=3)
    fast = utils.fast_split_to_train_test_set(0.5, dataset, num_classes=3, batch_size=2)

    assert regular[0].indices == fast[0].indices
    assert regular[1].indices == fast[1].indices


def test_event_binary_loaders_decode_documented_bit_layouts(tmp_path):
    timestamp = 0x12345
    atis_file = tmp_path / "events.bin"
    atis_file.write_bytes(
        bytes([3, 4, 0x80 | (timestamp >> 16), timestamp >> 8 & 0xFF, timestamp & 0xFF])
    )

    atis = utils.load_ATIS_bin(atis_file)
    assert {key: value.tolist() for key, value in atis.items()} == {
        "t": [timestamp],
        "x": [3],
        "y": [4],
        "p": [1],
    }

    aedat_file = tmp_path / "events.aedat"
    address = (5 << 17) | (6 << 2) | (1 << 1)
    packet_header = struct.pack("HHIIIIII", 1, 0, 8, 0, 0, 1, 1, 1)
    aedat_file.write_bytes(
        b"#!END-HEADER\r\n" + packet_header + struct.pack("II", address, 17)
    )

    aedat = utils.load_aedat_v3(aedat_file)
    assert {key: value.tolist() for key, value in aedat.items()} == {
        "t": [17],
        "x": [5],
        "y": [6],
        "p": [1],
    }


def test_load_npz_frames_returns_float32(tmp_path):
    frames_file = tmp_path / "frames.npz"
    np.savez(frames_file, frames=np.arange(4, dtype=np.uint8))

    frames = utils.load_npz_frames(frames_file)

    assert frames.dtype == np.float32
    assert np.array_equal(frames, np.arange(4, dtype=np.float32))


def test_create_sub_dataset_preserves_leaf_structure_and_ratio(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for class_name in ("a", "b"):
        class_dir = source / class_name
        class_dir.mkdir(parents=True)
        for index in range(4):
            (class_dir / f"{index}.dat").write_text(str(index))

    utils.create_sub_dataset(str(source), str(target), ratio=0.5, use_soft_link=False)

    for class_name in ("a", "b"):
        copied_files = list((target / class_name).glob("*.dat"))
        assert len(copied_files) == 2
        assert all(path.is_file() and not path.is_symlink() for path in copied_files)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("as_array", [np.asarray, torch.as_tensor])
def test_random_temporal_delete_preserves_type_order_and_layout(
    monkeypatch, batch_first, as_array
):
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([3, 1]))
    sequence = as_array(np.arange(8).reshape(4, 2))
    if batch_first:
        sequence = sequence.T

    actual = transform.random_temporal_delete(
        sequence, T_remain=2, batch_first=batch_first
    )
    expected = sequence[:, [1, 3]] if batch_first else sequence[[1, 3]]

    assert type(actual) is type(sequence)
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    else:
        assert np.array_equal(actual, expected)


def test_random_temporal_delete_module_uses_configured_time_layout(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([2, 0]))

    actual = transform.RandomTemporalDelete(T_remain=2, batch_first=True)(
        torch.arange(8).reshape(2, 4)
    )

    assert torch.equal(actual, torch.tensor([[0, 2], [4, 6]]))
