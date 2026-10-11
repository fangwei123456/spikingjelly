from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from spikingjelly.datasets import base
from spikingjelly.datasets.es_imagenet import (
    ESImageNetFrameFixedNumberBuilder,
)
from spikingjelly.datasets.shd import (
    SHDFrameFixedNumberBuilder,
    SpikingHeidelbergDigits,
)
from spikingjelly.datasets.speechcommands import SpeechCommands


def _fixed_number_config(root: Path) -> base.NeuromorphicDatasetConfig:
    return base.NeuromorphicDatasetConfig(
        root=root,
        train=True,
        data_type="frame",
        frames_number=2,
        split_by="number",
    )


def test_frame_builder_preserves_directory_structure_and_event_loader(
    tmp_path, monkeypatch
):
    root = tmp_path / "dataset"
    raw_root = root / "events"
    sample_dir = raw_root / "class0"
    sample_dir.mkdir(parents=True)
    sample_file = sample_dir / "sample.npz"
    np.savez(
        sample_file,
        x=np.array([0, 1]),
        y=np.array([0, 1]),
        t=np.array([0, 1]),
        p=np.array([0, 1]),
    )
    expected = np.arange(16).reshape(2, 2, 2, 2)

    def integrate(event_loader, events_file, output_dir, **_):
        assert set(event_loader(events_file)) == {"x", "y", "t", "p"}
        np.savez(output_dir / events_file.name, frames=expected)

    monkeypatch.setattr(
        base.utils,
        "integrate_events_file_to_frames_file_by_fixed_frames_number",
        integrate,
    )
    processed_root, loader = base.FrameFixedNumberBuilder(
        _fixed_number_config(root), raw_root, 2, 2
    ).build()

    output_file = processed_root / "class0" / "sample.npz"
    assert np.array_equal(loader(output_file), expected)
    assert not (root / "frames_number_2_split_by_number.building").exists()


def test_event_loader_supports_structured_npy(tmp_path):
    events = np.array([(1, 2)], dtype=[("t", "i8"), ("x", "i8")])
    sample = tmp_path / "sample.npy"
    np.save(sample, events)
    cfg = base.NeuromorphicDatasetConfig(tmp_path, True, data_type="event")

    loaded = base.EventBuilder(cfg, tmp_path).get_loader()(sample)

    assert np.array_equal(loaded, events)


def test_custom_frame_builder_saves_integrated_frames(tmp_path):
    root = tmp_path / "dataset"
    raw_root = root / "events"
    sample_dir = raw_root / "class0"
    sample_dir.mkdir(parents=True)
    np.savez(sample_dir / "sample.npz", x=np.array([1]))
    expected = np.arange(8).reshape(1, 2, 2, 2)
    cfg = base.NeuromorphicDatasetConfig(
        root=root,
        train=None,
        data_type="frame",
        custom_integrate_function=lambda events, height, width: expected,
    )

    processed_root, loader = base.FrameCustomIntegrateBuilder(
        cfg, raw_root, 2, 2
    ).build()

    assert np.array_equal(loader(processed_root / "class0" / "sample.npz"), expected)


def test_interrupted_frame_build_is_not_reused(tmp_path):
    root = tmp_path / "dataset"
    raw_root = root / "events"
    sample_dir = raw_root / "class0"
    sample_dir.mkdir(parents=True)
    np.savez(sample_dir / "sample.npz", x=np.array([1]))

    def fail(*_):
        raise RuntimeError("integration failed")

    cfg = base.NeuromorphicDatasetConfig(
        root=root,
        train=None,
        data_type="frame",
        custom_integrate_function=fail,
        custom_integrated_frames_dir_name="processed",
    )
    builder = base.FrameCustomIntegrateBuilder(cfg, raw_root, 2, 2)

    with pytest.raises(RuntimeError, match="integration failed"):
        builder.build()

    assert (root / "processed.building").exists()
    with pytest.raises(RuntimeError, match="unfinished"):
        builder.build()


def test_interrupted_raw_preparation_is_not_reused(tmp_path):
    class FailingDataset(base.NeuromorphicDatasetFolder):
        @classmethod
        def get_H_W(cls):
            return 2, 2

        @classmethod
        def resource_url_md5(cls):
            return []

        @classmethod
        def downloadable(cls):
            return True

        @classmethod
        def extract_downloaded_files(cls, download_root, extract_root):
            pass

        @classmethod
        def create_raw_from_extracted(cls, extract_root, raw_root):
            raise RuntimeError("conversion failed")

    with pytest.raises(RuntimeError, match="conversion failed"):
        FailingDataset(tmp_path)

    assert (tmp_path / "events_np.building").exists()
    with pytest.raises(RuntimeError, match="unfinished"):
        FailingDataset(tmp_path)


def test_es_imagenet_builder_only_overrides_event_format(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    raw_root = root / "events"
    sample_dir = raw_root / "class0"
    sample_dir.mkdir(parents=True)
    sample_file = sample_dir / "sample.npz"
    np.savez(
        sample_file,
        pos=np.array([[0, 1, 0], [1, 0, 2]]),
        neg=np.array([[1, 1, 1]]),
    )
    expected = np.ones((2, 2, 2, 2))

    def integrate(event_loader, events_file, output_dir, **_):
        assert set(event_loader(events_file)) == {"x", "y", "t", "p"}
        np.savez(output_dir / events_file.name, frames=expected)

    monkeypatch.setattr(
        base.utils,
        "integrate_events_file_to_frames_file_by_fixed_frames_number",
        integrate,
    )
    processed_root, loader = ESImageNetFrameFixedNumberBuilder(
        _fixed_number_config(root), raw_root, 2, 2
    ).build()

    assert np.array_equal(loader(processed_root / "class0" / "sample.npz"), expected)


def test_shd_builder_reuses_common_split_and_thread_lifecycle(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    raw_root = root / "events"
    raw_root.mkdir(parents=True)
    with h5py.File(raw_root / "shd_train.h5", "w") as h5_file:
        h5_file.create_dataset("labels", data=np.array([1, 0]))

    def integrate(
        h5_file,
        sample_index,
        output_dir,
        split_by,
        frames_num,
        W,
    ):
        label = int(h5_file["labels"][sample_index])
        np.savez(
            output_dir / str(label) / str(sample_index),
            frames=np.full((frames_num, W), sample_index),
        )

    monkeypatch.setattr(
        "spikingjelly.datasets.shd."
        "_integrate_events_file_to_frames_file_by_fixed_frames_number",
        integrate,
    )
    processed_root, loader = SHDFrameFixedNumberBuilder(
        _fixed_number_config(root),
        raw_root,
        W=4,
        splits=("train",),
        n_classes=2,
    ).build()

    for sample_index, label in enumerate((1, 0)):
        frames = loader(processed_root / "train" / str(label) / f"{sample_index}.npz")
        assert np.array_equal(frames, np.full((2, 4), sample_index))


def test_shd_event_dataset_works_with_spawn_workers(tmp_path):
    raw_root = tmp_path / "events_h5"
    raw_root.mkdir()
    with h5py.File(raw_root / "shd_train.h5", "w") as h5_file:
        spikes = h5_file.create_group("spikes")
        spikes.create_dataset("times", data=np.array([[0.1, 0.2], [0.3, 0.4]]))
        spikes.create_dataset("units", data=np.array([[1, 2], [3, 4]]))
        h5_file.create_dataset("labels", data=np.array([1, 0]))

    dataset = SpikingHeidelbergDigits(tmp_path, train=True, data_type="event")
    dataset[0]
    old_h5_file = dataset.h5_file
    dataset.h5_file_pid = -1
    dataset[0]
    assert not old_h5_file.id.valid
    samples = list(
        DataLoader(
            dataset,
            batch_size=None,
            num_workers=1,
            multiprocessing_context="spawn",
        )
    )

    assert len(samples) == 2
    assert [sample[1].item() for sample in samples] == [1, 0]


def test_speechcommands_loads_waveform_and_label(tmp_path, monkeypatch):
    dataset = SpeechCommands.__new__(SpeechCommands)
    dataset._path = str(tmp_path)
    dataset._walker = ["yes/speaker_nohash_0.wav"]
    dataset.label_dict = {"yes": 3}
    dataset.transform = None
    dataset.silence_cnt = 0
    monkeypatch.setattr(
        "spikingjelly.datasets.speechcommands.torchaudio.load",
        lambda _: (torch.tensor([[-2.0, 1.0]]), 16000),
    )

    waveform, label = dataset[0]

    assert torch.equal(waveform, torch.tensor([[-1.0, 0.5]]))
    assert label == 3


def test_speechcommands_discovers_noise_after_download(tmp_path, monkeypatch):
    def download(_, root, **__):
        (Path(root) / "archive.tar.gz").touch()

    def extract(_, path):
        path = Path(path)
        noise_dir = path / "_background_noise_"
        noise_dir.mkdir(parents=True)
        (noise_dir / "noise.wav").touch()
        (path / "testing_list.txt").touch()

    monkeypatch.setattr("spikingjelly.datasets.speechcommands.download_url", download)
    monkeypatch.setattr("spikingjelly.datasets.speechcommands.extract_archive", extract)

    dataset = SpeechCommands(
        {"_silence_": 0},
        str(tmp_path),
        silence_cnt=1,
        url="https://example.com/archive.tar.gz",
        split="test",
        download=True,
    )

    assert dataset.noise_list == [
        str(
            tmp_path / "SpeechCommands" / "archive" / "_background_noise_" / "noise.wav"
        )
    ]
