from pathlib import Path

import h5py
import numpy as np

from spikingjelly.datasets import base
from spikingjelly.datasets.es_imagenet import (
    ESImageNetFrameFixedNumberBuilder,
)
from spikingjelly.datasets.shd import SHDFrameFixedNumberBuilder


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

    def integrate(event_loader, events_file, output_dir, *_):
        assert set(event_loader(events_file).files) == {"x", "y", "t", "p"}
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

    def integrate(event_loader, events_file, output_dir, *_):
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
        frames_number,
        width,
        print_save,
    ):
        label = int(h5_file["labels"][sample_index])
        np.savez(
            output_dir / str(label) / str(sample_index),
            frames=np.full((frames_number, width), sample_index),
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
