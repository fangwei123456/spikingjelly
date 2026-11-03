import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from spikingjelly.datasets import base
from spikingjelly.datasets.shd import SpikingHeidelbergDigits, SpikingSpeechCommands


def _write_zipped_h5(root: Path, filename: str) -> None:
    """Mimic a zenkelab.org archive: ``x.h5.zip`` holds ``x.h5``."""
    h5_name = filename.removesuffix(".zip")
    h5_path = root / h5_name
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("labels", data=np.array([3, 7]))
        spikes = h5_file.create_group("spikes")
        times = spikes.create_dataset("times", (2,), dtype=h5py.vlen_dtype(np.float32))
        units = spikes.create_dataset("units", (2,), dtype=h5py.vlen_dtype(np.uint16))
        for i in range(2):
            times[i] = np.array([0.0, 0.1], dtype=np.float32)
            units[i] = np.array([1, 2], dtype=np.uint16)
    with zipfile.ZipFile(root / filename, "w") as archive:
        archive.write(h5_path, h5_name)
    h5_path.unlink()


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_ssc_resources_provide_the_file_of_each_split(split: str) -> None:
    """Each split has its SSC archive, and no SHD archive is listed."""
    names = [name for name, _, _ in SpikingSpeechCommands.resource_url_md5()]
    assert f"ssc_{split}.h5.zip" in names
    assert not any(name.startswith("shd_") for name in names)


def test_shd_and_ssc_resources_differ() -> None:
    """SHD and SSC do not share any archive URL."""
    shd_urls = {url for _, url, _ in SpikingHeidelbergDigits.resource_url_md5()}
    ssc_urls = {url for _, url, _ in SpikingSpeechCommands.resource_url_md5()}
    assert shd_urls.isdisjoint(ssc_urls)


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_ssc_download_and_load_with_mocked_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    """A fresh root downloads the three SSC archives and loads the split."""
    requested = []

    def fake_download_url(url, root, filename, md5=None):
        """Write a small archive instead of downloading it."""
        requested.append(filename)
        Path(root).mkdir(parents=True, exist_ok=True)
        _write_zipped_h5(Path(root), filename)

    monkeypatch.setattr(base, "download_url", fake_download_url)
    monkeypatch.setattr(
        base, "check_integrity", lambda fpath, md5=None: Path(fpath).exists()
    )

    dataset = SpikingSpeechCommands(root=str(tmp_path), split=split, data_type="event")

    assert sorted(requested) == [
        "ssc_test.h5.zip",
        "ssc_train.h5.zip",
        "ssc_valid.h5.zip",
    ]
    assert len(dataset) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_type": "event"},
        {"data_type": "frame", "frames_number": 2, "split_by": "number"},
    ],
    ids=["event", "frame"],
)
def test_ssc_root_without_ssc_files_raises_clear_error(
    tmp_path: Path, kwargs: dict
) -> None:
    """A root whose ``events_h5`` holds only SHD files asks for a separate root."""
    (tmp_path / "events_h5").mkdir()
    (tmp_path / "events_h5" / "shd_train.h5").touch()

    with pytest.raises(FileNotFoundError, match="separate, empty root"):
        SpikingSpeechCommands(root=str(tmp_path), split="valid", **kwargs)
