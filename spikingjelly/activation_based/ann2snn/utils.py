import os
import re

import requests
from tqdm import tqdm


def _validate_download_response(req, first_byte, file_size):
    if req.status_code not in (200, 206):
        req.raise_for_status()
    if req.status_code == 200:
        if first_byte != 0:
            return False
        return True
    content_range = req.headers.get("Content-Range", "")
    match = re.match(r"bytes (\d+)-(\d+)/(\d+)", content_range, re.IGNORECASE)
    return (
        match is not None
        and int(match.group(1)) == first_byte
        and int(match.group(2)) == file_size - 1
        and int(match.group(3)) == file_size
    )


def _download_without_resume(url, dst, headers):
    pbar = tqdm(unit="B", unit_scale=True, desc=dst)
    req = requests.get(url, headers=headers, stream=True, timeout=30)
    try:
        req.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    finally:
        req.close()
        pbar.close()
    return os.path.getsize(dst)


def download_url(url, dst):
    r"""
    **API Language** - :ref:`中文 <download_url-cn>` | :ref:`English <download_url-en>`

    ----

    .. _download_url-cn:

    * **中文**

    从指定 URL 下载文件并保存到目标路径。支持断点续传。

    :param url: 文件的下载链接
    :type url: str

    :param dst: 保存文件的目标路径
    :type dst: str

    :return: 文件的总大小(以字节为单位)
    :rtype: int

    ----

    .. _download_url-en:

    * **English**

    Download a file from a given URL and save it to a destination path. Supports resuming interrupted downloads.

    :param url: the download URL of the file
    :type url: str

    :param dst: the destination path to save the file
    :type dst: str

    :return: the total file size in bytes
    :rtype: int
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0"
    }

    response = requests.get(url, headers=headers, stream=True, timeout=30)  # (1)
    try:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
    finally:
        response.close()
    if content_length is None:
        return _download_without_resume(url, dst, headers)
    file_size = int(content_length)  # (2)
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)  # (3)
    else:
        first_byte = 0
    if first_byte >= file_size:  # (4)
        return file_size

    header = {"Range": f"bytes={first_byte}-{file_size - 1}"}

    pbar = tqdm(
        total=file_size, initial=first_byte, unit="B", unit_scale=True, desc=dst
    )
    req = requests.get(url, headers=header, stream=True, timeout=30)  # (5)
    try:
        mode = "ab"
        valid_response = _validate_download_response(req, first_byte, file_size)
        if first_byte > 0 and not valid_response:
            req.close()
            first_byte = 0
            pbar.reset(total=file_size)
            header = {"Range": f"bytes=0-{file_size - 1}"}
            req = requests.get(url, headers=header, stream=True, timeout=30)
            valid_response = _validate_download_response(req, first_byte, file_size)
            mode = "wb"
        if not valid_response:
            raise RuntimeError("Invalid response for download.")
        with open(dst, mode) as f:
            for chunk in req.iter_content(chunk_size=1024):  # (6)
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    finally:
        req.close()
        pbar.close()
    return file_size
