import requests
import os
from tqdm import tqdm

def download_url(url, dst):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0'
    }

    response = requests.get(url, headers=headers, stream=True)  # (1)
    file_size = int(response.headers['content-length']) # (2)
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)  # (3)
    else:
        first_byte = 0
    if first_byte >= file_size: # (4)
        return file_size

    header = {"Range": f"bytes={first_byte}-{file_size}"}

    pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=dst)
    req = requests.get(url, headers=header, stream=True)  # (5)
    with open(dst, 'ab') as f:
        for chunk in req.iter_content(chunk_size=1024):     # (6)
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size