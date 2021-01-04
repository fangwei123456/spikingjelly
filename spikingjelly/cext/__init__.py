import torch
import time
import numpy as np
def cal_fun_t(n, device, f, *args, **kwargs):
    if n <= 2:
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        f(*args, **kwargs)
        torch.cuda.synchronize(device)
        return (time.perf_counter() - t_start)
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize(device)

    t_list = []
    for _ in range(n * 2):
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        f(*args, **kwargs)
        torch.cuda.synchronize(device)
        t_list.append(time.perf_counter() - t_start)
    t_list = np.asarray(t_list)
    return t_list[n:].mean()