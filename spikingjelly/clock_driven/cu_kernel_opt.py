try:
    import cupy
    import torch
    nvcc_options = ('--use_fast_math',)

    threads = 1024

    def cal_blocks(numel: int):
        return (numel + threads - 1) // threads

    def wrap_args_to_raw_kernel(device: int, args_list: list):
        # check device and contiguous
        ret_list = []
        for item in args_list:
            if isinstance(item, torch.Tensor):
                assert item.get_device() == device
                item = item.contiguous()
                ret_list.append(item.data_ptr())
            elif isinstance(item, cupy.ndarray):
                assert item.device.id == device
                item = cupy.ascontiguousarray(item)
                ret_list.append(item)
            else:
                raise TypeError

        return tuple(ret_list)










except ImportError:
    pass