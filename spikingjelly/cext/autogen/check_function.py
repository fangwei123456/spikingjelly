def check_cuda_and_contiguous(x: str):
    return f'TORCH_CHECK({x}.device().is_cuda(),"{x} must be a CUDA tensor");' \
           f'if (! {x}.is_contiguous()){{{x} = {x}.contiguous();}}'

def check_cuda_operation(operation: str):
    return f'if({operation} != cudaSuccess)' \
           f'{{printf("CUDA error: %s\\n", cudaGetErrorString(cudaGetLastError()));exit(-1);}}'

