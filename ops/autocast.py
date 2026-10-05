import torch


def _register_autocast(qualified_name):
    namespace, name = qualified_name.split("::")
    operator = getattr(getattr(torch.ops, namespace), name).default
    libraries = []

    def kernel(*args, **kwargs):
        device_type = args[0].device.type
        dtype = torch.get_autocast_dtype(device_type)

        def cast(value):
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == device_type
                and value.is_floating_point()
                and value.dtype != torch.float64
            ):
                return value.to(dtype)
            return value

        # Cast before the custom-op autograd boundary so compilation sees the
        # casts and master parameters retain their gradient edges.
        with (
            torch.autocast("cpu", enabled=False),
            torch.autocast("cuda", enabled=False),
        ):
            return operator(
                *(cast(value) for value in args),
                **{key: cast(value) for key, value in kwargs.items()},
            )

    for key in ("AutocastCPU", "AutocastCUDA"):
        library = torch.library.Library(namespace, "IMPL", key)
        library.impl(name, kernel)
        libraries.append(library)
    return libraries
