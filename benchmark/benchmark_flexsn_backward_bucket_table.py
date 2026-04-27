import statistics
import time
from collections import Counter, defaultdict

import torch

from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn, spiking_vgg16_bn
from spikingjelly.activation_based.neuron.flexsn import FlexSN
from spikingjelly.activation_based.triton_kernel.flexsn.wrapper import flexsn_backward_ncl_bucket


sg = surrogate.Sigmoid(alpha=4.0)
BUCKET_LABELS = {
    0: "small (<= 4096)",
    1: "medium (4097..131072)",
    2: "large (131073..1048576)",
    3: "xlarge (1048577..8388608)",
    4: "xxlarge (> 8388608)",
}


def lif_core_sg(x: torch.Tensor, v: torch.Tensor):
    tau, v_th = 2.0, 1.0
    h = v + (x - v) / tau
    s = sg(h - v_th)
    return s, h * (1.0 - s)


def make_flexsn(**kwargs):
    return FlexSN(
        core=lif_core_sg,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode=kwargs.get("step_mode", "m"),
        backend="inductor",
    )


def build_workloads():
    return [
        {
            "name": "cifar_vgg11_bn",
            "builder": lambda: spiking_vgg11_bn(pretrained=False, spiking_neuron=make_flexsn, num_classes=10, step_mode="m"),
            "input_shape": (4, 32, 3, 32, 32),
        },
        {
            "name": "imagenet_vgg11_bn",
            "builder": lambda: spiking_vgg11_bn(pretrained=False, spiking_neuron=make_flexsn, num_classes=1000, step_mode="m"),
            "input_shape": (4, 8, 3, 224, 224),
        },
        {
            "name": "imagenet_vgg16_bn",
            "builder": lambda: spiking_vgg16_bn(pretrained=False, spiking_neuron=make_flexsn, num_classes=1000, step_mode="m"),
            "input_shape": (4, 8, 3, 224, 224),
        },
    ]


def collect_flexsn_shapes(workload):
    model = workload["builder"]().cuda()
    functional.set_step_mode(model, "m")
    model.train()
    records = []
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, FlexSN):
            def _hook(mod, inputs, *, _name=name):
                if not inputs:
                    return
                x_seq = inputs[0]
                step_shape = tuple(x_seq[0].shape)
                records.append((
                    _name,
                    step_shape,
                    x_seq[0].numel(),
                    flexsn_backward_ncl_bucket(x_seq[0].numel()),
                ))
            hooks.append(module.register_forward_pre_hook(_hook, with_kwargs=False))
    x = torch.randn(workload["input_shape"], device="cuda", dtype=torch.float32)
    with torch.no_grad():
        functional.reset_net(model)
        model(x)
    for hook in hooks:
        hook.remove()
    return records


def bench_single_shape(step_shape, repeats=20, warmup=6):
    T = 4
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
    x_base = torch.randn((T, *step_shape), device="cuda", dtype=dtype)
    example_inputs = (
        torch.zeros(step_shape, device="cuda", dtype=dtype),
        torch.zeros(step_shape, device="cuda", dtype=dtype),
    )
    module = FlexSN(
        core=lif_core_sg,
        num_inputs=1,
        num_states=1,
        num_outputs=1,
        step_mode="m",
        backend="inductor",
        store_state_seqs=False,
        example_inputs=example_inputs,
    ).cuda()

    def step():
        module.states = None
        x = x_base.clone().requires_grad_(True)
        y = module(x)
        (y.sum() + module.states[0].sum()).backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available - skipping benchmark.")
        return
    torch.backends.cudnn.benchmark = True
    print("GPU", torch.cuda.get_device_name(0))
    print("bucket thresholds", BUCKET_LABELS)
    summary = defaultdict(list)
    for workload in build_workloads():
        records = collect_flexsn_shapes(workload)
        counts = Counter((shape, ncl, bucket) for _, shape, ncl, bucket in records)
        print(f"\n## {workload['name']}")
        print("| step_shape | NCL | bucket | layers | bench_mean_ms | bench_median_ms |")
        print("| --- | ---: | --- | ---: | ---: | ---: |")
        for (shape, ncl, bucket), layers in sorted(counts.items(), key=lambda item: item[0][1]):
            bench = bench_single_shape(shape, repeats=8, warmup=3)
            summary[bucket].append((workload["name"], shape, ncl, bench["mean_ms"]))
            print(
                f"| {shape} | {ncl} | {BUCKET_LABELS[bucket]} | {layers} | "
                f"{bench['mean_ms']:.3f} | {bench['median_ms']:.3f} |"
            )
    print("\n## Bucket Summary")
    print("| bucket | workload_count | max_ncl | min_ncl |")
    print("| --- | ---: | ---: | ---: |")
    for bucket, entries in sorted(summary.items()):
        ncls = [ncl for _, _, ncl, _ in entries]
        print(f"| {BUCKET_LABELS[bucket]} | {len(entries)} | {max(ncls)} | {min(ncls)} |")


if __name__ == "__main__":
    main()
