import argparse
import copy
import json
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision
from tqdm import tqdm

from spikingjelly.activation_based import ann2snn
from spikingjelly.activation_based.ann2snn.sample_models import mnist_cnn


DEFAULT_CHECKPOINT_URL = "https://ndownloader.figshare.com/files/34960191"
DEFAULT_CHECKPOINT_PATH = "SJ-mnist-cnn_model-sample.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ann2snn conversion recipes on the MNIST CNN example."
    )
    parser.add_argument("--dataset-dir", default="./data/mnist")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--test-batch-size", type=int, default=50)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--plot-mode-sweep",
        action="store_true",
        help="Run the legacy max/ratio mode sweep and show the accuracy plot.",
    )
    return parser.parse_args()


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    if T is not None and T <= 0:
        raise ValueError("T must be positive.")
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
        reset_modules = [m for m in net.modules() if hasattr(m, "reset")]
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label).float().sum().item()
            else:
                for m in reset_modules:
                    m.reset()
                out = None
                for t in range(T):
                    step = net(img)
                    out = step if out is None else out + step
                    corrects[t] += (out.argmax(dim=1) == label).float().sum().item()
            total += out.shape[0]
    return correct / total if T is None else corrects / total


def save_results(results, output_path):
    if output_path is None:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def load_ann(device, checkpoint_path):
    model = mnist_cnn.CNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def download_checkpoint(checkpoint_url, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {checkpoint_path}...")
    try:
        ann2snn.download_url(checkpoint_url, str(checkpoint_path))
    except KeyError as exc:
        if exc.args != ("content-length",):
            raise
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) "
                "Gecko/20100101 Firefox/67.0"
            )
        }
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            response = requests.get(
                checkpoint_url, headers=headers, stream=True, timeout=30
            )
            response.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(checkpoint_path)
        except (requests.RequestException, OSError):
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def convert_and_eval(recipe, device, ann_model, test_data_loader, time_steps):
    model_converter = ann2snn.Converter(recipe=recipe, device=device)
    snn_model = model_converter.convert(copy.deepcopy(ann_model))
    return val(snn_model, device, test_data_loader, T=time_steps)


def run_recipe_comparison(
    device, calibration_data_loader, test_data_loader, time_steps, checkpoint_path
):
    results = {}

    model = load_ann(device, checkpoint_path)
    ann_acc = val(model, device, test_data_loader)
    print("ANN Validating Accuracy: %.4f" % ann_acc)
    results["ann"] = {"top1": ann_acc * 100.0}

    print("---------------------------------------------")
    print("Converting using RobustNorm scalar thresholds")
    robust_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(
            dataloader=calibration_data_loader,
            mode="99.9%",
        ),
        device,
        model,
        test_data_loader,
        time_steps,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f" % (time_steps, robust_accs[-1])
    )
    results[f"robust_scalar_t{time_steps}"] = {
        "time_steps": time_steps,
        "top1": robust_accs[-1] * 100.0,
    }

    print("---------------------------------------------")
    print("Converting using LocalThresholdBalancingRecipe")
    ltb_accs = convert_and_eval(
        ann2snn.LocalThresholdBalancingRecipe(
            dataloader=calibration_data_loader,
            time_steps=time_steps,
            mode="99.9%",
        ),
        device,
        model,
        test_data_loader,
        time_steps,
    )
    print("SNN accuracy (simulation %d time-steps): %.4f" % (time_steps, ltb_accs[-1]))
    results[f"ltb_t{time_steps}"] = {
        "time_steps": time_steps,
        "top1": ltb_accs[-1] * 100.0,
    }

    return results


def run_legacy_mode_sweep(
    device, calibration_data_loader, test_data_loader, time_steps, checkpoint_path
):
    import matplotlib.pyplot as plt

    print("---------------------------------------------")
    print("Converting using MaxNorm")
    mode_max_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="max"),
        device,
        test_data_loader,
        time_steps,
        checkpoint_path,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f"
        % (time_steps, mode_max_accs[-1])
    )

    print("---------------------------------------------")
    print("Converting using RobustNorm")
    mode_robust_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode="99.9%"),
        device,
        test_data_loader,
        time_steps,
        checkpoint_path,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f"
        % (time_steps, mode_robust_accs[-1])
    )

    print("---------------------------------------------")
    print("Converting using 1/2 max(activation) as scales...")
    mode_two_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 2),
        device,
        test_data_loader,
        time_steps,
        checkpoint_path,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f"
        % (time_steps, mode_two_accs[-1])
    )

    print("---------------------------------------------")
    print("Converting using 1/3 max(activation) as scales")
    mode_three_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 3),
        device,
        test_data_loader,
        time_steps,
        checkpoint_path,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f"
        % (time_steps, mode_three_accs[-1])
    )

    print("---------------------------------------------")
    print("Converting using 1/4 max(activation) as scales")
    mode_four_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 4),
        device,
        test_data_loader,
        time_steps,
        checkpoint_path,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f"
        % (time_steps, mode_four_accs[-1])
    )

    print("---------------------------------------------")
    print("Converting using 1/5 max(activation) as scales")
    mode_five_accs = convert_and_eval(
        ann2snn.RateCodingRecipe(dataloader=calibration_data_loader, mode=1.0 / 5),
        device,
        test_data_loader,
        time_steps,
        checkpoint_path,
    )
    print(
        "SNN accuracy (simulation %d time-steps): %.4f"
        % (time_steps, mode_five_accs[-1])
    )

    plt.figure()
    plt.plot(np.arange(0, time_steps), mode_max_accs, label="mode: max")
    plt.plot(np.arange(0, time_steps), mode_robust_accs, label="mode: 99.9%")
    plt.plot(np.arange(0, time_steps), mode_two_accs, label="mode: 1.0/2")
    plt.plot(np.arange(0, time_steps), mode_three_accs, label="mode: 1.0/3")
    plt.plot(np.arange(0, time_steps), mode_four_accs, label="mode: 1.0/4")
    plt.plot(np.arange(0, time_steps), mode_five_accs, label="mode: 1.0/5")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Acc")
    plt.show()


def main(args):
    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = args.device
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    T = args.time_steps

    train_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    calibration_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # for epoch in range(epochs):
    #     model.train()
    #     for (img, label) in train_data_loader:
    #         optimizer.zero_grad()
    #         out = model(img.to(device))
    #         loss = loss_function(out, label.to(device))
    #         loss.backward()
    #         optimizer.step()
    #     torch.save(model.state_dict(), 'SJ-mnist-cnn_model-sample.pth')
    #     print('Epoch: %d' % epoch)
    #     acc = val(model, device, train_data_loader)
    #     print('Validating Accuracy: %.3f' % (acc))
    #     print()

    if args.plot_mode_sweep:
        run_legacy_mode_sweep(
            device, calibration_data_loader, test_data_loader, T, args.checkpoint_path
        )
    else:
        metrics = run_recipe_comparison(
            device, calibration_data_loader, test_data_loader, T, args.checkpoint_path
        )
        results = {
            "dataset": "MNIST",
            "train_samples": len(train_data_dataset),
            "test_samples": len(test_data_dataset),
            "batch_size": batch_size,
            "test_batch_size": args.test_batch_size,
            "time_steps": T,
            "device": device,
            "metrics": metrics,
        }
        save_results(results, args.output)
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    args = parse_args()
    download_checkpoint(args.checkpoint_url, args.checkpoint_path)
    main(args)
