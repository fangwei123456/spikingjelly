import os

import torch
import torchvision
from tqdm import tqdm
import spikingjelly.activation_based.ann2snn as ann2snn
from spikingjelly.activation_based.ann2snn.sample_models import cifar10_resnet


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    if T is not None and T <= 0:
        raise ValueError("T must be positive.")
    reset_modules = [m for m in net.modules() if hasattr(m, "reset")] if T else []
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            label = label.to(device)
            if T is None:
                out = net(img)
            else:
                for m in reset_modules:
                    m.reset()
                out = net(img)
                for t in range(1, T):
                    out += net(img)
            correct += (out.argmax(dim=1) == label).float().sum().item()
            total += out.shape[0]
        acc = correct / total
        print("Validating Accuracy: %.3f" % (acc))
    return acc


def main(checkpoint_path):
    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = os.path.expanduser("~/dataset/cifar10")
    batch_size = 100
    T = 400

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    model = cifar10_resnet.ResNet18()
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, transform=transform, download=True
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False, transform=transform, download=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset, batch_size=50, shuffle=False, drop_last=False
    )

    print("ANN accuracy:")
    val(model, device, test_data_loader)
    print("Converting...")
    model_converter = ann2snn.Converter(
        recipe=ann2snn.RateCodingRecipe(dataloader=train_data_loader, mode="Max"),
        device=device,
    )
    snn_model = model_converter.convert(model)
    print("SNN accuracy:")
    val(snn_model, device, test_data_loader, T=T)


if __name__ == "__main__":
    checkpoint_path = "./SJ-cifar10-resnet18_model-sample.pth"
    print("Downloading SJ-cifar10-resnet18_model-sample.pth")
    ann2snn.download_url(
        "https://ndownloader.figshare.com/files/26676110",
        checkpoint_path,
    )
    expected_min_size = 1024 * 1024
    if (
        not os.path.isfile(checkpoint_path)
        or os.path.getsize(checkpoint_path) < expected_min_size
    ):
        raise RuntimeError(
            f"Checkpoint download failed or is truncated: {checkpoint_path}"
        )
    main(checkpoint_path)
