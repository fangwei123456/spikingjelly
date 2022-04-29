import torch
import torchvision
from tqdm import tqdm
import spikingjelly.clock_driven.ann2snn as ann2snn
from spikingjelly.clock_driven.ann2snn.sample_models import cifar10_resnet


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
        acc = correct / total
        print('Validating Accuracy: %.3f' % (acc))
    return acc

def main():
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:9'
    dataset_dir = '~/dataset/cifar10'
    batch_size = 100
    T = 400

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    model = cifar10_resnet.ResNet18()
    model.load_state_dict(torch.load('SJ-cifar10-resnet18_model-sample.pth'))

    train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform,
        download=True)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform,
        download=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=False)

    print('ANN accuracy:')
    val(model, device, test_data_loader)
    print('Converting...')
    model_converter = ann2snn.Converter(device=device,mode='Max', dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('SNN accuracy:')
    val(snn_model, device, test_data_loader, T=T)

if __name__ == '__main__':
    print('Downloading SJ-cifar10-resnet18_model-sample.pth')
    ann2snn.download_url("https://ndownloader.figshare.com/files/26676110",'./SJ-cifar10-resnet18_model-sample.pth')
    main()

