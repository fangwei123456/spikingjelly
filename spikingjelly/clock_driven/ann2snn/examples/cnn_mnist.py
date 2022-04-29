import torch
import torchvision
import torch.nn as nn
import spikingjelly
from spikingjelly.clock_driven import ann2snn
from tqdm import tqdm
from spikingjelly.clock_driven.ann2snn.sample_models import mnist_cnn
import numpy as np
import matplotlib.pyplot as plt

def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
    return correct / total if T is None else corrects / total

def main():
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda'
    dataset_dir = 'G:/Dataset/mnist'
    batch_size = 100
    T = 50
    # 训练参数
    lr = 1e-3
    epochs = 10

    model = mnist_cnn.CNN().to(device)
    train_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    test_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=False)

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

    model.load_state_dict(torch.load('SJ-mnist-cnn_model-sample.pth'))
    acc = val(model, device, test_data_loader)
    print('ANN Validating Accuracy: %.4f' % (acc))

    print('---------------------------------------------')
    print('Converting using MaxNorm')
    model_converter = ann2snn.Converter(mode='max', dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_max_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_max_accs[-1]))

    print('---------------------------------------------')
    print('Converting using RobustNorm')
    model_converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_robust_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_robust_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/2 max(activation) as scales...')
    model_converter = ann2snn.Converter(mode=1.0 / 2, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_two_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_two_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/3 max(activation) as scales')
    model_converter = ann2snn.Converter(mode=1.0 / 3, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_three_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_three_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/4 max(activation) as scales')
    model_converter = ann2snn.Converter(mode=1.0 / 4, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_four_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_four_accs[-1]))

    print('---------------------------------------------')
    print('Converting using 1/5 max(activation) as scales')
    model_converter = ann2snn.Converter(mode=1.0 / 5, dataloader=train_data_loader)
    snn_model = model_converter(model)
    print('Simulating...')
    mode_five_accs = val(snn_model, device, test_data_loader, T=T)
    print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_five_accs[-1]))

    fig = plt.figure()
    plt.plot(np.arange(0, T), mode_max_accs, label='mode: max')
    plt.plot(np.arange(0, T), mode_robust_accs, label='mode: 99.9%')
    plt.plot(np.arange(0, T), mode_two_accs, label='mode: 1.0/2')
    plt.plot(np.arange(0, T), mode_three_accs, label='mode: 1.0/3')
    plt.plot(np.arange(0, T), mode_four_accs, label='mode: 1.0/4')
    plt.plot(np.arange(0, T), mode_five_accs, label='mode: 1.0/5')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Acc')
    plt.show()


if __name__ == '__main__':
    print('Downloading SJ-mnist-cnn_model-sample.pth...')
    ann2snn.download_url("https://ndownloader.figshare.com/files/34960191", './SJ-mnist-cnn_model-sample.pth')
    main()
