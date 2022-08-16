import torch
import torch.nn as nn
import argparse
from spikingjelly.activation_based import lynxi_exchange
from spikingjelly.activation_based.examples import conv_fashion_mnist
import torchvision
import tqdm

'''
python w1.py -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -cupy -opt sgd -lr 0.1 -j 8

Namespace(T=4, device='cuda:0', b=128, epochs=64, j=8, data_dir='/datasets/FashionMNIST/', out_dir='./logs', resume=None, amp=False, cupy=True, opt='sgd', momentum=0.9, lr=0.1, channels=128, save_es=None)
./logs/T4_b128_sgd_lr0.1_c128_cupy
epoch = 63, train_loss = 0.0041, train_acc = 0.9836, test_loss = 0.0110, test_acc = 0.9312, max_test_acc = 0.9330
train speed = 8056.0318 images/s, test speed = 11152.5812 images/s
escape time = 2022-08-16 10:52:51

'''


class InferenceNet(nn.Module):
    def __init__(self, T: int, modules_list: list):
        super().__init__()
        self.T = T
        self.module_list = nn.Sequential(*modules_list)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.repeat(self.T, 1, 1, 1)

        # [N, C, H, W] -> [T, N, C, H, W]
        x = self.module_list(x)

        # [TN, *] -> [T, N, *]
        x = x.reshape(self.T, x.shape[0] // self.T, -1)

        return x.mean(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on Lynxi chips')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-pt-path', default='/home/cxhpc/fangwei/tempdir/fmnist_test/logs/T4_b128_sgd_lr0.1_c128_cupy/checkpoint_max.pth', type=str, help='checkpoint file path for conv_fashion_mnist.CSNN')
    parser.add_argument('-out-model-path', default='/home/cxhpc/fangwei/tempdir/fmnist_test/lynxi_model', type=str, help='path for saving the model compiled by lynxi')
    parser.add_argument('-lynxi-device', default=0, type=int, help='device id for lynxi')

    args = parser.parse_args()
    print(args)
    batch_size = args.b
    net_sj = conv_fashion_mnist.CSNN(T=args.T, channels=args.channels)
    net_sj.eval()
    lynxi_model_path = args.out_model_path
    device_id = args.lynxi_device

    ckp = torch.load(args.pt_path, map_location='cpu')
    print(f'max_test_acc={ckp["max_test_acc"]}')

    net_sj.load_state_dict(ckp['net'])

    module_list = lynxi_exchange.to_lynxi_supported_modules(net_sj.conv_fc, args.T)
    net = InferenceNet(args.T, module_list)

    net.eval()
    print(net)

    output_path = lynxi_exchange.compile_lynxi_model(lynxi_model_path, net, in_data_type='float16',
                                                     out_data_type='float16',
                                                     input_shape_dict={'x': torch.Size([batch_size, 1, 28, 28])})

    net_lynxi = lynxi_exchange.load_lynxi_model(device_id, output_path)

    test_set = torchvision.datasets.FashionMNIST(
        root='/home/cxhpc/fangwei/FashionMNIST',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.j
    )

    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for img, label in tqdm.tqdm(test_data_loader, disable=False):
            img = img.half()
            y = net_lynxi(lynxi_exchange.torch_tensor_to_lynxi(img, device_id))
            y = lynxi_exchange.lynxi_tensor_to_torch(y, shape=[label.shape[0], 10], dtype='float16')
            test_acc += (y.argmax(1) == label).half().sum().item()
            test_samples += img.shape[0]

    test_acc = test_acc / test_samples
    print(f'lynxi inference accuracy = {test_acc}')


