import torch
import torch.nn.functional as F
import torchvision.transforms
from torchvision.datasets import FashionMNIST
from matplotlib import pyplot as plt


train_set = FashionMNIST('D:/datasets/FashionMNIST', train=True, transform=torchvision.transforms.ToTensor())
to_img = torchvision.transforms.ToPILImage()
image, label = train_set[0]  # image.shape = [C, H, W]
image.unsqueeze_(0)
to_img(F.interpolate(image, scale_factor=4)[0]).save('./samples/origin.png')
for i in range(image.shape[3]):
    masked_image = torch.zeros_like(image)
    masked_image[..., i] = image[..., i]
    to_img(F.interpolate(masked_image, scale_factor=4)[0]).save(f'./samples/a/{i}.png')
    masked_image = image.clone()
    masked_image[..., i] = 0
    to_img(F.interpolate(masked_image, scale_factor=4)[0]).save(f'./samples/b/{i}.png')

