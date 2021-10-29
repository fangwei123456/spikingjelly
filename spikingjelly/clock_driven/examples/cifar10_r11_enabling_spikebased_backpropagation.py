"""
.. codeauthor:: Yanqi Chen <chyq@pku.edu.cn>

A reproduction of the paper `Enabling Spike-Based Backpropagation for Training Deep Neural Network Architectures <https://doi.org/10.3389/fnins.2020.00119>`_\ .

This code reproduces a novel gradient-based training method of SNN. We to some extent refer to the network structure and some other detailed implementation in the `authors' implementation <https://github.com/chan8972/Enabling_Spikebased_Backpropagation>`_\ . Since the training method and neuron models are slightly different from which in this framework, we rewrite them in a compatible style.

Assuming you have at least 1 Nvidia GPU.
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from spikingjelly.clock_driven import functional, layer

from tqdm import tqdm
import math


parser = argparse.ArgumentParser(description='spikingjelly CIFAR10 Training')
parser.add_argument('data', metavar='DIR',
					help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
					metavar='N')
parser.add_argument('-T', '--timesteps', default=100, type=int,
					help='Simulation timesteps')
parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained parameters.')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')


#### Surrogate function ####
class relu(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		return (x > 0).float()

	@staticmethod
	def backward(ctx, grad_output):
		inputs = ctx.saved_tensors[0]
		grad_x = grad_output.clone()
		grad_x[inputs <= 0.0] = 0
		return grad_x


#### Neurons ####
class BaseNode(nn.Module):
	def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=relu.apply, monitor=False):
		super().__init__()
		self.v_threshold = v_threshold
		self.v_reset = v_reset
		if self.v_reset is None:
			self.v = 0
		else:
			self.v = self.v_reset
		self.surrogate_function = surrogate_function
		self.v_acc = 0 # Accumulated voltage (Assuming NO fire for this neuron)
		self.v_acc_l = 0 # Accumulated voltage with leaky (Assuming NO fire for this neuron)
		if monitor:
			self.monitor = {'v':[], 's':[]}
		else:
			self.monitor = False

		self.new_grad = None


	def spiking(self):
		spike = self.v - self.v_threshold
		self.v.masked_fill_(spike > 0, self.v_reset)
		spike = self.surrogate_function(spike)
		
		return spike

	def forward(self, dv: torch.Tensor):
		raise NotImplementedError

	def reset(self):
		if self.v_reset is None:
			self.v = 0
		else:
			self.v = self.v_reset
		if self.monitor:
			self.monitor = {'v': [], 's': []}
		self.v_acc = 0
		self.v_acc_l = 0


class LIFNode(BaseNode):
	def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=relu.apply, fire=True):
		super().__init__(v_threshold, v_reset, surrogate_function)
		self.tau = tau
		self.fire = fire # If no fire, the voltage threshold of neuron is infinity
		self.new_grad = None

	def forward(self, dv: torch.Tensor):
		self.v += dv
		if self.fire:
			spike = self.spiking()
			self.v_acc += spike
			self.v_acc_l = self.v - ((self.v - self.v_reset) / self.tau) + spike

		self.v = self.v - ((self.v - self.v_reset) / self.tau).detach()

		if self.fire:
			if self.training:
				spike.register_hook(lambda grad: torch.mul(grad, self.new_grad))
			return spike

		return self.v

class IFNode(BaseNode):
	def __init__(self, v_threshold=0.75, v_reset=0.0, surrogate_function=relu.apply):
		super().__init__(v_threshold, v_reset, surrogate_function)

	def forward(self, dv: torch.Tensor):
		self.v += dv
		return self.spiking()


#### Network ####
class ResNet11(nn.Module):
	def __init__(self):
		super().__init__()
		self.train_epoch = 0

		self.cnn11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.lif11 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)
		self.avgpool1 = nn.AvgPool2d(kernel_size=2)
		self.if1 = IFNode()

		self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.lif21 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)
		self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.shortcut1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),)
		self.lif2 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)

		self.cnn31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
		self.lif31 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)
		self.cnn32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
		self.shortcut2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),)
		self.lif3 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)

		self.cnn41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
		self.lif41 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)
		self.cnn42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
		self.shortcut3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),)
		self.lif4 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)

		self.cnn51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
		self.lif51 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)
		self.cnn52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
		self.shortcut4 = nn.Sequential(nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2), padding=(0, 0)))
		self.lif5 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)

		self.fc0 = nn.Linear(512 * 4 * 4, 1024, bias=False)
		self.lif6 = nn.Sequential(
			LIFNode(),
			layer.Dropout(0.25)
		)
		self.fc1 = nn.Linear(1024, 10, bias=False)
		self.lif_out = LIFNode(fire=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				variance1 = math.sqrt(1.0 / n)
				m.weight.data.normal_(0, variance1)

			elif isinstance(m, nn.Linear):
				size = m.weight.size()
				fan_in = size[1]
				variance2 = math.sqrt(1.0 / fan_in)
				m.weight.data.normal_(0.0, variance2)

	def forward(self, x):
		x = self.if1(self.avgpool1(self.lif11(self.cnn11(x))))
		x = self.lif2(self.cnn22(self.lif21(self.cnn21(x))) + self.shortcut1(x)) 
		x = self.lif3(self.cnn32(self.lif31(self.cnn31(x))) + self.shortcut2(x))
		x = self.lif4(self.cnn42(self.lif41(self.cnn41(x))) + self.shortcut3(x))
		x = self.lif5(self.cnn52(self.lif51(self.cnn51(x))) + self.shortcut4(x))

		out = x.view(x.size(0), -1)
		out = self.lif_out(self.fc1(self.lif6(self.fc0(out))))

		return out

	def reset_(self):
		for item in self.modules():
			if hasattr(item, 'reset'):
				item.reset()

def main():
	args = parser.parse_args()

	torch.cuda.set_device(args.gpu)  
	learning_rate = args.lr
	batch_size = args.batch_size
	T = args.timesteps

	log_prefix = f'{sys.argv[0]}-lr-{learning_rate}-T-{T}-b-{batch_size}'
	writer = SummaryWriter('./logs/' + log_prefix)

	cudnn.benchmark = True

	dataset_root_dir = args.data

	# Load data
	train_data_loader = torch.utils.data.DataLoader(
		dataset=torchvision.datasets.CIFAR10(root=dataset_root_dir,
		train=True,
		transform=torchvision.transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
		]),
		download=True),
		batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.workers)

	test_data_loader = torch.utils.data.DataLoader(
		dataset=torchvision.datasets.CIFAR10(root=dataset_root_dir,
		train=False,
		transform=torchvision.transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
		]),
		download=True),
		batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.workers)

	# Prepare model
	net = ResNet11().cuda()
	if args.pretrained:
		# The pretrained parameter can either be downloaded in authors' Dropbox (https://www.dropbox.com/sh/vvq9afkq90refka/AAAIEnyBZ_wO7eM510GCyZ8ta?dl=0) or trained by yourself. 
		# Should be placed together with code before training!
		checkpoint = torch.load('./model_bestT1_cifar10_r11.pth.tar')
		net.load_state_dict(checkpoint['state_dict'])

	print(net)

	criterion = nn.MSELoss(reduction='sum').cuda()

	optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 100, 125], gamma=0.2)

	max_test_accuracy = 0
	train_epoch = 0
	step = 0

	while 1:
		print(log_prefix)
		#### Train ####
		for img, label in tqdm(train_data_loader):
			label = label.cuda()
			label = F.one_hot(label, 10).float()
			img = img.cuda()

			optimizer.zero_grad()

			for t in range(T - 1):
				# Poisson encoding
				rand_num = torch.rand_like(img).cuda()
				poisson_input = (torch.abs(img) > rand_num).float()
				poisson_input = torch.mul(poisson_input, torch.sign(img))

				net(poisson_input)

			output = net(poisson_input)

			for m in net.modules():
				if isinstance(m, LIFNode) and m.fire:
					m.v_acc += (m.v_acc < 1e-3).float()
					m.new_grad = (m.v_acc_l > 1e-3).float() + math.log(1 - 1 / m.tau) * torch.div(m.v_acc_l, m.v_acc)

			loss = criterion(output / T, label)
			loss.backward()
			optimizer.step()
			net.reset_()

			writer.add_scalar('train_loss', loss, step)

			step += 1

		#### Evaluate ####
		with torch.no_grad():
			print('Test:')
			net.eval()
			accuracy = 0
			test_num = 0
			for img, label in tqdm(test_data_loader):
				label = label.cuda()
				img = img.cuda()

				for t in range(T - 1):
					# Poisson encoding
					rand_num = torch.rand_like(img).cuda()
					poisson_input = (torch.abs(img) > rand_num).float()
					poisson_input = torch.mul(poisson_input, torch.sign(img))

					net(poisson_input)

				output = net(poisson_input)

				accuracy += (output.argmax(dim=1) == label).float().sum().item()
				test_num += label.numel()
				net.reset_()
			
			accuracy /= test_num 
 
			if max_test_accuracy < accuracy:
				max_test_accuracy = accuracy
				torch.save(net.state_dict(), './logs/' + log_prefix + '/model_bestT1_cifar10_r11.pth.tar')
				print('保存模型参数', './logs/' + log_prefix + '/model_bestT1_cifar10_r11.pth.tar')
			writer.add_scalar('test_acc', accuracy, train_epoch)
			print(f'Test Acc: {accuracy}, Max Acc: {max_test_accuracy}')

			net.train()

		train_epoch += 1
		net.train_epoch += 1
		scheduler.step()

if __name__ == '__main__':
	main()