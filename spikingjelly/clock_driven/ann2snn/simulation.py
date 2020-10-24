import torch
import torch.nn as nn
import spikingjelly.clock_driven.neuron as neuron
import spikingjelly.clock_driven.ann2snn.modules as modules
import spikingjelly.clock_driven.encoding as encoding
import spikingjelly.clock_driven.functional as functional
import matplotlib.pyplot as plt
import numpy as np
import tqdm
def simulate_snn(snn, device, data_loader, T, poisson=False, online_draw=False,fig_name='default',ann_baseline=0,save_acc_list=False,log_dir=None): # TODO ugly
    '''
    * :ref:`API in English <simulate_snn-en>`

    .. _simulate_snn-cn:

    :param snn: SNN模型
    :param device: 运行的设备
    :param data_loader: 测试数据加载器
    :param T: 仿真时长
    :param poisson: 当设置为 ``True`` ，输入采用泊松编码器；否则，采用恒定输入并持续T时间步
    :return: SNN模拟的准确率

    对SNN分类性能进行评估，并返回准确率

    * :ref:`中文API <simulate_snn-cn>`

    .. _simulate_snn-en:

    :param snn: SNN model
    :param device: running device
    :param data_loader: testing data loader
    :param T: simulating steps
    :param poisson: when ``True``, use poisson encoder; otherwise, use constant input over T steps
    :return: SNN simulating accuracy

    '''
    functional.reset_net(snn)
    if poisson:
        encoder = encoding.PoissonEncoder()
    correct_t = {}
    if online_draw:
        plt.ion()
    with torch.no_grad():
        snn.eval()
        correct = 0.0
        total = 0.0
        for batch, (img, label) in enumerate(data_loader):
            img = img.to(device)
            for t in tqdm.tqdm(range(T)):
                encoded = encoder(img).float() if poisson else img
                out = snn(encoded)
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                if t == 0:
                    out_spikes_counter = out
                else:
                    out_spikes_counter += out

                if t not in correct_t.keys():
                    correct_t[t] = (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                else:
                    correct_t[t] += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            correct += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            total += label.numel()
            functional.reset_net(snn)
            if online_draw:
                plt.cla()
                x = np.array(list(correct_t.keys())).astype(np.float32) + 1
                y = np.array(list(correct_t.values())).astype(np.float32) / total * 100
                plt.plot(x, y,label='SNN',c='b')
                if ann_baseline!=0:
                    plt.plot(x,np.ones_like(x)*ann_baseline,label='ANN',c='g',linestyle=':')
                    plt.text(0, ann_baseline + 1, "%.3f%%" % (ann_baseline), fontdict={'size': '8', 'color': 'g'})
                plt.title("%s SNN Simulation \n[test samples:%.1f%%]"%(
                    fig_name, 100*total/len(data_loader.dataset)
                ))
                plt.xlabel("T")
                plt.ylabel("Accuracy(%)")
                plt.legend()
                argmax = np.argmax(y)
                disp_bias = 0.3 * float(T) if x[argmax]/T > 0.7 else 0
                plt.text(x[argmax]-0.8-disp_bias, y[argmax]+0.8, "MAX:%.3f%% T=%d"%(y[argmax],x[argmax]),
                         fontdict={'size': '12', 'color': 'r'})

                plt.scatter([x[argmax]],[y[argmax]],c='r')
                plt.pause(0.01)
                if isinstance(log_dir,str):
                    plt.savefig(log_dir +'/' + fig_name + ".pdf")
            print('[SNN Simulating... %.2f%%] Acc:%.3f'%(100*total/len(data_loader.dataset),correct / total))
            if save_acc_list:
                acc_list = np.array(list(correct_t.values())).astype(np.float32) / total * 100
                np.save(log_dir + '/snn_acc-list' + ('-poisson' if poisson else '-constant'), acc_list)
        acc = correct / total
        print('SNN Simulating Accuracy:%.3f'%(acc))

    if online_draw:
        plt.savefig(log_dir +'/' + fig_name + ".pdf")
        plt.ioff()
        plt.close()
    return acc

import copy
import torch.utils.data
import threading
class FunctionThreadWithRet(threading.Thread):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.ret = None
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.ret = self.f(*self.args, **self.kwargs)

    def get_ret(self):
        return self.ret

def parallel_inference_correct_sum_loss(net: nn.Module, data_loader: torch.utils.data.DataLoader, device: str):
    '''
    :param net: 用于分类任务的网络，输出是一个 ``shape = [batch_size, C]`` 的tensor
    :type net: nn.Module
    :param data_loader: 分类任务数据（例如 ``torchvision.datasets.MNIST``）的加载器
    :type data_loader: torch.utils.data.DataLoader
    :param device: 网络所在的设备
    :type device: str
    :return: 分类正确的总数
    :rtype: int
    '''
    correct_sum = 0
    net.eval()
    with torch.no_grad():
        for x, y in data_loader:
            y_p = net(x.to(device))
            correct_sum += (y_p.argmax(dim=1) == y.to(device)).sum().item()
    return correct_sum

def parallel_inference(net: nn.Module, data_loaders: list, devices: list, loss_fun, *args, **kwargs):
    '''
    使用示例：
    .. code-block:: python

        class TestNet(nn.Module):
            def __init__(self, class_num):
                super().__init__()
                self.class_num = class_num
            def forward(self, x: torch.Tensor):
                return torch.rand(size=[x.shape[0], self.class_num]).to(x)
        class TestClassifyDataset(torch.utils.data.Dataset):
            def __init__(self, class_num, data_num):
                self.class_num = class_num
                self.data_num = data_num
                self.x = torch.rand(size=[self.data_num])
                self.y = torch.randint(0, self.class_num, size=[self.data_num])
            def __len__(self):
                return self.data_num
            def __getitem__(self, index):
                return self.x[index], self.y[index]

        class_num = 10
        origin_dataset = TestClassifyDataset(class_num=class_num, data_num=10000)
        data_num = origin_dataset.__len__()
        devices = ['cuda:9', 'cuda:10', 'cuda:11', 'cuda:13']
        block_num = data_num // devices.__len__()
        data_loaders = []
        for i in range(devices.__len__()):
            data_loaders.append(
                torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(origin_dataset, list(range(i * block_num, min(data_num, (i + 1) * block_num)))),
                    batch_size=128
                )
            )


        correct_sum = parallel_inference(TestNet(class_num=class_num), data_loaders, devices, parallel_inference_correct_sum_loss)
        print(f'accuracy={correct_sum / data_num}')
    '''
    inf_threads = []
    for i in range(devices.__len__()):
        net_c = copy.deepcopy(net).to(devices[i])
        inf_threads.append(FunctionThreadWithRet(loss_fun, net_c, data_loaders[i], devices[i], *args, **kwargs))
        print(f'start thread {i} on device {devices[i]}')
        inf_threads[-1].start()

    loss_sum = 0
    for i in range(devices.__len__()):
        inf_threads[i].join()
        print(f'finish thread {i} on device {devices[i]}')
        loss_sum += inf_threads[i].get_ret()
    return loss_sum



