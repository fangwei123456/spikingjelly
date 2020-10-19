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
