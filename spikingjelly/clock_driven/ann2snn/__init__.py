import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import json
from spikingjelly.clock_driven import neuron,encoding,functional
from collections import defaultdict
import copy
import time
import inspect
import matplotlib.pyplot as plt
import warnings

from spikingjelly.clock_driven.ann2snn.kernels.onnx import _o2p_converter as onnx2pytorch

class parser:
    def __init__(self, name='', kernel='onnx', **kargs):
        try:
            with open(kargs['json'], 'r') as f:
                self.config = json.load(f)
        except KeyError:
            try:
                self.log_dir = kargs['log_dir']
            except KeyError:
                from datetime import datetime
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                log_dir = os.path.join(
                    self.__class__.__name__ + '-' + current_time +
                    ('' if len(name) == 0 else '_' + name))
                self.log_dir = log_dir
            self.config = kargs
        print('parser log_dir:', self.log_dir)
        self.config['log_dir'] = self.log_dir
        self.kernel = kernel
        assert(self.kernel.lower() in ('onnx','pytorch'))
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        with open(os.path.join(self.log_dir,'parser_args.json'), 'w') as fw:
            json.dump(self.config, fw)

    def parse(self, model: nn.Module, data: torch.Tensor, **kargs) -> nn.Module:
        model_name = model.__class__.__name__
        model.eval()

        for m in model.modules():
            if hasattr(m,'weight'):
                assert(data.get_device() == m.weight.get_device())

        try:
            model = z_norm_integration(model=model, z_norm=self.config['z_norm'])
        except KeyError:
            pass
        layer_reduc = False
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                layer_reduc = True
                break
        if self.kernel.lower() == 'onnx':
            try:
                import onnx
                import onnxruntime as ort
            except ImportError:
                print(Warning("Package onnx or onnxruntime not found: launch pytorch convert engine,"
                              " only support very simple arctitecture"))
                self.kernel = 'pytorch'
            else:
                pass

        if self.kernel.lower() == 'onnx':
            # use onnx engine

            data = data.cpu()
            model = model.cpu()

            import spikingjelly.clock_driven.ann2snn.kernels.onnx as onnx_kernel

            onnx_model = onnx_kernel.pytorch2onnx_model(model=model, data=data, log_dir=self.config['log_dir'])
            # onnx_kernel.print_onnx_model(onnx_model.graph)
            onnx.checker.check_model(onnx_model)
            if layer_reduc:
                onnx_model = onnx_kernel.layer_reduction(onnx_model)
            # onnx.checker.check_model(onnx_model)
            onnx_model = onnx_kernel.rate_normalization(onnx_model, data.numpy(), **kargs) #**self.config['normalization']
            onnx_kernel.save_model(onnx_model,os.path.join(self.config['log_dir'],model_name+".onnx"))

            convert_methods = onnx2pytorch
            try:
                user_defined = kargs['user_methods']
                assert (user_defined is dict)
                for k in user_defined:
                    convert_methods.add_method(op_name=k, func=user_defined[k])
            except KeyError:
                print('no user-defined conversion method found, use default')
            except AssertionError:
                print('user-defined conversion method should be organized into a dict!')
            model = onnx_kernel.onnx2pytorch_model(onnx_model, convert_methods)
        else:
            # use pytorch engine

            import spikingjelly.clock_driven.ann2snn.kernels.pytorch as pytorch_kernel

            if layer_reduc:
                model = pytorch_kernel.layer_reduction(model)
            model = pytorch_kernel.rate_normalization(model, data)#, **self.config['normalization']

        self.ann_filename = os.path.join(self.config['log_dir'], model_name + ".pth")
        torch.save(model, os.path.join(self.config['log_dir'], "debug.pth"))
        torch.save(model, self.ann_filename)
        model = self.to_snn(model)
        return model

    def to_snn(self, model: nn.Module, **kargs) -> nn.Module:
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = self.to_snn(module, **kargs)
            if module.__class__.__name__ == "AvgPool2d":
                new_module = nn.Sequential(module, neuron.IFNode(v_reset=None))
                model._modules[name] = new_module
            if "BatchNorm" in module.__class__.__name__:
                try:
                    # NSIFNode是能够产生正负脉冲的模型，现在版本被删除
                    new_module = nn.Sequential(module, neuron.NSIFNode(v_threshold=(-1.0, 1.0), v_reset=None))
                except AttributeError:
                    new_module = module
                model._modules[name] = new_module
            if module.__class__.__name__ == "ReLU":
                new_module = neuron.IFNode(v_reset=None)
                model._modules[name] = new_module
            try:
                if module.__class__.__name__ == 'PReLU':
                    p = module.weight
                    assert (p.size(0) == 1 and p != 0)
                    if -1 / p.item() > 0:
                        model._modules[name] = neuron.NSIFNode(v_threshold=(1.0 / p.item(), 1.0),
                                                                     bipolar=(1.0, 1.0), v_reset=None)
                    else:
                        model._modules[name] = neuron.NSIFNode(v_threshold=(-1 / p.item(), 1.0),
                                                                     bipolar=(-1.0, 1.0), v_reset=None)
            except AttributeError:
                assert False, 'NSIFNode has been removed.'
            if module.__class__.__name__ == "MaxPool2d":
                new_module = nn.AvgPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding)
                model._modules[name] = new_module
        return model

def z_norm_integration(model: nn.Module, z_norm=None) -> nn.Module:
    if z_norm is not None:
        (z_norm_mean, z_norm_std) = z_norm
        z_norm_mean = torch.from_numpy(np.array(z_norm_mean).astype(np.float32))
        z_norm_std = torch.from_numpy(np.array(z_norm_std).astype(np.float32))
        bn = nn.BatchNorm2d(num_features=len(z_norm_std))
        bn.weight.data = torch.ones_like(bn.weight.data)
        bn.bias.data = torch.zeros_like(bn.bias.data)
        bn.running_var.data = torch.pow(z_norm_std, exponent=2) - bn.eps
        bn.running_mean.data = z_norm_mean
        return nn.Sequential(bn, model)
    else:
        return model

import threading
mutex_schedule = threading.Lock()
mutex_shared = threading.Lock()
global_shared = {}

class simulator:
    def __init__(self, snn, device, name='', **kargs):
        snn.eval()
        try:
            self.log_dir = kargs['log_dir']
        except KeyError:
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                self.__class__.__name__ + '-' + current_time +
                ('' if len(name)==0 else '_' + name))
            self.log_dir = log_dir
        print('simulator log_dir:',self.log_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        try:
            self.fig = kargs['canvas']
            self.ax = self.fig.add_subplot(1, 1, 1)
            plt.ion()
        except KeyError:
            self.fig = None

        try:
            encoder = kargs['encoder']
        except KeyError:
            encoder = 'constant'
        if encoder == 'poisson':
            self.encoder = encoding.PoissonEncoder()
        else:
            self.encoder = lambda x: x

        if isinstance(device,(list,set,tuple)):
            if len(device)==1:
                device = device[0]
                self.pi = False
            else:
                self.pi = True # parallel inference
        else:
            self.pi = False
        if self.pi:
            print('simulator is working on the parallel mode, device(s):', device)
        else:
            print('simulator is working on the normal mode, device:', device)
        self.device = device

        global global_shared, mutex_schedule, mutex_shared
        self.mutex_shared = mutex_shared
        self.mutex_schedule = mutex_schedule
        self.global_shared = global_shared
        if self.pi:
            self.global_shared['device_used'] = defaultdict(int)
            self.global_shared['device_stat'] = defaultdict(int)
            self.global_shared['distri_model'] = {}
            self.global_shared['batch'] = 0
            self.global_shared['batch_sum'] = 0
            self.global_shared['T'] = None
            for dev in self.device:
                self.global_shared['distri_model'][dev] = copy.deepcopy(snn).to(dev)
        else:
            self.global_shared['distri_model'] = {}
            self.global_shared['distri_model'][self.device] = copy.deepcopy(snn).to(self.device)
        self.config = dict()
        self.config['device'] = self.device
        self.config['name'] = name
        self.config['log_dir'] = self.log_dir
        self.config = {**self.config, **kargs}


    def simulate(self, data_loader, T, **kargs):
        self.config['T'] = T
        self.config = {**self.config, **kargs}
        with open(os.path.join(self.log_dir,'simulator_args.json'), 'w') as fw:
            json.dump({k: self.config[k] for k in self.config.keys() if _if_json_serializable(self.config[k])}
                      , fw)
        try:
            if kargs['online_drawer']:
                if isinstance(self.device, (list, set, tuple)):
                    warnings.warn("online drawer deprecated because package Matplotlib is not thread safe!")
        except KeyError:
            pass
        try:
            func_dict = kargs['func_dict']
        except KeyError:
            func_dict = {}
            for n in self._get_user_defined_static_methods():
                func_dict[n] = getattr(self,n)
        try:
            assert(len(func_dict.keys())>0)
        except AssertionError:
            raise KeyError("Please add valid func_dict for simulator, or use pre-defined subclass of ``simulator``!")
        if self.pi:
            threads = []
        start = time.perf_counter()
        global global_shared
        self.global_shared['T'] = T
        for value_name in func_dict:
            self.global_shared[value_name] = []
        self.global_shared['batch_sum'] = len(data_loader)
        for batch, (data, targets) in enumerate(tqdm(data_loader)):
            self.global_shared['batch'] = batch
            if self.pi:
                distributed = False
                while not distributed:
                    time.sleep(0.001) # time delay
                    for device in self.device:
                        if self.global_shared['device_used'][device] == 0:
                            t = threading.Thread(target=self.get_values,
                                                 kwargs=dict(data=data,
                                                             targets=targets,
                                                             device=device,
                                                             T=T,
                                                             func_dict=func_dict,
                                                             **kargs)
                                                 )
                            t.start()
                            threads.append(t)
                            distributed = True
                            self.global_shared['device_stat'][device] += 1
                            break
            else:
                self.get_values(data=data,
                                targets=targets,
                                device=self.device,
                                T=T,
                                func_dict=func_dict,
                                **kargs)
        if self.pi:
            for t in threads:
                t.join()
        elapsed = time.perf_counter() - start
        print('--------------------simulator summary--------------------')
        print('time elapsed:', elapsed, '(sec)')
        if self.pi:
            print('load stat:',self.global_shared['device_stat'])
        print('---------------------------------------------------------')

        try:
            if kargs['canvas'] is not None:
                plt.ioff()
                plt.close()
        except KeyError:
            pass

        ret_dict = {}

        for value_name in func_dict:
            ret_dict[value_name] = self.global_shared[value_name]
        return ret_dict

    def get_values(self, data, targets, device, T, func_dict, **kargs):
        if self.pi:
            if mutex_shared.acquire():
                getattr(self, '_pre_batch_sim')(**kargs)
                mutex_shared.release()
        else:
            getattr(self, '_pre_batch_sim')(**kargs)
        global global_shared
        data = data.to(device)
        targets = targets.to(device)
        values_list = defaultdict(list)

        if self.pi:
            if mutex_schedule.acquire():
                self.global_shared['device_used'][device] = 1
                mutex_schedule.release()

        snn = self.global_shared['distri_model'][device]
        functional.reset_net(snn)
        with torch.no_grad():
            for t in range(T):
                enc = self.encoder(data).float().to(device)
                out = snn(enc)
                if t == 0:
                    counter = out
                else:
                    counter += out
                for value_name in func_dict.keys():
                    value = func_dict[value_name](data=data,
                                                  targets=targets,
                                                  out_spike=out,
                                                  out_spike_cnt=counter,
                                                  device=device,
                                                  **kargs)
                    values_list[value_name].append(value)

        for value_name in func_dict.keys():
            values_list[value_name] = np.array(values_list[value_name]).astype(np.float32)

        if self.pi:
            if mutex_shared.acquire():
                for value_name in func_dict.keys():
                    self.global_shared[value_name].append(values_list[value_name])
                getattr(self, '_after_batch_sim')(**kargs)
                mutex_shared.release()
        else:
            for value_name in func_dict.keys():
                self.global_shared[value_name].append(values_list[value_name])
            getattr(self, '_after_batch_sim')(**kargs)

        if self.pi:
            if mutex_schedule.acquire():
                self.global_shared['device_used'][device] = 0
                mutex_schedule.release()

    def _get_user_defined_static_methods(self):
        method = []
        attrs = dir(self)
        for attr in attrs:
            if attr[0] != '_':
                user_defined = inspect.isroutine(getattr(self, attr))
                static_method = False
                for cls in inspect.getmro(type(self)):
                    if attr in cls.__dict__:
                        v = cls.__dict__[attr]
                        if isinstance(v, staticmethod):
                            static_method = True
                if user_defined and static_method:
                    method.append(attr)
        return method

    def _pre_batch_sim(self, **kargs):
        pass

    def _after_batch_sim(self, **kargs):
        pass



class classify_simulator(simulator):  # 一个分类任务的实例
    def __init__(self, snn, device, **kargs):
        super().__init__(snn, device, **kargs)
        self.global_shared['accu_correct'] = 0.0
        self.global_shared['accu_total'] = 0.0
        self.global_shared['acc'] = 0.0
        # try:
        #     self.fig = kargs['canvas']
        #     self.ax = self.fig.add_subplot(1, 1, 1)
        #     plt.ion()
        # except KeyError:
        #     self.fig = None

    @staticmethod
    def correct_num(targets, out_spike_cnt, **kargs) -> float:
        n = (out_spike_cnt.max(1)[1] == targets).float().sum().item()
        return n

    @staticmethod
    def total_num(targets, **kargs) -> float:
        n = targets.size(0)
        return n

    def _after_batch_sim(self, **kargs):
        import matplotlib.pyplot as plt
        T = self.global_shared['T']
        self.global_shared['accu_correct'] += self.global_shared['correct_num'][-1]
        self.global_shared['accu_total'] += self.global_shared['total_num'][-1]
        self.global_shared['acc'] = self.global_shared['accu_correct'] \
                                       / self.global_shared['accu_total']
        np.savetxt(os.path.join(self.log_dir, 'acc.csv'), self.global_shared['acc'], delimiter=",")

        if self.fig is not None:
            self.ax.cla()
            x = np.arange(self.global_shared['acc'].shape[0])
            self.ax.plot(x,self.global_shared['acc'] * 100,label='SNN Acc')

            try:
                ann_acc = kargs['ann_acc'] * 100
                self.ax.plot(x, np.ones_like(x) * ann_acc, label='ANN', c='g', linestyle=':')
                self.ax.text(0, ann_acc + 1, "%.3f%%" % (ann_acc), fontdict={'size': '8', 'color': 'g'})
            except KeyError:
                pass
            try:
                self.ax.set_title("%s\n[%.1f%% dataset]" % (
                    kargs['fig_name'],
                    100.0 * (self.global_shared['batch']+1) / self.global_shared['batch_sum']
                ))
            except KeyError:
                pass
            try:
                if kargs['step_max']:
                    y = self.global_shared['acc'] * 100
                    argmax = np.argmax(y)
                    disp_bias = 0.3 * float(T) if x[argmax] / T > 0.7 else 0
                    self.ax.text(x[argmax] - 0.8 - disp_bias, y[argmax] + 0.8, "MAX:%.3f%% T=%d" % (y[argmax], x[argmax]),
                             fontdict={'size': '12', 'color': 'r'})
                    self.ax.scatter([x[argmax]], [y[argmax]], c='r')
            except KeyError:
                pass

            self.ax.set_xlabel("T")
            self.ax.set_ylabel("Percentage(%)")
            self.ax.legend()
            plt.savefig(os.path.join(self.log_dir,'plot.pdf'))

            try:
                if kargs['online_drawer']:
                    if not isinstance(self.device, (list, set, tuple)):
                        plt.pause(0.001)
            except KeyError:
                pass

def _if_json_serializable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False