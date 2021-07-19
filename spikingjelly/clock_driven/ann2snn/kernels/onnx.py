import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import collections
import numpy as np
import torch
import torch.nn as nn
import os
import tqdm
import onnxruntime as ort
from collections import defaultdict
import json


class Mul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        return input1 * input2


class Add(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input1,input2):
        return input1 + input2


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return torch.reshape(input1,shape=list(input2))


class Concat(nn.Module):
    def __init__(self, dim=[1]):
        super().__init__()
        self.dim = dim
        if not isinstance(self.dim, list):
            self.dim = [self.dim]
        for i, d in enumerate(self.dim):
            if not isinstance(d, int):
                self.dim[i] = int(d)

    def forward(self, *args):
        args = list(args)
        for i,a in enumerate(args):
            args[i] = a.type_as(args[0])
        return torch.cat(args,dim=self.dim[0])

class Shape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.IntTensor([input.size(i) for i in range(len(input.size()))])

class Gather(nn.Module):
    def __init__(self,dim=1):
        super().__init__()
        self.dim= int(dim)

    def forward(self, input1, input2):
        return torch.gather(input1,dim=self.dim,index=input2.cpu())

class Unsqueeze(nn.Module):
    def __init__(self, dim=[1]):
        super().__init__()
        self.dim = dim
        if not isinstance(self.dim, list):
            self.dim = [self.dim]
        for i,d in enumerate(self.dim):
            if not isinstance(d,int):
                self.dim[i] = int(d)

    def forward(self, input):
        x = input
        for i in self.dim:
            x = torch.unsqueeze(x,dim=i)
        return x

class TopologyAnalyser:
    def __init__(self):
        '''
        * :ref:`API in English <TopologyAnalyser.__init__-en>`

        .. _TopologyAnalyser.__init__-cn:

        这个类通过onnx分析模型的拓扑结构，方便后续处理
        此处还有更多更好的实现方法，欢迎开发者不断优化

        * :ref:`API in English <TopologyAnalyser.__init__-cn>`

        .. _TopologyAnalyser.__init__-en:

        This class analyzes the topological structure of the model through onnx to facilitate subsequent processing
        There are better implementation methods here, developers are welcome to continue to optimize
        '''
        self.data_nodes = []
        self.module_output = collections.OrderedDict()
        self.module_op = collections.OrderedDict()
        self.module_idx = collections.OrderedDict()
        self.param_idx = collections.OrderedDict()
        self.edge = collections.OrderedDict()
        self.reverse_edge = collections.OrderedDict()  # 快速计算前驱结点

    def add_data_node(self, a):
        if not a in self.data_nodes:
            self.data_nodes.append(a)

    def insert(self, a, b, info=None):
        self.add_data_node(a)
        self.add_data_node(b)
        if a not in self.edge.keys():
            self.edge[a] = [(b, info)]
        else:
            self.edge[a].append((b, info))
        if b not in self.reverse_edge.keys():
            self.reverse_edge[b] = [a]
        else:
            self.reverse_edge[b].append(a)

    def findNext(self, id):
        if isinstance(id, str):
            if id in self.edge.keys():
                return self.edge[id]
            else:
                return []
        elif isinstance(id, list):
            l = []
            for i in id:
                l += self.findNext(i)
            return l

    def findPre(self, id):
        l = []
        if isinstance(id, str):
            for pre_id in self.reverse_edge[id]:
                if pre_id in self.reverse_edge.keys():
                    for pre_pre_id in self.reverse_edge[pre_id]:
                        if pre_pre_id in self.edge.keys():
                            for item in self.edge[pre_pre_id]:
                                if item[0] == pre_id:
                                    l.append(item)
        elif isinstance(id, list):
            for i in id:
                l += self.findPre(i)
        return l

    def find_pre_module(self, module_name):
        if module_name in self.module_output.keys():
            ids = self.module_output[module_name]
            return set(['%s:%s' % (k[1]['op'], k[1]['param_module_name']) for k in self.findPre(ids)])
        else:
            return set()

    def find_next_module(self, module_name):
        if module_name in self.module_output.keys():
            ids = self.module_output[module_name]
            return set(['%s:%s' % (k[1]['op'], k[1]['param_module_name']) for k in self.findNext(ids)])
        else:
            return set()

    def update_module_idx(self, onnx_graph):
        for idx, n in enumerate(onnx_graph.node):
            trainable_input = n.input[1:]
            op = n.op_type
            k = set()
            for i in trainable_input:
                n = self._get_module_name_from_value_name(i)
                if n is not None:
                    k.add(n)
            if len(k) > 1:
                # TODO: sanity check, raise error
                pass
            if len(k) == 1:
                param_module_name = list(k)[0]
                self.module_op[param_module_name] = op
                self.module_idx[param_module_name] = idx

    def analyse(self, onnx_graph):  # 输入的onnx graph需要保证所以常量在已经在initializer中
        # 先把该往initializer放下面的参数，保证下面只有运算没有常量
        for idx, constant in enumerate(onnx_graph.initializer):
            self.param_idx[constant.name] = idx
        for idx, n in enumerate(onnx_graph.node):
            param_module_name = None
            op = n.op_type
            inputs = n.input
            outputs = n.output
            # print(inputs,outputs)
            k = set()
            trainable_input = inputs[1:]
            for i in trainable_input:
                n = self._get_module_name_from_value_name(i)
                if n is not None:
                    k.add(n)
            if len(k) > 1:
                # TODO: sanity check, raise error
                pass
            if len(k) == 1:
                param_module_name = list(k)[0]
                self.module_op[param_module_name] = op
                self.module_idx[param_module_name] = idx
            if op is not None:
                for o in outputs:
                    for i in inputs:
                        self.insert(i, o, {'op': op, 'param_module_name': param_module_name})
                    if param_module_name is not None:
                        if param_module_name not in self.module_output.keys():
                            self.module_output[param_module_name] = [o]
                        else:
                            self.module_output[param_module_name].append(o)
        return self

    @staticmethod
    def _get_module_name_from_value_name(value_name):
        module_name = None
        if len(value_name.split('.')) > 1:
            l = value_name.split('.')[:-1]
            l = '.'.join(l)
            module_name = l  # [1:]
            module_name.replace(' ', '')
        return module_name

def pytorch2onnx_model(model: nn.Module, data, **kargs) -> onnx.ModelProto:
    '''

    * :ref:`API in English <pytorch2onnx_model-en>`

    .. _pytorch2onnx_model-cn:

    :param model: 待转换的PyTorch模型

    :param data: 用于转换的数据（用来确定输入维度）

    :param log_dir: 输出文件夹

    转换PyTorch模型到onnx模型

    * :ref:`API in English <pytorch2onnx_model-cn>`

    .. _pytorch2onnx_model-en:

    :param model: the PyTorch model to be converted

    :param data: The data used for conversion (used to determine the input dimension)

    :param log_dir: output folder

    Convert PyTorch model to onnx model

    '''
    try:
        log_dir = kargs['log_dir']
    except KeyError:
        print('pytorch2onnx_model need argument log_dir!')
    dump_input_size = [data.size(i) for i in range(len(data.size()))]
    dump_input_size[0] = 1
    fname = os.path.join(log_dir,'onnxmodel')
    try:
        dynamic_axes = {'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
        torch.onnx.export(model, torch.ones(dump_input_size), fname,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)
    except BaseException:
        raise NotImplementedError("Models with multiple inputs are not supported yet!")
    return onnx.load(fname)

def onnx2pytorch_model(model: onnx.ModelProto, _converter) -> nn.Module:
    model = _pt_model(model, _converter)
    model = model.reduce()
    return model

def layer_reduction(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    topo_analyser = TopologyAnalyser()
    graph = move_constant_to_initializer(graph)
    topo_analyser.analyse(graph)

    absorb_bn(graph, topo_analyser)
    remove_unreferenced_initializer(graph)
    update_topology(graph)
    print("Finish layer reduction!")
    return model

def rate_normalization(model: onnx.ModelProto, data: torch.Tensor, **kargs) -> onnx.ModelProto:
    '''

    * :ref:`API in English <rate_normalization-en>`

    .. _rate_normalization-cn:

    :param model: ANN模型，类型为onnx.ModelProto

    :param data: 用于转换的数据，类型为torch.Tensor

    :param channelwise: 如果为``True``，则控制激活幅值的统计是channelwise的；否则，控制激活幅值的统计是layerwise的

    :param robust: 如果为``True``，则控制激活幅值的统计是激活的99.9百分位；否则，控制激活幅值的统计是激活的最值

    :param eps: epsilon；未设置值时默认1e-5

    发放率归一化

    * :ref:`API in English <rate_normalization-cn>`

    .. _rate_normalization-en:

    :param model: ANN model, the type is onnx.ModelProto

    :param data: the data used for conversion, the type is torch.Tensor

    :param channelwise: If ``True`` , the statistics that control the activation amplitude are channelwise; otherwise, the statistics that control the activation amplitude are layerwise

    :param robust: If ``True``, the statistic of the control activation amplitude is the 99.9th percentile of activation; otherwise, the statistic of the activation amplitude is the maximum value of activation

    :param eps: epsilon; if no value is set, the default is 1e-5

    normalize the firing rate

    '''

    try:
        channelwise = kargs['channelwise']
    except KeyError:
        channelwise = False
    try:
        robust_norm = kargs['robust']
    except KeyError:
        robust_norm = False
    try:
        eps = kargs['eps']
    except KeyError:
        eps = 1e-5
    topo_analyser = update_topology(model.graph)
    output_debug = {}
    output_statistics = get_intermediate_output_statistics(model, data,
                                                           channelwise=channelwise) # if want debug, debug=output_debug
    model = normalize_model(model, output_statistics, topo_analyser, robust_norm=robust_norm,
                            channelwise=channelwise, eps=eps)
    return model

def save_model(model: onnx.ModelProto, f=None):
    fb = model.SerializeToString()
    if f is not None:
        if hasattr(f, 'write'):
            f.write(fb)
        else:
            with open(f, "wb") as f:
                f.write(fb)
    return fb

def move_constant_to_initializer(graph):
    constant_idx = []
    for idx, n in enumerate(graph.node):
        op = n.op_type
        if op == 'Constant':
            constant_idx.append(idx)
    if len(constant_idx):
        for idx in reversed(constant_idx):
            n = graph.node[idx]
            graph.initializer.append(
                numpy_helper.from_array(numpy_helper.to_array(n.attribute[0].t), n.output[0]))
            graph.node.remove(n)
    return graph

def print_onnx_model(graph):
    print(onnx.helper.printable_graph(graph))

def absorb_bn(graph, topo_analyser):
    print("\nAbsorbing BatchNorm Parameters...\n")
    for mn in tqdm.tqdm(reversed(topo_analyser.module_output.keys())):
        if topo_analyser.module_op[mn] == 'BatchNormalization':
            pre_m = topo_analyser.find_pre_module(mn)
            next_m = topo_analyser.find_next_module(mn)
            bn_weight_idx = topo_analyser.param_idx[graph.node[topo_analyser.module_idx[mn]].input[1]]
            bn_weight = np.array(numpy_helper.to_array(graph.initializer[bn_weight_idx]))
            bn_bias_idx = topo_analyser.param_idx[graph.node[topo_analyser.module_idx[mn]].input[2]]
            bn_bias = np.array(numpy_helper.to_array(graph.initializer[bn_bias_idx]))
            bn_mean_idx = topo_analyser.param_idx[graph.node[topo_analyser.module_idx[mn]].input[3]]
            bn_mean = np.array(numpy_helper.to_array(graph.initializer[bn_mean_idx]))
            bn_var_idx = topo_analyser.param_idx[graph.node[topo_analyser.module_idx[mn]].input[4]]
            bn_var = np.array(numpy_helper.to_array(graph.initializer[bn_var_idx]))
            bn_eps = graph.node[topo_analyser.module_idx[mn]].attribute[0].f
            bn_std = np.sqrt(bn_var + bn_eps)
            if len(pre_m) == 1 and list(pre_m)[0].split(':')[0] in ['Conv', 'Gemm']:
                pre_mn = list(pre_m)[0].split(':')[1]
                weight_idx = topo_analyser.param_idx[graph.node[topo_analyser.module_idx[pre_mn]].input[1]]
                weight = np.array(numpy_helper.to_array(graph.initializer[weight_idx]))
                if len(graph.node[topo_analyser.module_idx[pre_mn]].input) == 2:
                    bias = None
                else:
                    bias_idx = topo_analyser.param_idx[graph.node[topo_analyser.module_idx[pre_mn]].input[2]]
                    bias = np.array(numpy_helper.to_array(graph.initializer[bias_idx]))
                wrsp_args = (-1, 1) if len(weight.shape) == 2 else (-1, 1, 1, 1)

                weight_ = weight * bn_weight.reshape(*wrsp_args) / bn_std.reshape(*wrsp_args)
                bias_ = ((bias if bias is not None else 0) - bn_mean.reshape(-1)) * bn_weight.reshape(
                    -1) / bn_std.reshape(-1) \
                        + bn_bias.reshape(-1)
                assert (list(pre_m)[0].split(':')[0] in ['Conv', 'Gemm'])
                args = {}
                for attr in graph.node[topo_analyser.module_idx[pre_mn]].attribute:
                    args[attr.name] = helper.get_attribute_value(attr)
                new_node = onnx.helper.make_node(
                    list(pre_m)[0].split(':')[0],
                    inputs=[graph.node[topo_analyser.module_idx[pre_mn]].input[0], pre_mn + ".new.weight", pre_mn + ".new.bias"],
                    outputs=[graph.node[topo_analyser.module_idx[mn]].output[0]],
                    **args
                )
                graph.initializer.append(numpy_helper.from_array(weight_.astype(np.float32), pre_mn + ".new.weight"))
                graph.initializer.append(numpy_helper.from_array(bias_.astype(np.float32), pre_mn + ".new.bias"))
                graph.node.remove(graph.node[topo_analyser.module_idx[pre_mn]])
                graph.node.insert(topo_analyser.module_idx[pre_mn], new_node)
                graph.node.remove(graph.node[topo_analyser.module_idx[mn]])
            else:
                weight_ = bn_weight / bn_std
                bias_ = bn_bias - bn_weight * bn_mean / bn_std
                name = graph.initializer[bn_weight_idx].name
                graph.initializer.remove(graph.initializer[bn_weight_idx])
                graph.initializer.insert(bn_weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
                name = graph.initializer[bn_bias_idx].name
                graph.initializer.remove(graph.initializer[bn_bias_idx])
                graph.initializer.insert(bn_bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))
                name = graph.initializer[bn_mean_idx].name
                graph.initializer.remove(graph.initializer[bn_mean_idx])
                graph.initializer.insert(bn_mean_idx,
                                   numpy_helper.from_array(np.zeros_like(bn_mean).astype(np.float32), name))
                name = graph.initializer[bn_var_idx].name
                graph.initializer.remove(graph.initializer[bn_var_idx])
                graph.initializer.insert(bn_var_idx,
                                   numpy_helper.from_array(np.ones_like(bn_var).astype(np.float32), name))

def remove_unreferenced_initializer(graph):
    in_graph = set()
    in_initializer = set()
    for node in graph.node:
        in_graph.update(node.input)
        in_graph.update(node.output)
    for init in graph.initializer:
        in_initializer.add(init.name)
    not_in_graph = in_initializer - in_graph
    l = len(graph.initializer)
    for i in range(l - 1, -1, -1):
        if graph.initializer[i].name in not_in_graph:
            graph.initializer.remove(graph.initializer[i])

def update_topology(graph):
    topo_analyser = TopologyAnalyser()
    move_constant_to_initializer(graph)
    topo_analyser.analyse(graph)
    return topo_analyser

def find_node_by_output(output_name, graph):
    flag = False
    idx, node = None, None
    for idx, node in enumerate(graph.node):
        if output_name in node.output:
            flag = True
            break
    if not flag:
        idx, node = None, None
    return idx, node

def scale_node_weight_bias(topo_analyser, graph, node_idx, scale):
    initializer = graph.initializer
    node = graph.node[node_idx]
    if len(node.input) < 2:
        return
    weight_idx = topo_analyser.param_idx[node.input[1]]
    bias_idx = topo_analyser.param_idx[node.input[2]] if len(node.input) >= 3 else None
    weight = np.array(numpy_helper.to_array(initializer[weight_idx]))
    bias = np.array(numpy_helper.to_array(initializer[bias_idx])) if bias_idx is not None else None

    w_scale = scale.reshape([*scale.shape] + [1 for _ in range(len(weight.shape) - 1)]) \
        if len(scale.shape) == 1 else scale
    b_scale = scale

    weight_ = weight * w_scale
    name = initializer[weight_idx].name
    initializer.remove(initializer[weight_idx])
    initializer.insert(weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
    if bias is not None:
        bias_ = bias * b_scale
        name = initializer[bias_idx].name
        initializer.remove(initializer[bias_idx])
        initializer.insert(bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))

def get_onnx_output(model, numpy_tensor):
    ort_session = ort.InferenceSession(model.SerializeToString())
    outputs = ort_session.run(None, {'input': numpy_tensor})
    return outputs

def get_intermediate_output_statistics(model, numpy_tensor, channelwise=False, debug=None):
    graph = model.graph
    output_needed_module = {}
    output_needed_all_input = {}
    for idx, node in enumerate(graph.node):
        output = node.output
        input = node.input
        if 'input' in node.input:
            for out in output:
                output_needed_module[out] = set([idx])
                output_needed_all_input[out] = set(input)
        else:
            s = set()
            s_i = set()
            for in_ in input:
                s |= (output_needed_module[in_] if in_ in output_needed_module.keys() else set())
                s_i |= (output_needed_all_input[in_] if in_ in output_needed_all_input.keys() else set())
            for out in output:
                output_needed_module[out] = s | set([idx])
                output_needed_all_input[out] = s_i | set(input)

    output_statistics = {}
    if not channelwise:
        statistic = {'shape': numpy_tensor.shape,
                     'min': np.min(numpy_tensor),
                     'max': np.max(numpy_tensor) if np.max(numpy_tensor) > 0 else np.abs(np.min(numpy_tensor)),
                     '99.9': np.percentile(numpy_tensor, 99.9)
                     }
    else:
        axis_args = (0, 2, 3) if len(numpy_tensor.shape) == 4 else (0)
        statistic = {'shape': numpy_tensor.shape,
                     'min': np.min(numpy_tensor, axis=axis_args),
                     'max': np.max(numpy_tensor, axis=axis_args),
                     '99.9': np.percentile(numpy_tensor, 99.9, axis=axis_args)
                     }
    output_statistics['input'] = statistic
    print("\nGetting intermediate output statistics...\n")
    for out in tqdm.tqdm(output_needed_module.keys()):
        keep_nodes = [graph.node[i] for i in list(output_needed_module[out])]
        keep_initializer = [init for init in graph.initializer
                            if init.name in list(output_needed_all_input[out])]
        var_out = []
        value_info = onnx.ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
        new_graph = onnx.helper.make_graph(keep_nodes, graph.name, graph.input,
                                       var_out, keep_initializer)
        tmp_model = onnx.helper.make_model(new_graph)
        tmp_model.ir_version = model.ir_version
        tmp_model.producer_name = model.producer_name
        tmp_model.producer_version = model.producer_version
        tmp_model.domain = model.domain
        tmp_model.model_version = model.model_version
        tmp_model.doc_string = model.doc_string
        if len(tmp_model.metadata_props) > 0:
            values = {p.key: p.value for p in model.metadata_props}
            onnx.helper.set_model_props(tmp_model, values)
        # fix opset import
        for oimp in model.opset_import:
            op_set = tmp_model.opset_import.add()
            op_set.domain = oimp.domain
            op_set.version = oimp.version

        ort_session = ort.InferenceSession(tmp_model.SerializeToString())
        outputs = ort_session.run(None, {'input': numpy_tensor})
        if debug is not None:
            # print(out,outputs[0].reshape(1,-1)[0,10:20])
            debug[out] = outputs[0]
        if not channelwise:
            statistic = {'shape': outputs[0].shape,
                         'min': np.min(outputs[0]),
                         'max': np.max(outputs[0]) if np.max(outputs[0]) > 0 else np.abs(np.min(outputs[0])),
                         '99.9': np.percentile(outputs[0], 99.9) if np.percentile(outputs[0], 99.9) > 0 else np.abs(np.min(outputs[0]))
                         }
        else:
            axis_args = (0, 2, 3) if len(outputs[0].shape) == 4 else (0)
            statistic = {'shape': outputs[0].shape,
                         'min': np.min(outputs[0], axis=axis_args),
                         'max': np.max(outputs[0], axis=axis_args),
                         '99.9': np.percentile(outputs[0], 99.9, axis=axis_args)
                         }
            # print(np.max(statistic['max']),np.max(outputs[0]))
        output_statistics[out] = statistic
    print("Finished getting intermediate output statistics!")
    if debug is not None:
        return output_statistics,debug
    else:
        return output_statistics

def normalize_model(model, output_statistics, topo_analyser, robust_norm=True, channelwise=False, eps=1e-5):
    nodes = model.graph.node
    graph = model.graph
    initializer = model.graph.initializer
    if robust_norm:
        statistic_key = '99.9'
    else:
        statistic_key = 'max'
    node_scaled_range = {}
    seperate_scale = collections.OrderedDict()
    print("\nNormalizing model...\n")
    for node_idx, node in enumerate(tqdm.tqdm(nodes)):
        output = node.output
        input = node.input
        op = node.op_type
        if input[0] == 'input':  # single input model
            l = output_statistics[input[0]]['shape'][1]
            node_scaled_range[input[0]] = np.ones(l) if channelwise else 1.0 \
                                                                         * output_statistics[input[0]][
                                                                             statistic_key]

        if op in ['Conv', 'Gemm']:
            weight_idx = topo_analyser.param_idx[input[1]]
            bias_idx = topo_analyser.param_idx[input[2]] if len(input) == 3 else None
            weight = np.array(numpy_helper.to_array(initializer[weight_idx]))
            bias = np.array(numpy_helper.to_array(initializer[bias_idx])) if bias_idx is not None else None

            l = output_statistics[output[0]]['shape'][1]
            input_real_range = node_scaled_range[input[0]]
            input_range = output_statistics[input[0]][statistic_key]
            output_range = output_statistics[output[0]][statistic_key]
            demand = np.ones(l) if channelwise else 1.0
            w_scale = demand / (output_range + eps) * (
                        input_range / (input_real_range + eps)) if not channelwise else \
                (demand / (output_range + eps)).reshape(-1, 1).dot(
                    (input_range / (input_real_range + eps)).reshape(1, -1))
            w_scale = w_scale.reshape([*w_scale.shape, 1, 1]) if len(weight.shape) == 4 else w_scale
            b_scale = 1 / (output_range + eps)
            node_scaled_range[output[0]] = demand

            weight_ = weight * w_scale

            name = initializer[weight_idx].name
            initializer.remove(initializer[weight_idx])
            initializer.insert(weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
            if bias is not None:
                bias_ = bias * b_scale
                name = initializer[bias_idx].name
                initializer.remove(initializer[bias_idx])
                initializer.insert(bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))

        elif op == 'BatchNormalization':  # var=1 mean=0
            weight_idx = topo_analyser.param_idx[input[1]]
            bias_idx = topo_analyser.param_idx[input[2]]
            weight = np.array(numpy_helper.to_array(initializer[weight_idx]))
            bias = np.array(numpy_helper.to_array(initializer[bias_idx]))

            # node_scaled_range[output[0]] = node_scaled_range[input[0]] * self.output_statistics[input[0]][statistic_key] / self.output_statistics[output[0]][statistic_key]
            # lamda_last = self.output_statistics[input[0]][statistic_key]
            # lamda = self.output_statistics[output[0]][statistic_key]
            # weight_ = weight * node_scaled_range[output[0]]
            # bias_ = bias / lamda

            # print(output_statistics[output[0]])
            input_real_range = node_scaled_range[input[0]]
            input_range = output_statistics[input[0]][statistic_key]
            output_range = output_statistics[output[0]][statistic_key]
            demand = 1.0
            w_scale = demand / (output_range + eps) * (input_range / (input_real_range + eps))
            b_scale = 1 / (output_range + eps)
            node_scaled_range[output[0]] = demand
            weight_ = weight * w_scale
            bias_ = bias * b_scale
            # print(output[0],op,input[0], input_range, output_range, demand, input_real_range, w_scale)

            name = initializer[weight_idx].name
            initializer.remove(initializer[weight_idx])
            initializer.insert(weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
            name = initializer[bias_idx].name
            initializer.remove(initializer[bias_idx])
            initializer.insert(bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))

        elif op == 'Add':
            l = output_statistics[output[0]]['shape'][1]
            demand = np.ones(l) if channelwise else 1.0
            node_scaled_range[output[0]] = demand
            output_range = output_statistics[output[0]][statistic_key]

            # node_scaled_range[output[0]] = 1.0
            # lamda = self.output_statistics[output[0]][statistic_key]
            # lamda_lasts = {}
            for i in input:
                if i in output_statistics.keys():
                    # lamda_lasts[i] = self.output_statistics[i][statistic_key]
                    # scale = lamda_lasts[i] / lamda
                    input_real_range = node_scaled_range[i]
                    input_range = output_statistics[i][statistic_key]
                    scale = demand / (output_range + eps) * (input_range / (input_real_range + eps))

                    # print(output[0], op, i, input_range, output_range, demand, input_real_range, scale)

                    idx, _ = find_node_by_output(i, graph)
                    if idx is not None and nodes[idx].op_type in ['Conv', 'Gemm', 'BatchNormalization']:
                        scale_node_weight_bias(topo_analyser, graph, idx, scale)
                    else:
                        scale = scale.reshape(
                            [1, *scale.shape] + [1 for _ in range(len(output_statistics[i]['shape']) - 2)]) \
                            if len(scale.shape) == 1 else scale
                        initializer.append(numpy_helper.from_array(scale.astype(np.float32), "scale_" + i))
                        if idx not in seperate_scale.keys():
                            # seperate_scale[node_idx] = [(i,"scale_"+i,"scaled_"+i)]
                            seperate_scale[node_idx] = {i: ("scale_" + i, "scaled_" + i)}
                        else:
                            # seperate_scale[node_idx].append((i,"scale_"+i,"scaled_"+i))
                            seperate_scale[node_idx][i] = ("scale_" + i, "scaled_" + i)
            pass
        elif op in ['Gather', 'Unsqueeze', 'Shape', 'Concat']:
            continue
        # elif op == "Concat":
        # raise NotImplementedError("Not supported %s yet!"%(op))
        elif op == "Softmax":
            raise NotImplementedError("Not supported %s yet!" % (op))
        else:  # single input single output module
            # print(op,self.output_statistics[output[0]]['shape'])
            input_range = output_statistics[input[0]][statistic_key]
            output_range = output_statistics[output[0]][statistic_key]
            input_scaled_range = node_scaled_range[input[0]]
            output_scaled_range = input_scaled_range / (input_range + eps) * output_range
            node_scaled_range[output[0]] = output_scaled_range

            # print(output[0], op, input[0], input_range, output_range, output_scaled_range)
            # print(op, node_scaled_range[output[0]],'=',input_scaled_range,'/',input_range,'*',output_range)
        # else:
        #     raise NotImplementedError("Not supported yet! %s"%(op))

    if len(seperate_scale.keys()) != 0:
        print("Making new scale node...")

    for node_idx in reversed(seperate_scale.keys()):
        args = {}
        for attr in nodes[node_idx].attribute:
            args[attr.name] = helper.get_attribute_value(attr)
        input = [str(i) if i not in seperate_scale[node_idx].keys() else seperate_scale[node_idx][i][1] \
                 for i in nodes[node_idx].input]

        output = [str(i) for i in nodes[node_idx].output]

        new_node = onnx.helper.make_node(
            nodes[node_idx].op_type,
            inputs=input,
            outputs=output,
            **args
        )
        nodes.remove(nodes[node_idx])
        nodes.insert(node_idx, new_node)

        for i in seperate_scale[node_idx].keys():
            new_node = onnx.helper.make_node(
                'Mul',
                inputs=[seperate_scale[node_idx][i][0], i],
                outputs=[seperate_scale[node_idx][i][1]]
            )
            nodes.insert(node_idx, new_node)
    print("Finished normalizing model!")
    return model

def _pre_onnx_shape_inference(model:onnx.ModelProto):
    '''
    为了对模型进行shape inference，需要先对onnxmodel运行此函数进行准备

    To perform shape inference for model, need to run this function on onnxmodel to prepare

    This function has referenced code in https://github.com/onnx/onnx/issues/2660#issuecomment-605874784
    '''
    if model.ir_version < 4:
        return

    def add_some_graph_info(graph:onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        vi_dict = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            if init.name in inputs:
                continue
            vi = vi_dict.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            tensor_type = vi.type.tensor_type
            if tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
                tensor_type.elem_type = init.data_type
            if not tensor_type.HasField("shape"):
                tensor_type.shape.dim.extend([])
                for dim in init.dims:
                    tensor_type.shape.dim.add().dim_value = dim

        for node in graph.node:
            for attr in node.attribute:
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_some_graph_info(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_some_graph_info(g)
    return add_some_graph_info(model.graph)

class _pt_model(nn.Module):
    def __init__(self, path_or_model, _converter=None):
        super(_pt_model, self).__init__()
        if path_or_model is not None:
            if isinstance(path_or_model, str):
                onnx_model = onnx.load(path_or_model)
            else:
                onnx_model = path_or_model
            self.onnx_model = onnx_model

            self.loaded_weights = load_parameters(self, onnx_model.graph.initializer)
            self.module_list = nn.ModuleList([])
            self.op_tree = {}

            _pre_onnx_shape_inference(onnx_model)
            inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
            self.value_info = inferred_model.graph.value_info
            self.dim_info = {}
            for idx, v in enumerate(self.value_info):
                self.dim_info[v.name] = len(v.type.tensor_type.shape.dim)

            self.graph = defaultdict(list)
            # self.V = set()
            for idx, node in enumerate(onnx_model.graph.node):
                op = node.op_type
                # if op=='MatMul': # TODO temporary
                #     op = 'Gemm'
                (op_idx, inputs, outputs) = getattr(_converter, 'convert_' + op.lower())(node, self)
                for out_seq, output in enumerate(outputs):
                    self.op_tree[str(output)] = (int(op_idx), [str(i) for i in inputs], out_seq)
                for output in outputs:
                    for input in inputs:
                        self.graph[input].append(output)
                # self.V.update(inputs)
                # self.V.update(outputs)
            # print(self.V)
            self.op_tree = json.dumps(self.op_tree)

            self.input_name = [i.name for i in onnx_model.graph.input]
            self.output_name = [i.name for i in onnx_model.graph.output]

            for out in onnx_model.graph.output:
                self.graph[out.name] = []

            def TopologicalSort(G):
                in_degrees = dict((u, 0) for u in G)
                for u in G:
                    for v in G[u]:
                        in_degrees[v] += 1
                Q = [u for u in G if in_degrees[u] == 0]
                res = []
                while Q:
                    u = Q.pop()
                    res.append(u)
                    for v in G[u]:
                        in_degrees[v] -= 1
                        if in_degrees[v] == 0:
                            Q.append(v)
                return res

            self.compute_seq = TopologicalSort(self.graph)
            # print(self.compute_seq)
            # self.tensors = {}

            for k in self.loaded_weights.keys():
                if isinstance(self.loaded_weights[k], torch.FloatTensor):
                    setattr(self, 'P' + k.replace('.', '@'), torch.nn.Parameter(self.loaded_weights[k]))
                else:
                    # print(self.loaded_weights[k])
                    self.register_buffer('P' + k.replace('.', '@'), self.loaded_weights[k])

            self.reserved_tensors_name = list(self.loaded_weights.keys())
            # print('reserve', self.reserved_tensors_name)
            # print(self.tensors)

    def refresh_running_tensor(self):
        self.tensors = {}
        for k in set(self.tensors.keys()) | set(self.reserved_tensors_name):
            if k not in self.reserved_tensors_name:
                del self.tensors[k]
            else:
                self.tensors[k] = getattr(self, 'P' + k.replace('.', '@'))


    def forward(self, input):
        op_tree = json.loads(self.op_tree)

        tensors = {}
        for k in set(tensors.keys()) | set(self.reserved_tensors_name):
            if k not in self.reserved_tensors_name:
                del tensors[k]
            else:
                tensors[k] = getattr(self, 'P' + k.replace('.', '@'))
        
        # self.refresh_running_tensor()
        if not isinstance(input, list) or not isinstance(input, tuple):
            input = [input]
        for i, n in enumerate(self.input_name):
            tensors[n] = input[i]
        for name in self.compute_seq:
            if name in op_tree.keys():
                op_idx, inputs, out_seq = op_tree[name]
                # print(name,op_idx, inputs,out_seq)
                args = []
                for input in inputs:
                    args.append(tensors[input])
                # print(len(args))
                # print(type(args[0]))
                result = self.module_list[op_idx](*args)
                
                if not isinstance(result, tuple):
                    tensors[name] = result
                    # print('    %s = self.module_list[%d] (%s)'%(name,op_idx,inputs))
                else:
                    tensors[name] = result[out_seq]
                    # print('    %s = self.module_list[%d] (%s)[%d]'%(name,op_idx,inputs,out_seq) )

        if len(self.output_name) == 1:
            return tensors[self.output_name[0]]
        else:
            ret = []
            for output in self.output_name:
                ret.append(tensors[output])
            return ret

    def reduce(self):
        import copy
        net = _pt_model(None)
        for k in self.reserved_tensors_name:
            if isinstance(self.loaded_weights[k], torch.FloatTensor):
                setattr(net, 'P' + k.replace('.', '@'),
                        torch.nn.Parameter(getattr(self,'P' + k.replace('.', '@')).data.detach().clone()) )
            else:
                net.register_buffer('P' + k.replace('.', '@'),
                                    getattr(self,'P' + k.replace('.', '@')).data.clone() )

        net.compute_seq = copy.deepcopy(self.compute_seq)

        net.input_name = copy.deepcopy(self.input_name)
        net.output_name = copy.deepcopy(self.output_name)
        net.module_list = copy.deepcopy(self.module_list)
        net.op_tree = copy.deepcopy(self.op_tree)
        net.reserved_tensors_name = copy.deepcopy(self.reserved_tensors_name)
        return net

def load_parameters(model:_pt_model, initializer):
    param_dict = {}
    for init in initializer:
        param_dict[init.name] = torch.from_numpy(numpy_helper.to_array(init).copy())
    return param_dict

class _o2p_converter:
    def __init__(self):
        '''
        * :ref:`API in English <ONNX_Converter.__init__-en>`

        .. _ONNX_Converter.__init__-cn:

        该类主要将onnx模型转换为Pytorch的ANN模型，从而转换为SpikingJelly的SNN模型
        链接中 [＃f1]_ 提供了一个onnx-pytorch转换的主要版本。更复杂的版本可以在这里找到。
        大多数使用过的onnx运算符已在此处定义，但仍然有一些未被覆盖，或没有被完美实现
        用户可以通过添加如下面例子所示的静态方法来定义您的例外情况

        * :ref:`API in English <ONNX_Converter.__init__-cn>`

        .. _ONNX_Converter.__init__-en:

        This class mainly convert an onnx model to Pytorch ANN model, and thus to SpikingJelly SNN model
        The link [#f1]_ has provided a primary version of onnx-pytorch conversion. More complex version can be found here.
        Most used onnx operators has covered here, yet still there are some left, or not being defined perfectly
        User can define your exceptions by adding static method like below

        .. [#f1] https://gist.github.com/qinjian623/6aa777037534c1c1dccbb66f832e93b8
        '''
        pass

    def add_method(self, op_name, func):
        setattr(self, 'convert_'+op_name, staticmethod(func))

    @staticmethod
    def convert_conv(node, model:_pt_model):
        attr_map = {
            "pads": "padding",
            "strides": "stride",
            "kernel_shape": "kernel_size",
            "group": "groups",
            "dilations": "dilation"
        }
        assert len(node.output) == 1
        with_bias = False
        if len(node.input) == 3:
            with_bias = True
            bias = model.loaded_weights[node.input[2]]
            del model.loaded_weights[node.input[2]]
        weight = model.loaded_weights[node.input[1]]
        del model.loaded_weights[node.input[1]]
        in_channels = weight.shape[1]
        out_channels = weight.shape[0]
        kwargs = {}
        for att in node.attribute:
            kwargs[attr_map[att.name]] = list(att.ints) if att.name != 'group' else att.i
        if 'padding' in kwargs:
            assert(kwargs["padding"][0]==kwargs["padding"][2] and kwargs["padding"][1]==kwargs["padding"][3])
            kwargs["padding"] = kwargs["padding"][0],kwargs["padding"][1]
        groups = 1 if 'groups' not in kwargs else kwargs['groups']
        in_channels *= groups
        conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=with_bias)
        conv.weight.data = weight
        if with_bias:
            conv.bias.data = bias
        model.module_list.append(conv)
        return len(model.module_list)-1, node.input[:1], node.output

    @staticmethod
    def convert_relu(node, model:_pt_model):
        relu = nn.ReLU()
        model.module_list.append(relu)
        return len(model.module_list)-1, node.input, node.output

    @staticmethod
    def convert_prelu(node, model:_pt_model):
        weight = model.loaded_weights[node.input[1]]
        del model.loaded_weights[node.input[1]]
        prelu = nn.PReLU()
        prelu.weight.data = weight
        model.module_list.append(prelu)
        return len(model.module_list) - 1, node.input[:-1], node.output

    @staticmethod
    def convert_shape(node, model:_pt_model):
        shape = Shape()
        model.module_list.append(shape)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_gather(node, model:_pt_model):
        attr_map = {
            "axis": "dim"
        }
        kwargs = {}
        for att in node.attribute:
            if att.name in attr_map:
                kwargs[attr_map[att.name]] = att.f
        gather = Gather(**kwargs)
        model.module_list.append(gather)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_unsqueeze(node, model:_pt_model):
        attr_map = {
            "axes": "dim"
        }
        kwargs = {}
        for att in node.attribute:
            if att.name in attr_map:
                kwargs[attr_map[att.name]] = att.f
        unsqueeze = Unsqueeze(**kwargs)
        model.module_list.append(unsqueeze)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_concat(node, model:_pt_model):
        attr_map = {
            "axis": "dim"
        }
        kwargs = {}
        for att in node.attribute:
            if att.name in attr_map:
                kwargs[attr_map[att.name]] = att.f

        concat = Concat(**kwargs)
        model.module_list.append(concat)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_reshape(node, model:_pt_model):
        reshape = Reshape()
        model.module_list.append(reshape)
        return len(model.module_list) - 1, node.input, node.output

    # @staticmethod
    # def convert_matmul(node, model:_pt_model):
    #     class MatMul(nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #         def forward(self,input1,input2):
    #             return input1 @ input2
    #     mul = MatMul()
    #     model.module_list.append(mul)
    #     return len(model.module_list)-1, node.input, node.output


    @staticmethod
    def convert_batchnormalization(node, model:_pt_model):
        attr_map = {
            "epsilon": "eps",
            "momentum": "momentum"
        }
        assert len(node.input) == 5
        assert len(node.output) == 1
        weight = model.loaded_weights[node.input[1]]
        bias = model.loaded_weights[node.input[2]]
        running_mean = model.loaded_weights[node.input[3]]
        running_var = model.loaded_weights[node.input[4]]
        del model.loaded_weights[node.input[1]]
        del model.loaded_weights[node.input[2]]
        del model.loaded_weights[node.input[3]]
        del model.loaded_weights[node.input[4]]
        dim = weight.shape[0]
        kwargs = {}
        # _check_attr(node.attribute, rebuild_batchnormalization.bn_attr_map)
        for att in node.attribute:
            if att.name in attr_map:
                kwargs[attr_map[att.name]] = att.f
        bn = None
        if model.dim_info[node.output[0]] == 5:
            bn = nn.BatchNorm3d(num_features=dim)
        elif model.dim_info[node.output[0]] == 4:
            bn = nn.BatchNorm2d(num_features=dim)
        elif model.dim_info[node.output[0]] == 2 or model.dim_info[node.output[0]] == 3:
            bn = nn.BatchNorm1d(num_features=dim)
        bn.weight.data = weight
        bn.bias.data = bias
        bn.running_mean.data = running_mean
        bn.running_var.data = running_var
        model.module_list.append(bn)
        return len(model.module_list)-1, node.input[:1], node.output

    @staticmethod
    def convert_add(node, model:_pt_model):
        add = Add()
        model.module_list.append(add)
        return len(model.module_list)-1, node.input, node.output

    @staticmethod
    def convert_mul(node, model:_pt_model):
        mul = Mul()
        model.module_list.append(mul)
        return len(model.module_list)-1, node.input, node.output

    @staticmethod
    def convert_averagepool(node, model:_pt_model):
        attr_map = {
            "pads": "padding",
            "strides": "stride",
            "kernel_shape": "kernel_size",
        }
        kwargs = {}
        for att in node.attribute:
            kwargs[attr_map[att.name]] = list(att.ints)
        if 'padding' in kwargs:
            assert (kwargs["padding"][0] == kwargs["padding"][2] and kwargs["padding"][1] == kwargs["padding"][3])
            kwargs["padding"] = kwargs["padding"][0], kwargs["padding"][1]
        ap = nn.AvgPool2d(**kwargs)
        model.module_list.append(ap)
        return len(model.module_list)-1, node.input, node.output

    @staticmethod
    def convert_globalaveragepool(node, model:_pt_model):
        gap = nn.AdaptiveAvgPool2d((1, 1))
        model.module_list.append(gap)
        model.module_list.append(gap)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_maxpool(node, model:_pt_model):
        attr_map = {
            "pads": "padding",
            "strides": "stride",
            "kernel_shape": "kernel_size",
        }
        kwargs = {}
        for att in node.attribute:
            kwargs[attr_map[att.name]] = list(att.ints)
        if 'padding' in kwargs:
            assert (kwargs["padding"][0] == kwargs["padding"][2] and kwargs["padding"][1] == kwargs["padding"][3])
            kwargs["padding"] = kwargs["padding"][0], kwargs["padding"][1]
        ap = nn.MaxPool2d(**kwargs)
        model.module_list.append(ap)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_flatten(node, model:_pt_model):
        if len(node.attribute) == 0:
            axis = 1
        else:
            axis = node.attribute[0].i
        if axis==1:
            flatten = nn.Flatten()
            model.module_list.append(flatten)
            return len(model.module_list)-1, node.input, node.output
        else:
            raise NotImplementedError("Not Implemented yet!")

    @staticmethod
    def convert_gemm(node, model:_pt_model):
        weight = model.loaded_weights[node.input[1]]
        bias = model.loaded_weights[node.input[2]]
        del model.loaded_weights[node.input[2]]
        del model.loaded_weights[node.input[1]]
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        linear = nn.Linear(in_features=in_features, out_features=out_features)
        linear.weight.data = weight
        linear.bias.data = bias
        model.module_list.append(linear)
        return len(model.module_list)-1, node.input[:1], node.output

    @staticmethod
    def convert_pad(node, model:_pt_model):
        mode = node.attribute[0].s
        pads = list(node.attribute[1].ints)
        value = node.attribute[2].f
        try:
            assert(mode == b'constant')
            assert(sum(pads[:4]) == 0)
        except AssertionError:
            print("Now only support converting to nn.ConstantPad2d")
        pad = nn.ConstantPad2d([*pads[2:4],*pads[3:5]],value)
        model.module_list.append(pad)
        return len(model.module_list)-1, node.input, node.output
