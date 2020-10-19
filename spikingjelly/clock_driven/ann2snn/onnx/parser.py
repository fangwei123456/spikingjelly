import re
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import collections
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
import os
import tqdm
import onnxruntime as ort


class Pytorch_Wrapper(nn.Module):
    def __init__(self, original_nn, z_norm=None):
        super(Pytorch_Wrapper, self).__init__()
        self.z_norm = z_norm
        if z_norm is not None:
            (z_norm_mean,z_norm_std) = z_norm
            self.z_norm_mean = torch.from_numpy(np.array(z_norm_mean).astype(np.float32))
            self.z_norm_std = torch.from_numpy(np.array(z_norm_std).astype(np.float32))
            self.z_score_layer = nn.BatchNorm2d(num_features=len(z_norm_std))
            self.z_score_layer.weight.data = torch.ones_like(self.z_score_layer.weight.data)
            self.z_score_layer.bias.data = torch.zeros_like(self.z_score_layer.bias.data)
            self.z_score_layer.running_var.data = torch.pow(self.z_norm_std,exponent=2) \
                                              - self.z_score_layer.eps
            self.z_score_layer.running_mean.data = self.z_norm_mean
        self.nn = original_nn

    def forward(self,x):
        if self.z_norm is not None:
            x = self.z_score_layer(x)
        x = self.nn(x)
        return x

    def export_to_onnx(self, dir, model_name, dump_input, single_input=True):
        onnx_model_file = os.path.join(dir, "%s.onnx" % (model_name))
        if single_input:
            dynamic_axes = {'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            torch.onnx.export(self, dump_input, onnx_model_file,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes=dynamic_axes)
            print("ONNX Model saved to",onnx_model_file)
        else:
            raise NotImplementedError("Model with multiple inputs not supported yet!")
        return onnx_model_file

class ONNX_Parser:
    def __init__(self,filename):
        self.onnx_model = onnx.load(filename)
        onnx.checker.check_model(self.onnx_model)
        self.graph = self.onnx_model.graph
        self.node = self.graph.node
        self.initializer = self.graph.initializer
        self.topo_analyser = TopologyAnalyser()
        self.move_constant_to_initializer()
        self.topo_analyser.analyse(self.graph)
        self.output_debug = {}

    def test_output(self,numpy_tensor):
        ort_session = ort.InferenceSession(self.onnx_model.SerializeToString())
        outputs = ort_session.run(None, {'input': numpy_tensor})
        return outputs

    def print_model(self):
        print(onnx.helper.printable_graph(self.graph))
        
    def absorb_bn(self):
        print("\nAbsorbing BatchNorm Parameters...\n")
        for mn in tqdm.tqdm(reversed(self.topo_analyser.module_output.keys())):
            if self.topo_analyser.module_op[mn] == 'BatchNormalization':
                # print(mn)
                pre_m = self.topo_analyser.find_pre_module(mn)
                next_m = self.topo_analyser.find_next_module(mn)

                bn_weight_idx = self.topo_analyser.param_idx[self.node[self.topo_analyser.module_idx[mn]].input[1]]
                bn_weight = np.array(numpy_helper.to_array(self.initializer[bn_weight_idx]))
                bn_bias_idx = self.topo_analyser.param_idx[self.node[self.topo_analyser.module_idx[mn]].input[2]]
                bn_bias = np.array(numpy_helper.to_array(self.initializer[bn_bias_idx]))
                bn_mean_idx = self.topo_analyser.param_idx[self.node[self.topo_analyser.module_idx[mn]].input[3]]
                bn_mean = np.array(numpy_helper.to_array(self.initializer[bn_mean_idx]))
                bn_var_idx = self.topo_analyser.param_idx[self.node[self.topo_analyser.module_idx[mn]].input[4]]
                bn_var = np.array(numpy_helper.to_array(self.initializer[bn_var_idx]))
                bn_eps = self.node[self.topo_analyser.module_idx[mn]].attribute[0].f
                bn_std = np.sqrt(bn_var + bn_eps)

                if len(pre_m) == 1 and list(pre_m)[0].split(':')[0] in ['Conv', 'Gemm']:
                    pre_mn = list(pre_m)[0].split(':')[1]
                    # print(node[self.topo_analyser.module_idx[pre_mn]].input, node[self.topo_analyser.module_idx[pre_mn]].output)
                    # print(node[self.topo_analyser.module_idx[mn]].input,node[self.topo_analyser.module_idx[mn]].output)

                    weight_idx = self.topo_analyser.param_idx[self.node[self.topo_analyser.module_idx[pre_mn]].input[1]]
                    weight = np.array(numpy_helper.to_array(self.initializer[weight_idx]))
                    if len(self.node[self.topo_analyser.module_idx[pre_mn]].input) == 2:
                        bias = None
                    else:
                        bias_idx = self.topo_analyser.param_idx[self.node[self.topo_analyser.module_idx[pre_mn]].input[2]]
                        bias = np.array(numpy_helper.to_array(self.initializer[bias_idx]))
                    wrsp_args = (-1, 1) if len(weight.shape) == 2 else (-1, 1, 1, 1)

                    weight_ = weight * bn_weight.reshape(*wrsp_args) / bn_std.reshape(*wrsp_args)
                    bias_ = ((bias if bias is not None else 0) - bn_mean.reshape(-1)) * bn_weight.reshape(
                        -1) / bn_std.reshape(-1) \
                            + bn_bias.reshape(-1)
                    #print(weight_.shape, bias_.shape)

                    assert (list(pre_m)[0].split(':')[0] in ['Conv', 'Gemm'])

                    args = {}
                    for attr in self.node[self.topo_analyser.module_idx[pre_mn]].attribute:
                        args[attr.name] = helper.get_attribute_value(attr)
                    new_node = onnx.helper.make_node(
                        list(pre_m)[0].split(':')[0],
                        inputs=[self.node[self.topo_analyser.module_idx[pre_mn]].input[0], pre_mn + ".new.weight", pre_mn + ".new.bias"],
                        outputs=[self.node[self.topo_analyser.module_idx[mn]].output[0]],
                        **args
                    )
                    self.initializer.append(numpy_helper.from_array(weight_.astype(np.float32), pre_mn + ".new.weight"))
                    self.initializer.append(numpy_helper.from_array(bias_.astype(np.float32), pre_mn + ".new.bias"))
                    self.node.remove(self.node[self.topo_analyser.module_idx[pre_mn]])
                    self.node.insert(self.topo_analyser.module_idx[pre_mn], new_node)
                    self.node.remove(self.node[self.topo_analyser.module_idx[mn]])
                else:
                    #print("seperate batchnormscale")

                    weight_ = bn_weight / bn_std
                    bias_ = bn_bias - bn_weight * bn_mean / bn_std
                    name = self.initializer[bn_weight_idx].name
                    self.initializer.remove(self.initializer[bn_weight_idx])
                    self.initializer.insert(bn_weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
                    name = self.initializer[bn_bias_idx].name
                    self.initializer.remove(self.initializer[bn_bias_idx])
                    self.initializer.insert(bn_bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))
                    name = self.initializer[bn_mean_idx].name
                    self.initializer.remove(self.initializer[bn_mean_idx])
                    self.initializer.insert(bn_mean_idx,
                                       numpy_helper.from_array(np.zeros_like(bn_mean).astype(np.float32), name))
                    name = self.initializer[bn_var_idx].name
                    self.initializer.remove(self.initializer[bn_var_idx])
                    self.initializer.insert(bn_var_idx,
                                       numpy_helper.from_array(np.ones_like(bn_var).astype(np.float32), name))
        self.remove_unreferenced_initializer()
        self.update_topology()
        print("Finished absorbing BatchNorm Parameters!")

    def update_topology(self):
        self.topo_analyser = TopologyAnalyser()
        self.move_constant_to_initializer()
        self.topo_analyser.analyse(self.graph)

    def remove_unreferenced_initializer(self):
        in_graph = set()
        in_initializer = set()
        for node in self.node:
            in_graph.update(node.input)
            in_graph.update(node.output)
        for init in self.initializer:
            in_initializer.add(init.name)
        not_in_graph = in_initializer - in_graph
        l = len(self.initializer)
        for i in range(l - 1, -1, -1):
            if self.initializer[i].name in not_in_graph:
                self.initializer.remove(self.initializer[i])
        pass  # TODO

    def save_model(self, filename=None):
        """
        Saves a model as a file or bytes.

        :param model: *ONNX* model#TODO
        :param filename: filename or None to return bytes
        :return: bytes
        """
        content = self.onnx_model.SerializeToString()
        if filename is not None:
            if hasattr(filename, 'write'):
                filename.write(content)
            else:
                with open(filename, "wb") as f:
                    f.write(content)
        return content

    def move_constant_to_initializer(self):
        constant_idx = []
        for idx, n in enumerate(self.graph.node):
            op = n.op_type
            if op == 'Constant':
                constant_idx.append(idx)
        if len(constant_idx):
            for idx in reversed(constant_idx):
                n = self.graph.node[idx]
                self.graph.initializer.append(
                    numpy_helper.from_array(numpy_helper.to_array(n.attribute[0].t), n.output[0]))
                self.graph.node.remove(n)

    def get_intermediate_output_statistics(self, numpy_tensor, channelwise = False,test=False):
        output_needed_module = {}
        output_needed_all_input = {}
        for idx, node in enumerate(self.node):
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
        # print(output_needed_module)
        # print(output_needed_all_input)

        self.output_statistics = {}
        if not channelwise:
            statistic = {'shape': numpy_tensor.shape,
                         'min': np.min(numpy_tensor),
                         'max': np.max(numpy_tensor) if np.max(numpy_tensor) > 0 else np.abs(np.min(numpy_tensor)),
                         '99.9': np.percentile(numpy_tensor, 99.9)}
        else:
            axis_args = (0, 2, 3) if len(numpy_tensor.shape) == 4 else (0)
            statistic = {'shape': numpy_tensor.shape,
                         'min': np.min(numpy_tensor, axis=axis_args),
                         'max': np.max(numpy_tensor, axis=axis_args),
                         '99.9': np.percentile(numpy_tensor, 99.9, axis=axis_args)}
        self.output_statistics['input'] = statistic
        print("\nGetting intermediate output statistics...\n")
        for out in tqdm.tqdm(output_needed_module.keys()):
            keep_nodes = [self.node[i] for i in list(output_needed_module[out])]
            keep_initializer = [init for init in self.initializer
                                if init.name in list(output_needed_all_input[out])]
            var_out = []
            value_info = onnx.ValueInfoProto()
            value_info.name = out
            var_out.append(value_info)
            graph = onnx.helper.make_graph(keep_nodes, self.graph.name, self.graph.input,
                                           var_out, keep_initializer)
            tmp_model = onnx.helper.make_model(graph)
            tmp_model.ir_version = self.onnx_model.ir_version
            tmp_model.producer_name = self.onnx_model.producer_name
            tmp_model.producer_version = self.onnx_model.producer_version
            tmp_model.domain = self.onnx_model.domain
            tmp_model.model_version = self.onnx_model.model_version
            tmp_model.doc_string = self.onnx_model.doc_string
            if len(tmp_model.metadata_props) > 0:
                values = {p.key: p.value for p in self.onnx_model.metadata_props}
                onnx.helper.set_model_props(tmp_model, values)
            # fix opset import
            for oimp in self.onnx_model.opset_import:
                op_set = tmp_model.opset_import.add()
                op_set.domain = oimp.domain
                op_set.version = oimp.version

            ort_session = ort.InferenceSession(tmp_model.SerializeToString())
            outputs = ort_session.run(None, {'input': numpy_tensor})
            if test:
                print(out,outputs[0].reshape(1,-1)[0,10:20])
                self.output_debug[out] = outputs[0]
            # print(outputs[0].shape)
            if not channelwise:
                statistic = {'shape': outputs[0].shape,
                             'min': np.min(outputs[0]),
                             'max': np.max(outputs[0]) if np.max(outputs[0]) > 0 else np.abs(np.min(outputs[0])),
                             '99.9': np.percentile(outputs[0], 99.9)} if np.percentile(outputs[0], 99.9) > 0 else np.abs(np.min(outputs[0])),
                # print(np.percentile(outputs[0],99.9))
            else:
                axis_args = (0, 2, 3) if len(outputs[0].shape) == 4 else (0)
                statistic = {'shape': outputs[0].shape,
                             'min': np.min(outputs[0], axis=axis_args),
                             'max': np.max(outputs[0], axis=axis_args),
                             '99.9': np.percentile(outputs[0], 99.9, axis=axis_args)}
                # print(np.max(statistic['max']),np.max(outputs[0]))
            self.output_statistics[out] = statistic
        print("Finished getting intermediate output statistics!")

    def normalize_model(self,robust_norm=True,use_NSIF=True,channelwise=False,eps=1e-5):
        if robust_norm:
            statistic_key = '99.9'
        else:
            statistic_key = 'max'
        node_scaled_range = {}
        seperate_scale = collections.OrderedDict()
        print("\nNormalizing model...\n")
        for node_idx, node in enumerate(tqdm.tqdm(self.node)):
            output = node.output
            input = node.input
            op = node.op_type
            if input[0] == 'input': # single input model
                l = self.output_statistics[input[0]]['shape'][1]
                node_scaled_range[input[0]] = np.ones(l) if channelwise else 1.0 \
                                              * self.output_statistics[input[0]][statistic_key]

            if op in ['Conv', 'Gemm']:
                weight_idx = self.topo_analyser.param_idx[input[1]]
                bias_idx = self.topo_analyser.param_idx[input[2]] if len(input) == 3 else None
                weight = np.array(numpy_helper.to_array(self.initializer[weight_idx]))
                bias = np.array(numpy_helper.to_array(self.initializer[bias_idx])) if bias_idx is not None else None

                l = self.output_statistics[output[0]]['shape'][1]
                input_real_range = node_scaled_range[input[0]]
                input_range = self.output_statistics[input[0]][statistic_key]
                output_range = self.output_statistics[output[0]][statistic_key]
                demand = np.ones(l) if channelwise else 1.0
                w_scale = demand / (output_range+eps) * (input_range / (input_real_range+eps)) if not channelwise else \
                    (demand / (output_range+eps)).reshape(-1, 1).dot((input_range / (input_real_range+eps)).reshape(1, -1))
                w_scale = w_scale.reshape([*w_scale.shape, 1, 1]) if len(weight.shape) == 4 else w_scale
                b_scale = 1/ (output_range+eps)
                node_scaled_range[output[0]] = demand
                #print(output[0],op,input[0], input_range, output_range, demand, input_real_range, w_scale)
                
                # node_scaled_range[output[0]] = node_scaled_range[input[0]] * self.output_statistics[input[0]][statistic_key] / self.output_statistics[output[0]][statistic_key]
                # lamda_last = node_scaled_range[output[0]]
                # lamda = self.output_statistics[output[0]][statistic_key]
                # print(op,output,lamda_last , lamda,node_scaled_range[output[0]])
                #
                # w_scale = lamda_last / lamda if not channelwise else (1.0 / lamda).reshape(-1, 1).dot(
                #     lamda_last.reshape(1, -1))
                # w_scale = w_scale.reshape([*w_scale.shape, 1, 1]) if len(weight.shape) == 4 else w_scale
                # b_scale = lamda
                weight_ = weight * w_scale
                
                name = self.initializer[weight_idx].name
                self.initializer.remove(self.initializer[weight_idx])
                self.initializer.insert(weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
                if bias is not None:
                    bias_ = bias * b_scale
                    name = self.initializer[bias_idx].name
                    self.initializer.remove(self.initializer[bias_idx])
                    self.initializer.insert(bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))

            elif op == 'BatchNormalization':  # var=1 mean=0
                weight_idx = self.topo_analyser.param_idx[input[1]]
                bias_idx = self.topo_analyser.param_idx[input[2]]
                weight = np.array(numpy_helper.to_array(self.initializer[weight_idx]))
                bias = np.array(numpy_helper.to_array(self.initializer[bias_idx]))
                if not use_NSIF:
                    Warning(
                        "Not using IFNode with negative spike can result in significant accuracy loss due to vanilla IFNode's half wave feature!")
                
                # node_scaled_range[output[0]] = node_scaled_range[input[0]] * self.output_statistics[input[0]][statistic_key] / self.output_statistics[output[0]][statistic_key]
                # lamda_last = self.output_statistics[input[0]][statistic_key]
                # lamda = self.output_statistics[output[0]][statistic_key]
                # weight_ = weight * node_scaled_range[output[0]]
                # bias_ = bias / lamda


                input_real_range = node_scaled_range[input[0]]
                input_range = self.output_statistics[input[0]][statistic_key]
                output_range = self.output_statistics[output[0]][statistic_key]
                demand = 1.0
                w_scale = demand / (output_range+eps) * (input_range / (input_real_range+eps))
                b_scale = 1 / (output_range+eps)
                node_scaled_range[output[0]] = demand
                weight_ = weight * w_scale
                bias_ = bias * b_scale
                #print(output[0],op,input[0], input_range, output_range, demand, input_real_range, w_scale)

                name = self.initializer[weight_idx].name
                self.initializer.remove(self.initializer[weight_idx])
                self.initializer.insert(weight_idx, numpy_helper.from_array(weight_.astype(np.float32), name))
                name = self.initializer[bias_idx].name
                self.initializer.remove(self.initializer[bias_idx])
                self.initializer.insert(bias_idx, numpy_helper.from_array(bias_.astype(np.float32), name))

            elif op == 'Add':
                l = self.output_statistics[output[0]]['shape'][1]
                demand = np.ones(l) if channelwise else 1.0
                node_scaled_range[output[0]] = demand
                output_range = self.output_statistics[output[0]][statistic_key]

                # node_scaled_range[output[0]] = 1.0
                # lamda = self.output_statistics[output[0]][statistic_key]
                # lamda_lasts = {}
                for i in input:
                    if i in self.output_statistics.keys():
                        # lamda_lasts[i] = self.output_statistics[i][statistic_key]
                        # scale = lamda_lasts[i] / lamda
                        input_real_range = node_scaled_range[i]
                        input_range = self.output_statistics[i][statistic_key]
                        scale = demand / (output_range+eps) * (input_range / (input_real_range+eps))

                        #print(output[0], op, i, input_range, output_range, demand, input_real_range, scale)

                        idx, _ = self.find_node_by_output(i, self.graph)
                        if idx is not None and self.node[idx].op_type in ['Conv', 'Gemm', 'BatchNormalization']:
                            self.scale_node_weight_bias(self.topo_analyser, self.graph, idx, scale)
                        else:
                            scale = scale.reshape(
                                [1, *scale.shape] + [1 for _ in range(len(self.output_statistics[i]['shape']) - 2)]) \
                                if len(scale.shape) == 1 else scale
                            self.initializer.append(numpy_helper.from_array(scale.astype(np.float32), "scale_" + i))
                            if idx not in seperate_scale.keys():
                                # seperate_scale[node_idx] = [(i,"scale_"+i,"scaled_"+i)]
                                seperate_scale[node_idx] = {i: ("scale_" + i, "scaled_" + i)}
                            else:
                                # seperate_scale[node_idx].append((i,"scale_"+i,"scaled_"+i))
                                seperate_scale[node_idx][i] = ("scale_" + i, "scaled_" + i)
                pass

            elif op == "Concat":
                raise NotImplementedError("Not supported yet! Concat")
            elif op == "Softmax":
                raise NotImplementedError("Not supported yet! Softmax")
            else: # single input single output module
                #print(op,self.output_statistics[output[0]]['shape'])
                input_range = self.output_statistics[input[0]][statistic_key]
                output_range = self.output_statistics[output[0]][statistic_key]
                input_scaled_range = node_scaled_range[input[0]]
                output_scaled_range = input_scaled_range / (input_range+eps) * output_range
                node_scaled_range[output[0]] = output_scaled_range

                #print(output[0], op, input[0], input_range, output_range, output_scaled_range)
                #print(op, node_scaled_range[output[0]],'=',input_scaled_range,'/',input_range,'*',output_range)
            # else:
            #     raise NotImplementedError("Not supported yet! %s"%(op))

        if len(seperate_scale.keys())!=0:
            print("Making new scale node...")

        for node_idx in reversed(seperate_scale.keys()):
            args = {}
            for attr in self.node[node_idx].attribute:
                args[attr.name] = helper.get_attribute_value(attr)
            input = [str(i) if i not in seperate_scale[node_idx].keys() else seperate_scale[node_idx][i][1] \
                     for i in self.node[node_idx].input]

            output = [str(i) for i in self.node[node_idx].output]

            new_node = onnx.helper.make_node(
                self.node[node_idx].op_type,
                inputs=input,
                outputs=output,
                **args
            )
            self.node.remove(self.node[node_idx])
            self.node.insert(node_idx, new_node)

            for i in seperate_scale[node_idx].keys():
                new_node = onnx.helper.make_node(
                    'Mul',
                    inputs=[seperate_scale[node_idx][i][0], i],
                    outputs=[seperate_scale[node_idx][i][1]]
                )
                self.node.insert(node_idx, new_node)
        print("Finished normalizing model!")

    @staticmethod # TODO whether static
    def find_node_by_output(output_name, onnx_graph):
        flag = False
        idx, node = None, None
        for idx, node in enumerate(onnx_graph.node):
            if output_name in node.output:
                flag = True
                break
        if not flag:
            idx, node = None, None
        return idx, node

    @staticmethod # TODO whether static
    def scale_node_weight_bias(topo_analyser, onnx_graph, node_idx, scale):
        initializer = onnx_graph.initializer
        node = onnx_graph.node[node_idx]
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


class TopologyAnalyser:
    def __init__(self):
        self.data_nodes = []
        self.module_output = collections.OrderedDict()
        self.module_op = collections.OrderedDict()
        self.module_idx = collections.OrderedDict()
        self.param_idx = collections.OrderedDict()
        self.edge = collections.OrderedDict()
        self.reverse_edge = collections.OrderedDict() # 快速计算前驱结点

    def add_data_node(self,a):
        if not a in self.data_nodes:
            self.data_nodes.append(a)

    def insert(self, a, b, info=None):
        self.add_data_node(a)
        self.add_data_node(b)
        if a not in self.edge.keys():
            self.edge[a] = [(b,info)]
        else:
            self.edge[a].append((b,info))
        if b not in self.reverse_edge.keys():
            self.reverse_edge[b] = [a]
        else:
            self.reverse_edge[b].append(a)

    def findNext(self, id):
        if isinstance(id,str):
            if id in self.edge.keys():
                return self.edge[id]
            else:
                return []
        elif isinstance(id,list):
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
        elif isinstance(id,list):
            for i in id:
                l += self.findPre(i)
        return l

    def find_pre_module(self,module_name):
        if module_name in self.module_output.keys():
            ids = self.module_output[module_name]
            return set(['%s:%s'%(k[1]['op'],k[1]['param_module_name']) for k in self.findPre(ids)])
        else:
            return set()

    def find_next_module(self,module_name):
        if module_name in self.module_output.keys():
            ids = self.module_output[module_name]
            return set(['%s:%s'%(k[1]['op'],k[1]['param_module_name']) for k in self.findNext(ids)])
        else:
            return set()


    def update_module_idx(self,onnx_graph):
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

    def analyse(self,onnx_graph):  # 输入的onnx graph需要保证所以常量在已经在initializer中
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
            module_name = l#[1:]
            module_name.replace(' ','')
        return module_name















