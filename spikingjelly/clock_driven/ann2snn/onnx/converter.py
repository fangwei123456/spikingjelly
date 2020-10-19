import onnx
import torch
import torch.nn as nn
import onnx.numpy_helper as numpy_helper
import spikingjelly.clock_driven.neuron as neuron
from collections import defaultdict
import spikingjelly.clock_driven.ann2snn.modules as snn_modules

def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)
    return add_const_value_infos_to_graph(model.graph)


def load_parameters(initializer):
    param_dict = {}
    for init in initializer:
        param_dict[init.name] = torch.from_numpy(numpy_helper.to_array(init).copy())
    return param_dict


class ONNXConvertedModel(nn.Module):
    def __init__(self, path_or_model):
        super(ONNXConvertedModel, self).__init__()
        self.is_ann = True
        if isinstance(path_or_model,str):
            onnx_model = onnx.load(path_or_model)
        else:
            onnx_model = path_or_model
        self.onnx_model = onnx_model

        self.loaded_weights = load_parameters(onnx_model.graph.initializer)
        self.module_list = nn.ModuleList([])
        self.op_tree = {}

        add_value_info_for_constants(onnx_model)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        self.value_info = inferred_model.graph.value_info
        self.dim_info = {}
        for idx,v in enumerate(self.value_info):
            self.dim_info[v.name]=len(v.type.tensor_type.shape.dim)

        self.graph = defaultdict(list)
        #self.V = set()
        for idx,node in enumerate(onnx_model.graph.node):
            op = node.op_type
            # if op=='MatMul': # TODO temporary
            #     op = 'Gemm'
            (op_idx,inputs,outputs) = getattr(ONNX_Converter,'convert_'+op.lower())(node, self)
            for out_seq,output in enumerate(outputs):
                #self.op_tree[output] = (op_idx, inputs, out_seq)
                self.op_tree[output] = (op_idx, inputs, out_seq)
            for output in outputs:
                for input in inputs:
                    self.graph[input].append(output)
            # self.V.update(inputs)
            # self.V.update(outputs)
        # print(self.V)

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
        #print(self.compute_seq)

        for k in self.loaded_weights.keys():
                self.loaded_weights[k] = torch.nn.Parameter(self.loaded_weights[k])


    def forward(self, input):
        self.tensors = self.loaded_weights.copy()
        #print(self.loaded_weights)
        if not isinstance(input,list) or not isinstance(input,tuple):
            input = [input]
        for i,n in enumerate(self.input_name):
            self.tensors[n] = input[i]
        for name in self.compute_seq:
            if name in self.op_tree.keys():
                op_idx, inputs, out_seq = self.op_tree[name]
                #print(name,op_idx, inputs)
                args = []
                for input in inputs:
                    args.append(self.tensors[input])
                #print(len(args))
                result = self.module_list[op_idx](*args)
                if not isinstance(result, tuple):
                    self.tensors[name] = result
                else:
                    self.tensors[name] = result[out_seq]

        if len(self.output_name)==1:
            return self.tensors[self.output_name[0]]
        else:
            ret = []
            for output in self.output_name:
                ret.append(self.tensors[output])
            return ret

    def to(self, device):
        super().to(device)
        for k in self.loaded_weights.keys():
                self.loaded_weights[k] = self.loaded_weights[k].to(device)
        return self

    def convert_to_snn(self):
        if not self.is_ann:
            return self
        for name, module in self.module_list._modules.items():
            if "BatchNorm" in module.__class__.__name__:
                new_module = nn.Sequential(module, neuron.NSIFNode(v_threshold=(-1.0, 1.0), v_reset=None))
                self.module_list._modules[name] = new_module
            if module.__class__.__name__ == "AvgPool2d":
                # new_module = nn.Sequential(module, neuron.NSIFNode(v_threshold=(-1.0, 1.0), v_reset=None))
                new_module = nn.Sequential(module, neuron.IFNode(v_reset=None))
                self.module_list._modules[name] = new_module
            if module.__class__.__name__ == "ReLU":
                # converted_model._modules[name] = neuron.NSIFNode(v_threshold=(0,1.0), v_reset=None)
                self.module_list._modules[name] = neuron.IFNode(v_reset=None)
            if module.__class__.__name__ == "MaxPool2d":
                new_module = snn_modules.MaxPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding #,momentum=0.5
                    )
                self.module_list._modules[name] = new_module
            # if module.__class__.__name__ == "MaxPool2d":
            #     new_module = nn.AvgPool2d(
            #         kernel_size=module.kernel_size,
            #             stride=module.stride,
            #             padding=module.padding)
            #     new_module = nn.Sequential(module, neuron.IFNode(v_reset=None))
            #     self.module_list._modules[name] = new_module
        self.is_ann = False
        return self


class ONNX_Converter:
    @staticmethod
    def convert_conv(node, model:ONNXConvertedModel):
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
    def convert_relu(node, model:ONNXConvertedModel):
        relu = nn.ReLU()
        model.module_list.append(relu)
        return len(model.module_list)-1, node.input, node.output

    # @staticmethod
    # def convert_matmul(node, model: ONNXConvertedModel):
    #     class MatMul(nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #         def forward(self,input1,input2):
    #             return input1 @ input2
    #     mul = MatMul()
    #     model.module_list.append(mul)
    #     return len(model.module_list)-1, node.input, node.output


    @staticmethod
    def convert_batchnormalization(node, model:ONNXConvertedModel):
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
    def convert_add(node, model: ONNXConvertedModel):
        class Add(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self,input1,input2):
                return input1 + input2
        add = Add()
        model.module_list.append(add)
        return len(model.module_list)-1, node.input, node.output

    @staticmethod
    def convert_mul(node, model: ONNXConvertedModel):
        class Mul(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self,input1,input2):
                return input1 * input2
        mul = Mul()
        model.module_list.append(mul)
        return len(model.module_list)-1, node.input, node.output

    @staticmethod
    def convert_averagepool(node, model: ONNXConvertedModel):
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
    def convert_globalaveragepool(node, model: ONNXConvertedModel):
        gap = nn.AdaptiveAvgPool2d((1, 1))
        model.module_list.append(gap)
        model.module_list.append(gap)
        return len(model.module_list) - 1, node.input, node.output

    @staticmethod
    def convert_maxpool(node, model: ONNXConvertedModel):
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
    def convert_flatten(node, model: ONNXConvertedModel):
        if len(node.attribute) == 0:
            axis = 1
        else:
            axis = node.attribute[0].i
        # def flatten(x):
        #     o_shape = []
        #     for i in range(axis):
        #         o_shape.append(x.shape[i])
        #     o_shape.append(-1)
        #     return x.view(*o_shape)
        if axis==1:
            flatten = nn.Flatten()
            model.module_list.append(flatten)
            return len(model.module_list)-1, node.input, node.output
        else:
            raise NotImplementedError("Not Implemented yet!")
            #return flatten, node.input, node.output

    @staticmethod
    def convert_gemm(node, model: ONNXConvertedModel):
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
    def convert_pad(node, model: ONNXConvertedModel):
        mode = node.attribute[0].s
        pads = list(node.attribute[1].ints)
        #print(pads,[*pads[2:4],*pads[3:5]])
        value = node.attribute[2].f
        try:
            assert(mode == b'constant')
            assert(sum(pads[:4]) == 0)
        except AssertionError:
            print("Now only support converting to nn.ConstantPad2d") #TODO pad
        # pad = nn.ConstantPad2d(pads[4:], value)
        pad = nn.ConstantPad2d([*pads[2:4],*pads[3:5]],value)
        model.module_list.append(pad)
        return len(model.module_list)-1, node.input, node.output