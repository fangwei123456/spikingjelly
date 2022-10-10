import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
import copy
from typing import Callable
import numpy as np


def hash_str(x: object):
    hash_code = hash(x)
    if hash_code < 0:
        return f'_{-hash_code}'
    else:
        return hash_code


class VarNode:
    def __init__(self, prefix: str, name: str, instance: object, value=None):
        self.debug_name = name  # 原始的name形如 %8, v_last.1
        # 将原始的name进行转换
        self.name = prefix + '_' + name.replace('.', '_')

        self.instance = str(instance)
        # 中间节点的self.instance，在生成前向传播cuda代码时，若debug_instance为Tensor，self.instance会被修改为float
        self.value = value
        self.requires_grad = False
        self.cu_var_suffix = ''


    @property
    def name_bp(self):
        return 'grad_' + self.name


    @property
    def cu_var(self):
        # 前向传播时，在cuda代码中的变量名

        # 如果value非空，表明其是一个常数值，直接返回数值即可，例如 value = 0.1 返回 '0.1f'
        if self.value is not None:
            if self.instance == 'int':
                return self.name
            elif self.instance == 'float':
                return self.name + 'f'
            else:
                raise ValueError(self.instance)

        # value空，表示其是一个变量

        return self.name + self.cu_var_suffix

    @property
    def cu_var_bp(self):

        # 反向传播时在cuda代码中的变量名
        if self.value is not None:
            raise ValueError
        else:
            return 'grad_' + self.cu_var

    def __repr__(self):
        return f'({self.debug_name}, {self.name}, {self.instance}, value={self.value}, rg={self.requires_grad})'


def analyse_graph(custom_fun, requires_grad: tuple):
    graph: torch.Graph = torch.jit.script(custom_fun).graph

    logging.debug(f'\ngraph = {graph}')
    # 生成 输入 中间 输出 节点
    assert sys.version_info.major >= 3 and sys.version_info.minor >= 6
    # python >= 3.6时，字典默认是有序的
    # key是VarNode.debug_name，value是VarNode
    input_nodes = {}
    output_nodes = {}
    inter_nodes = {}

    assert custom_fun.__annotations__.__len__() >= 2
    for i, (item, name) in enumerate(zip(graph.inputs(), custom_fun.__annotations__.keys())):
        # 要求custom_fun一定是custom_fun(x: torch.Tensor, v_last: torch.Tensor, ...)的形式
        if i == 0:
            assert str(item.type()) == 'Tensor' and name == 'x'
        elif i == 1:
            assert str(item.type()) == 'Tensor' and name == 'v_last'

        # 用python函数中的name覆盖掉jit自动生成的name
        # 仅包括输入。中间变量的命名仍然是jit设置的，不会被更改
        item.setDebugName(name)

        node = VarNode(prefix='input', name=item.debugName(), instance=item.type())
        if node.instance == 'Tensor' and requires_grad[i]:
            node.requires_grad = True

        logging.debug(f'\ninput node [{i}] = {node}')
        assert node not in input_nodes
        input_nodes[node.debug_name] = node

    for i, item in enumerate(graph.outputs()):

        if i == 0:
            assert str(item.type()) == 'Tensor'
            item.setDebugName('h')

        elif i > 0:
            raise NotImplementedError('For the moment, we only support for single output!')

        node = VarNode(prefix='output', name=item.debugName(), instance=item.type())

        logging.debug(f'\noutput node [{i}] = {node}')
        assert node not in output_nodes
        output_nodes[node.debug_name] = node

    cmds = []
    # cmds的元素是一个元组，为 (output, fun, inputs)
    # 这里的output是VarNode，fun是str，inputs是(VarNode)
    for node in graph.nodes():
        # item: torch.Note
        fun = node.kind()
        if fun == 'prim::Constant':

            item = node.output()
            assert item.debugName() not in input_nodes and item.debugName() not in output_nodes

            i_node = VarNode(prefix='inter', name=item.debugName(), instance=item.type())
            value = None

            # 从命令中提取出常数值
            if i_node.instance == 'int':
                pattern = re.compile(r'.*prim::Constant\[value=([0-9]+)\]')
                m = pattern.match(str(node))
                value = int(m.groups()[0])


            elif i_node.instance == 'float':
                pattern = re.compile(r'.*prim::Constant\[value=([0-9\.]+)\]')
                m = pattern.match(str(node))
                value = float(m.groups()[0])

            else:
                raise NotImplementedError

            i_node.value = value
            assert i_node.debug_name not in input_nodes
            assert i_node.debug_name not in output_nodes
            if i_node.debug_name not in inter_nodes:
                inter_nodes[i_node.debug_name] = i_node

            cmds.append((i_node, fun, ()))


        else:
            inputs = []

            for item in node.inputs():

                if item.debugName() in input_nodes:
                    i_node = input_nodes[item.debugName()]

                elif item.debugName() in output_nodes:
                    i_node = output_nodes[item.debugName()]

                else:
                    # 只有既不为输入node也不为输出node的node，才会被视作中间node
                    if item.debugName() in inter_nodes:
                        i_node = inter_nodes[item.debugName()]
                    else:
                        i_node = VarNode(prefix='inter', name=item.debugName(), instance=item.type())
                        inter_nodes[i_node.debug_name] = i_node

                inputs.append(i_node)

            item = node.output()
            if item.debugName() in input_nodes:
                i_node = input_nodes[item.debugName()]

            elif item.debugName() in output_nodes:
                i_node = output_nodes[item.debugName()]

            else:
                # 只有既不为输入node也不为输出node的node，才会被视作中间node
                if item.debugName() in inter_nodes:
                    i_node = inter_nodes[item.debugName()]
                else:
                    i_node = VarNode(prefix='inter', name=item.debugName(), instance=item.type())
                    inter_nodes[i_node.debug_name] = i_node

            cmds.append((i_node, fun, tuple(inputs)))

    for i, node in enumerate(inter_nodes.values()):
        logging.debug(f'\ninter node [{i}] = {node}')

    return input_nodes, inter_nodes, output_nodes, cmds


def gen_forward_codes(input_nodes: dict, inter_nodes: dict, output_nodes: dict, cmds: list, hard_reset: bool):
    # 暂时只支持单个输出
    assert output_nodes.__len__() == 1

    # 代码生成
    codes = '\n'
    codes += '                '
    codes += '{\n'

    for node in input_nodes.values():
        # 赋值到代码段的变量
        if node.debug_name == 'x':
            codes += '                  '
            codes += f'const float {node.cu_var} = x_seq[t];\n'
        elif node.debug_name == 'v_last':
            codes += '                  '
            codes += f'const float {node.cu_var} = v_v_seq[t];\n'
        else:
            if node.instance == 'Tensor':
                node.cu_var_suffix = '_t'
                codes += '                  '
                codes += f'const float {node.cu_var} = {node.name}[t];\n'

            # instance为float的不需要提前赋值，因为不需要索引（直接从cuda函数的参数中取出即可）


    # 记录在自动生成的cuda代码段中，哪些cu_var是已经声明的
    code_block_nodes = {}

    cuda_cmds = []
    for item in cmds:
        output, fun, inputs = item
        codes += '                  '
        if fun == 'prim::Constant':
            gen_cmd = '\n'
        elif fun in ['aten::add', 'aten::sub']:
            # z = x + y * alpha
            x, y, alpha = inputs
            z = output
            z.requires_grad = x.requires_grad or y.requires_grad
            if z.cu_var not in code_block_nodes:
                code_block_nodes[z.cu_var] = z
                codes += 'float '

            if fun == 'aten::add':
                op = '+'
            else:
                op = '-'

            if alpha.value == 1:
                gen_cmd = f'{z.cu_var} = {x.cu_var} {op} {y.cu_var};\n'
            else:
                gen_cmd = f'{z.cu_var} = {x.cu_var} {op} {y.cu_var} * {alpha.cu_var};\n'

        elif fun in ['aten::mul', 'aten::div']:
            x, y = inputs
            z = output
            z.requires_grad = x.requires_grad or y.requires_grad
            if z.cu_var not in code_block_nodes:
                code_block_nodes[z.cu_var] = z
                codes += 'float '
            if fun == 'aten::mul':
                op = '*'
            else:
                op = '/'

            gen_cmd = f'{z.cu_var} = {x.cu_var} {op} {y.cu_var};\n'
        else:
            raise NotImplementedError(fun)

        codes += gen_cmd
        cuda_cmds.append(gen_cmd)

    for i, node in enumerate(output_nodes.values()):
        # 代码段的变量赋值到输出
        if i == 0:
            codes += '                  '
            codes += f'h_seq[t] = {node.name};\n'

    codes += '                '
    codes += '}\n'

    # CUDA函数的参数
    params = [
        ('x_seq', 'const float *'),
        ('v_v_seq', 'float *'),
        ('h_seq', 'float *'),
        ('spike_seq', 'float *'),
        ('v_threshold', 'const float &')
    ]
    if hard_reset:
        params.append(('v_reset', 'const float &'))

    params.extend([
        ('neuron_num', 'const int &'),
        ('numel', 'const int &'),
    ])
    params_name = []
    for item in params:
        params_name.append(item[0])

    # 在CUDA函数参数中增加参数，同时检测命名冲突

    for node in inter_nodes.values():
        assert node.name not in params_name

    for node in input_nodes.values():
        if node.debug_name in ['x', 'v_last']:
            pass
        else:
            assert node.name not in params_name

            if node.instance == 'Tensor':
                param = (node.name, 'const float *')
            elif node.instance == 'float':
                param = (node.name, 'const float &')
            else:
                raise NotImplementedError
            params.append(param)

    for node in output_nodes.values():
        assert node.name not in params_name

    for i in range(params.__len__()):
        param = params[i]
        params[i] = param[1] + param[0]

    head = ', '.join(params)
    head = '(' + head + ')'

    head += '''
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {
            const int dt = neuron_num;
            for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
            {
                const int t = index + mem_offset;
    '''
    tail = '''
                if (h_seq[t] >= v_threshold)

                {
                    spike_seq[t] = 1.0f;
                    v_v_seq[t + dt] = v_reset;
                }

                else
                {
                    spike_seq[t] = 0.0f;
                    v_v_seq[t + dt] = h_seq[t];
                }
            }
        }
    }
    '''

    codes = head + codes + tail

    kernel_name = f'forward_kernel_{hash_str(codes)}'
    codes = f'''
    extern "C" __global__
    void {kernel_name}
    ''' + codes

    return codes, kernel_name, cuda_cmds


def gen_backward_codes(cuda_cmds: list, input_nodes: dict, output_nodes: dict, cmds: list, hard_reset: bool,
                       detach_reset: bool, surrogate_fuction):
    '''
    用户定义的前向传播函数为
    h_seq[t] = fun(x_seq[t], v_v_seq[t], ...)

    需要计算出 h_seq[t] -> x_seq[t] 的梯度和 h_seq[t] -> v_v_seq[t]的梯度
    还需要考虑 ... 中如果有tensor，可以增加flag，决定是否计算h_seq[t]对其的梯度
    '''

    input_bp_nodes = {}
    '''
    在反向传播时，输入梯度是output_nodes的梯度
    有些变量的梯度在计算时，需要用到其他变量，例如z = x * y，计算grad_x需要用到y
    input_bp_nodes用来记录哪些node要用到
    '''

    # 记录在自动生成的cuda代码段中，哪些cu_var是已经声明的
    code_block_nodes = {}

    codes = '\n'


    for i in range(cmds.__len__()):
        output, fun, inputs = cmds[cmds.__len__() - 1 - i]
        codes += '\n'
        codes += '                 '
        codes += f'// {cuda_cmds[cmds.__len__() - 1 - i]}'
        if fun == 'prim::Constant':
            codes += '\n'
        elif fun == 'aten::add':
            # z = x + y * alpha
            x, y, alpha = inputs
            z = output
            if alpha.value == 1:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += '                 '
                        codes += f'float {x.cu_var_bp} = {z.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{x.cu_var_bp} += {z.cu_var_bp};\n'
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += '                 '
                        codes += f'float {y.cu_var_bp} = {z.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{y.cu_var_bp} += {z.cu_var_bp};\n'
            else:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += '                 '
                        codes += f'float {x.cu_var_bp} = {z.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{x.cu_var_bp} += {z.cu_var_bp};\n'
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += '                 '
                        codes += f'float {y.cu_var_bp} = {z.cu_var_bp} * {alpha.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{y.cu_var_bp} += {z.cu_var_bp} * {alpha.cu_var_bp};\n'

        elif fun == 'aten::sub':
            # z = x - y * alpha
            x, y, alpha = inputs
            z = output
            if alpha.value == 1:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += '                 '
                        codes += f'float {x.cu_var_bp} = {z.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{x.cu_var_bp} += {z.cu_var_bp};\n'
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += '                 '
                        codes += f'float {y.cu_var_bp} = - {z.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{y.cu_var_bp} += - {z.cu_var_bp};\n'
            else:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += '                 '
                        codes += f'float {x.cu_var_bp} = {z.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{x.cu_var_bp} += {z.cu_var_bp};\n'
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += '                 '
                        codes += f'float {y.cu_var_bp} = - {z.cu_var_bp} * {alpha.cu_var_bp};\n'
                    else:
                        codes += '                 '
                        codes += f'{y.cu_var_bp} += - {z.cu_var_bp} * {alpha.cu_var_bp};\n'


        elif fun == 'aten::mul':
            # z = x * y
            x, y = inputs
            z = output
            if x.requires_grad:
                if x.cu_var_bp not in code_block_nodes:
                    code_block_nodes[x.cu_var_bp] = x
                    codes += '                 '
                    codes += f'float {x.cu_var_bp} = {z.cu_var_bp} * {y.cu_var};\n'
                else:
                    codes += '                 '
                    codes += f'{x.cu_var_bp} += {z.cu_var_bp} * {y.cu_var};\n'
                input_bp_nodes[y.name] = y
            if y.requires_grad:
                if y.cu_var_bp not in code_block_nodes:
                    code_block_nodes[y.cu_var_bp] = y
                    codes += '                 '
                    codes += f'float {y.cu_var_bp} = {z.cu_var_bp} * {x.cu_var};\n'
                else:
                    codes += '                 '
                    codes += f'{y.cu_var_bp} += {z.cu_var_bp} * {x.cu_var};\n'
                input_bp_nodes[x.name] = x

        elif fun == 'aten::div':
            # z = x / y
            x, y = inputs
            z = output
            if x.requires_grad:
                if x.cu_var_bp not in code_block_nodes:
                    code_block_nodes[x.cu_var_bp] = x
                    codes += '                 '
                    codes += f'float {x.cu_var_bp} = {z.cu_var_bp} / {y.cu_var};\n'
                else:
                    codes += '                 '
                    codes += f'{x.cu_var_bp} += {z.cu_var_bp} / {y.cu_var};\n'
                input_bp_nodes[y.name] = y
            if y.requires_grad:
                if y.cu_var_bp not in code_block_nodes:
                    code_block_nodes[y.cu_var_bp] = y
                    codes += '                 '
                    codes += f'float {y.cu_var_bp} = - {z.cu_var_bp} * {x.cu_var} / ({y.cu_var} * {y.cu_var});\n'
                else:
                    codes += '                 '
                    codes += f'{y.cu_var_bp} += - {z.cu_var_bp} * {x.cu_var} / ({y.cu_var} * {y.cu_var});\n'
                input_bp_nodes[x.name] = x
                input_bp_nodes[y.name] = y

    for i, node in enumerate(input_bp_nodes):
        logging.debug(f'\ninput bp node [{i}] = {node}')

    # CUDA函数的参数
    cuda_params = {
        'grad_spike_seq': 'const float *',
        'grad_v_seq': 'const float *',
        'h_seq': 'const float *',
        'spike_seq': 'const float *',
        'grad_x_seq': 'float *',
        'grad_v_init': 'float *',
        'v_threshold': 'const float &',
    }

    if hard_reset:
        cuda_params['v_reset'] = 'const float &'

    cuda_params['neuron_num'] = 'const int &'
    cuda_params['numel'] = 'const int &'


    # 在CUDA函数参数中增加参数，同时检测命名冲突

    # 这里增加的是用户自定义的除了x和v_last外，其他需要梯度的python函数的参数
    for i, node in enumerate(input_nodes.values()):
        if i >= 2:
            if node.name_bp not in cuda_params:
                if node.requires_grad:
                    cuda_params[node.name_bp] = 'const float *'





    # 这里增加的是反向传播所需要的参数
    for node in input_bp_nodes.values():
        if node.name not in cuda_params:
            assert node.debug_name in input_nodes or node.debug_name in output_nodes

            if node.instance == 'Tensor':
                cuda_params[node.name] = 'const float *'
            elif node.instance == 'float':
                cuda_params[node.name] = 'const float &'
            else:
                raise NotImplementedError(node)

    params = []
    for cuda_param, cuda_param_instance in cuda_params.items():
        params.append(cuda_param_instance + cuda_param)

    head = ', '.join(params)
    head = '(' + head + ')'

    head += '''
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {   
            float grad_output_h = 0.0f;  // grad_output_h will be used recursively
            for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
            {
                const int t = index + mem_offset;
                const float over_th = h_seq[t] - v_threshold;
    '''
    head += surrogate_fuction.cuda_code(x='over_th', y='grad_s_to_h', dtype='fp32')

    head += '        '
    if detach_reset:
        if hard_reset:
            head += 'const float grad_v_to_h = 1.0f - spike_seq[t];\n'
        else:
            head += 'const float grad_v_to_h = 1.0f;\n'
    else:
        if hard_reset:
            head += 'const float grad_v_to_h = 1.0f - spike_seq[t] + (-h_seq[t] + v_reset) * grad_s_to_h;\n'
        else:
            head += 'const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;\n'





    tail = ''
    # grad_input_x, grad_input_v_last是自动生成的代码计算出来的
    tail += '              '
    tail += 'grad_output_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_input_v_last) * grad_v_to_h;\n'

    for i, node in enumerate(input_nodes.values()):
        if i >= 2:
            if node.requires_grad:
                tail += '              '
                tail += f'{node.name_bp}[t] = {node.cu_var_bp};\n'








    tail += '''
            }
    '''



    tail += codes
    # += codes 是为了计算grad_v_init[index]
    tail += '''
            grad_v_init[index] = grad_input_v_last;
        }
    }
    '''
    codes = head + codes + tail
    kernel_name = f'backward_kernel_{hash_str(codes)}'
    codes = f'''
    extern "C" __global__
    void {kernel_name}
    ''' + codes

    input_bp_vars = []
    # input_bp_vars记录了python函数中的哪些输入变量，是计算反向传播所需的
    for node in input_bp_nodes.values():
        input_bp_vars.append(node.debug_name)
    return codes, kernel_name, input_bp_vars
