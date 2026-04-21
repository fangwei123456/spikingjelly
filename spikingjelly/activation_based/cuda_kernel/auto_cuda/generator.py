import logging
import operator as _op
import typing
import torch
import torch.fx as _fx
import sys


def hash_str(x: object):
    hash_code = hash(x)
    if hash_code < 0:
        return f"_{-hash_code}"
    else:
        return hash_code


class VarNode:
    def __init__(self, prefix: str, name: str, instance: object, value=None):
        self.debug_name = name  # 原始的name形如 %8, v_last.1
        # 将原始的name进行转换
        self.name = prefix + "_" + name.replace(".", "_")

        self.instance = str(instance)
        # 中间节点的self.instance，在生成前向传播cuda代码时，若debug_instance为Tensor，self.instance会被修改为float
        self.value = value
        self.requires_grad = False
        self.cu_var_suffix = ""

    @property
    def name_bp(self):
        return "grad_" + self.name

    @property
    def cu_var(self):
        # 前向传播时，在cuda代码中的变量名

        # 如果value非空，表明其是一个常数值，直接返回数值即可，例如 value = 0.1 返回 '0.1f'
        if self.value is not None:
            if self.instance == "int":
                return str(int(self.value))
            elif self.instance == "float":
                return f"{float(self.value)}f"
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
            return "grad_" + self.cu_var

    def __repr__(self):
        return f"({self.debug_name}, {self.name}, {self.instance}, value={self.value}, rg={self.requires_grad})"


def analyse_graph(custom_fun, requires_grad: tuple):
    # Map from torch.fx operator targets to aten-style kind strings
    _FX_TO_KIND = {
        _op.add: "aten::add",
        _op.sub: "aten::sub",
        _op.mul: "aten::mul",
        _op.truediv: "aten::div",
    }

    # typing.get_type_hints() resolves string annotations produced by
    # 'from __future__ import annotations' (PEP 563), unlike __annotations__.
    try:
        annotations = typing.get_type_hints(custom_fun)
    except (NameError, AttributeError):
        annotations = custom_fun.__annotations__
    assert len(annotations) >= 2  # at least x and v_last (plus optional 'return')

    gm = _fx.symbolic_trace(custom_fun)
    graph = gm.graph

    logging.debug(f"\ngraph = {graph}")
    assert sys.version_info.major >= 3 and sys.version_info.minor >= 6

    input_nodes = {}
    output_nodes = {}
    inter_nodes = {}
    _fx_to_var = {}
    _const_id = [0]

    def _make_const(value):
        _const_id[0] += 1
        cname = f"const_{_const_id[0]}"
        instance = "int" if isinstance(value, int) else "float"
        var = VarNode(prefix="inter", name=cname, instance=instance, value=value)
        inter_nodes[cname] = var
        return var

    def _resolve(arg):
        if isinstance(arg, _fx.Node):
            return _fx_to_var[arg]
        if isinstance(arg, (int, float)):
            return _make_const(arg)
        raise NotImplementedError(f"unsupported arg type: {type(arg)}")

    cmds = []
    ph_idx = 0

    for fx_node in graph.nodes:
        if fx_node.op == "placeholder":
            name = fx_node.name
            ann = annotations.get(name)
            if ann is None:
                raise TypeError(
                    f"Parameter '{name}' of custom_fun must have a type annotation "
                    "(torch.Tensor, float, or int)."
                )
            if ann is torch.Tensor:
                inst = "Tensor"
            elif ann is float:
                inst = "float"
            elif ann is int:
                inst = "int"
            else:
                inst = str(ann)

            if ph_idx == 0:
                if inst != "Tensor" or name != "x":
                    raise ValueError(
                        f"First parameter of custom_fun must be 'x: torch.Tensor', "
                        f"got '{name}: {ann}'"
                    )
            elif ph_idx == 1:
                if inst != "Tensor" or name != "v_last":
                    raise ValueError(
                        f"Second parameter of custom_fun must be 'v_last: torch.Tensor', "
                        f"got '{name}: {ann}'"
                    )

            var = VarNode(prefix="input", name=name, instance=inst)
            if inst == "Tensor" and ph_idx < len(requires_grad) and requires_grad[ph_idx]:
                var.requires_grad = True

            logging.debug(f"\ninput node [{ph_idx}] = {var}")
            input_nodes[name] = var
            _fx_to_var[fx_node] = var
            ph_idx += 1

        elif fx_node.op == "call_function":
            target = fx_node.target
            if target not in _FX_TO_KIND:
                raise NotImplementedError(f"unsupported operation: {target}")
            kind = _FX_TO_KIND[target]

            var = VarNode(prefix="inter", name=fx_node.name, instance="Tensor")
            inter_nodes[fx_node.name] = var
            _fx_to_var[fx_node] = var

            in_vars = tuple(_resolve(a) for a in fx_node.args)
            # aten::add/sub require a trailing alpha=1 constant for compatibility with
            # gen_forward_codes / gen_backward_codes which expect (x, y, alpha) inputs
            if kind in ("aten::add", "aten::sub") and len(in_vars) == 2:
                in_vars = (*in_vars, _make_const(1))

            cmds.append((var, kind, in_vars))

        elif fx_node.op in ("call_method", "call_module", "get_attr"):
            raise NotImplementedError(
                f"fx node op '{fx_node.op}' (target={fx_node.target!r}) is not supported. "
                "custom_fun must use only Python arithmetic operators (+, -, *, /) on its arguments."
            )

        elif fx_node.op == "output":
            ret = fx_node.args[0]
            if not isinstance(ret, _fx.Node):
                raise NotImplementedError(f"unsupported output type: {type(ret)}")

            h_var = VarNode(prefix="output", name="h", instance="Tensor")
            output_nodes["h"] = h_var
            logging.debug(f"\noutput node [0] = {h_var}")

            src = _fx_to_var[ret]
            # Replace src with h_var everywhere in cmds — both as a cmd's
            # output and inside any cmd's in_vars.  Updating only the output
            # position would leave stale references to src in in_vars of cmds
            # that happen to consume the output node's value (e.g. dead-code
            # operations that follow the return expression in the source).
            # Those stale references would produce undeclared CUDA identifiers.
            for i, (out, fun, ins) in enumerate(cmds):
                new_out = h_var if out is src else out
                new_ins = tuple(h_var if v is src else v for v in ins)
                if new_out is not out or new_ins != ins:
                    cmds[i] = (new_out, fun, new_ins)
            if src.debug_name in inter_nodes:
                del inter_nodes[src.debug_name]
            _fx_to_var[ret] = h_var

            # If no cmd was renamed to produce h_var (e.g. the function directly
            # returns an input placeholder without any computation), insert a
            # synthetic identity operation so gen_forward_codes emits a valid
            # 'float output_h = <input>;' declaration instead of referencing an
            # undeclared variable.
            if not any(out is h_var for out, _, _ in cmds):
                cmds.append((h_var, "aten::add", (src, _make_const(0.0), _make_const(1))))

    for i, node in enumerate(inter_nodes.values()):
        logging.debug(f"\ninter node [{i}] = {node}")

    return input_nodes, inter_nodes, output_nodes, cmds


def gen_forward_codes(
    input_nodes: dict,
    inter_nodes: dict,
    output_nodes: dict,
    cmds: list,
    hard_reset: bool,
):
    # 暂时只支持单个输出
    assert output_nodes.__len__() == 1

    # 代码生成
    codes = "\n"
    codes += "                "
    codes += "{\n"

    for node in input_nodes.values():
        # 赋值到代码段的变量
        if node.debug_name == "x":
            codes += "                  "
            codes += f"const float {node.cu_var} = x_seq[t];\n"
        elif node.debug_name == "v_last":
            codes += "                  "
            codes += f"const float {node.cu_var} = v_v_seq[t];\n"
        else:
            if node.instance == "Tensor":
                node.cu_var_suffix = "_t"
                codes += "                  "
                codes += f"const float {node.cu_var} = {node.name}[t];\n"

            # instance为float的不需要提前赋值，因为不需要索引（直接从cuda函数的参数中取出即可）

    # 记录在自动生成的cuda代码段中，哪些cu_var是已经声明的
    code_block_nodes = {}

    cuda_cmds = []
    for item in cmds:
        output, fun, inputs = item
        codes += "                  "
        if fun in ["aten::add", "aten::sub"]:
            # z = x + y * alpha
            x, y, alpha = inputs
            z = output
            z.requires_grad = x.requires_grad or y.requires_grad
            if z.cu_var not in code_block_nodes:
                code_block_nodes[z.cu_var] = z
                codes += "float "

            if fun == "aten::add":
                op = "+"
            else:
                op = "-"

            if alpha.value == 1:
                gen_cmd = f"{z.cu_var} = {x.cu_var} {op} {y.cu_var};\n"
            else:
                gen_cmd = f"{z.cu_var} = {x.cu_var} {op} {y.cu_var} * {alpha.cu_var};\n"

        elif fun in ["aten::mul", "aten::div"]:
            x, y = inputs
            z = output
            z.requires_grad = x.requires_grad or y.requires_grad
            if z.cu_var not in code_block_nodes:
                code_block_nodes[z.cu_var] = z
                codes += "float "
            if fun == "aten::mul":
                op = "*"
            else:
                op = "/"

            gen_cmd = f"{z.cu_var} = {x.cu_var} {op} {y.cu_var};\n"
        else:
            raise NotImplementedError(fun)

        codes += gen_cmd
        cuda_cmds.append(gen_cmd)

    for i, node in enumerate(output_nodes.values()):
        # 代码段的变量赋值到输出
        if i == 0:
            codes += "                  "
            codes += f"h_seq[t] = {node.name};\n"

    codes += "                "
    codes += "}\n"

    # CUDA函数的参数
    params = [
        ("x_seq", "const float *"),
        ("v_v_seq", "float *"),
        ("h_seq", "float *"),
        ("spike_seq", "float *"),
        ("v_threshold", "const float &"),
    ]
    if hard_reset:
        params.append(("v_reset", "const float &"))

    params.extend(
        [
            ("neuron_num", "const int &"),
            ("numel", "const int &"),
        ]
    )
    params_name = []
    for item in params:
        params_name.append(item[0])

    # 在CUDA函数参数中增加参数，同时检测命名冲突

    for node in inter_nodes.values():
        assert node.name not in params_name

    for node in input_nodes.values():
        if node.debug_name in ["x", "v_last"]:
            pass
        else:
            assert node.name not in params_name

            if node.instance == "Tensor":
                param = (node.name, "const float *")
            elif node.instance == "float":
                param = (node.name, "const float &")
            elif node.instance == "int":
                param = (node.name, "const int &")
            else:
                raise NotImplementedError
            params.append(param)

    for node in output_nodes.values():
        assert node.name not in params_name

    for i in range(params.__len__()):
        param = params[i]
        params[i] = param[1] + param[0]

    head = ", ".join(params)
    head = "(" + head + ")"

    head += """
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {
            const int dt = neuron_num;
            for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
            {
                const int t = index + mem_offset;
    """
    tail = """
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
    """

    codes = head + codes + tail

    kernel_name = f"forward_kernel_{hash_str(codes)}"
    codes = (
        f"""
    extern "C" __global__
    void {kernel_name}
    """
        + codes
    )

    return codes, kernel_name, cuda_cmds


def gen_backward_codes(
    cuda_cmds: list,
    input_nodes: dict,
    output_nodes: dict,
    cmds: list,
    hard_reset: bool,
    detach_reset: bool,
    surrogate_fuction,
):
    """
    用户定义的前向传播函数为
    h_seq[t] = fun(x_seq[t], v_v_seq[t], ...)

    需要计算出 h_seq[t] -> x_seq[t] 的梯度和 h_seq[t] -> v_v_seq[t]的梯度
    还需要考虑 ... 中如果有tensor，可以增加flag，决定是否计算h_seq[t]对其的梯度
    """

    input_bp_nodes = {}
    """
    在反向传播时，输入梯度是output_nodes的梯度
    有些变量的梯度在计算时，需要用到其他变量，例如z = x * y，计算grad_x需要用到y
    input_bp_nodes用来记录哪些node要用到
    """

    # 记录在自动生成的cuda代码段中，哪些cu_var是已经声明的
    code_block_nodes = {}

    codes = "\n"

    for i in range(cmds.__len__()):
        output, fun, inputs = cmds[cmds.__len__() - 1 - i]
        codes += "\n"
        codes += "                 "
        codes += f"// {cuda_cmds[cmds.__len__() - 1 - i]}"
        if fun == "aten::add":
            # z = x + y * alpha
            x, y, alpha = inputs
            z = output
            if alpha.value == 1:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += "                 "
                        codes += f"float {x.cu_var_bp} = {z.cu_var_bp};\n"
                    else:
                        codes += "                 "
                        codes += f"{x.cu_var_bp} += {z.cu_var_bp};\n"
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += "                 "
                        codes += f"float {y.cu_var_bp} = {z.cu_var_bp};\n"
                    else:
                        codes += "                 "
                        codes += f"{y.cu_var_bp} += {z.cu_var_bp};\n"
            else:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += "                 "
                        codes += f"float {x.cu_var_bp} = {z.cu_var_bp};\n"
                    else:
                        codes += "                 "
                        codes += f"{x.cu_var_bp} += {z.cu_var_bp};\n"
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += "                 "
                        codes += f"float {y.cu_var_bp} = {z.cu_var_bp} * {alpha.cu_var};\n"
                    else:
                        codes += "                 "
                        codes += (
                            f"{y.cu_var_bp} += {z.cu_var_bp} * {alpha.cu_var};\n"
                        )

        elif fun == "aten::sub":
            # z = x - y * alpha
            x, y, alpha = inputs
            z = output
            if alpha.value == 1:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += "                 "
                        codes += f"float {x.cu_var_bp} = {z.cu_var_bp};\n"
                    else:
                        codes += "                 "
                        codes += f"{x.cu_var_bp} += {z.cu_var_bp};\n"
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += "                 "
                        codes += f"float {y.cu_var_bp} = - {z.cu_var_bp};\n"
                    else:
                        codes += "                 "
                        codes += f"{y.cu_var_bp} += - {z.cu_var_bp};\n"
            else:
                if x.requires_grad:
                    if x.cu_var_bp not in code_block_nodes:
                        code_block_nodes[x.cu_var_bp] = x
                        codes += "                 "
                        codes += f"float {x.cu_var_bp} = {z.cu_var_bp};\n"
                    else:
                        codes += "                 "
                        codes += f"{x.cu_var_bp} += {z.cu_var_bp};\n"
                if y.requires_grad:
                    if y.cu_var_bp not in code_block_nodes:
                        code_block_nodes[y.cu_var_bp] = y
                        codes += "                 "
                        codes += f"float {y.cu_var_bp} = - {z.cu_var_bp} * {alpha.cu_var};\n"
                    else:
                        codes += "                 "
                        codes += (
                            f"{y.cu_var_bp} += - {z.cu_var_bp} * {alpha.cu_var};\n"
                        )

        elif fun == "aten::mul":
            # z = x * y
            x, y = inputs
            z = output
            if x.requires_grad:
                if x.cu_var_bp not in code_block_nodes:
                    code_block_nodes[x.cu_var_bp] = x
                    codes += "                 "
                    codes += f"float {x.cu_var_bp} = {z.cu_var_bp} * {y.cu_var};\n"
                else:
                    codes += "                 "
                    codes += f"{x.cu_var_bp} += {z.cu_var_bp} * {y.cu_var};\n"
                if y.value is None:  # constants are inlined via cu_var; no CUDA param needed
                    input_bp_nodes[y.name] = y
            if y.requires_grad:
                if y.cu_var_bp not in code_block_nodes:
                    code_block_nodes[y.cu_var_bp] = y
                    codes += "                 "
                    codes += f"float {y.cu_var_bp} = {z.cu_var_bp} * {x.cu_var};\n"
                else:
                    codes += "                 "
                    codes += f"{y.cu_var_bp} += {z.cu_var_bp} * {x.cu_var};\n"
                if x.value is None:
                    input_bp_nodes[x.name] = x

        elif fun == "aten::div":
            # z = x / y
            x, y = inputs
            z = output
            if x.requires_grad:
                if x.cu_var_bp not in code_block_nodes:
                    code_block_nodes[x.cu_var_bp] = x
                    codes += "                 "
                    codes += f"float {x.cu_var_bp} = {z.cu_var_bp} / {y.cu_var};\n"
                else:
                    codes += "                 "
                    codes += f"{x.cu_var_bp} += {z.cu_var_bp} / {y.cu_var};\n"
                if y.value is None:
                    input_bp_nodes[y.name] = y
            if y.requires_grad:
                if y.cu_var_bp not in code_block_nodes:
                    code_block_nodes[y.cu_var_bp] = y
                    codes += "                 "
                    codes += f"float {y.cu_var_bp} = - {z.cu_var_bp} * {x.cu_var} / ({y.cu_var} * {y.cu_var});\n"
                else:
                    codes += "                 "
                    codes += f"{y.cu_var_bp} += - {z.cu_var_bp} * {x.cu_var} / ({y.cu_var} * {y.cu_var});\n"
                if x.value is None:
                    input_bp_nodes[x.name] = x
                if y.value is None:
                    input_bp_nodes[y.name] = y

    for i, node in enumerate(input_bp_nodes):
        logging.debug(f"\ninput bp node [{i}] = {node}")

    # CUDA函数的参数
    cuda_params = {
        "grad_spike_seq": "const float *",
        "grad_v_seq": "const float *",
        "h_seq": "const float *",
        "spike_seq": "const float *",
        "grad_x_seq": "float *",
        "grad_v_init": "float *",
        "v_threshold": "const float &",
    }

    if hard_reset:
        cuda_params["v_reset"] = "const float &"

    cuda_params["neuron_num"] = "const int &"
    cuda_params["numel"] = "const int &"

    # 在CUDA函数参数中增加参数，同时检测命名冲突

    # 这里增加的是用户自定义的除了x和v_last外，其他需要梯度的python函数的参数
    for i, node in enumerate(input_nodes.values()):
        if i >= 2:
            if node.name_bp not in cuda_params:
                if node.requires_grad:
                    cuda_params[node.name_bp] = "const float *"

    # 这里增加的是反向传播所需要的参数
    for node in input_bp_nodes.values():
        if node.name not in cuda_params:
            assert node.debug_name in input_nodes or node.debug_name in output_nodes

            if node.instance == "Tensor":
                cuda_params[node.name] = "const float *"
            elif node.instance == "float":
                cuda_params[node.name] = "const float &"
            elif node.instance == "int":
                cuda_params[node.name] = "const int &"
            else:
                raise NotImplementedError(node)

    params = []
    for cuda_param, cuda_param_instance in cuda_params.items():
        params.append(cuda_param_instance + cuda_param)

    head = ", ".join(params)
    head = "(" + head + ")"

    head += """
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < neuron_num)
        {   
            float grad_output_h = 0.0f;  // grad_output_h will be used recursively
            for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
            {
                const int t = index + mem_offset;
                const float over_th = h_seq[t] - v_threshold;
    """
    head += surrogate_fuction.cuda_code(x="over_th", y="grad_s_to_h", dtype="fp32")

    head += "        "
    if detach_reset:
        if hard_reset:
            head += "const float grad_v_to_h = 1.0f - spike_seq[t];\n"
        else:
            head += "const float grad_v_to_h = 1.0f;\n"
    else:
        if hard_reset:
            head += "const float grad_v_to_h = 1.0f - spike_seq[t] + (-h_seq[t] + v_reset) * grad_s_to_h;\n"
        else:
            head += "const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;\n"

    tail = ""
    # grad_input_x, grad_input_v_last是自动生成的代码计算出来的
    tail += "              "
    tail += "grad_output_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_input_v_last) * grad_v_to_h;\n"

    for i, node in enumerate(input_nodes.values()):
        if i >= 2:
            if node.requires_grad:
                tail += "              "
                tail += f"{node.name_bp}[t] = {node.cu_var_bp};\n"

    tail += """
            }
    """

    tail += codes
    # += codes 是为了计算grad_v_init[index]
    tail += """
            grad_v_init[index] = grad_input_v_last;
        }
    }
    """
    codes = head + codes + tail
    kernel_name = f"backward_kernel_{hash_str(codes)}"
    codes = (
        f"""
    extern "C" __global__
    void {kernel_name}
    """
        + codes
    )

    input_bp_vars = []
    # input_bp_vars记录了python函数中的哪些输入变量，是计算反向传播所需的
    for node in input_bp_nodes.values():
        input_bp_vars.append(node.debug_name)
    return codes, kernel_name, input_bp_vars
