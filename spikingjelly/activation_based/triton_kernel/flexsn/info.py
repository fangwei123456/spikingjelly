from collections import namedtuple

import torch.fx as fx

__all__ = ["FlexSNInfo", "extract_info"]


FlexSNInfo = namedtuple(
    typename="FlexSNInfo",
    field_names=[
        "num_inputs",
        "num_outputs",
        "num_states",
        "fwd_core_args",
        "fwd_core_returns",
        "fwd_core_recipients",
        "fwd_kernel_returns",
        "num_fwd_kernel_returns",
        "c2k_return_mapping",
    ],
)


def extract_info(
    fwd_graph: fx.Graph,
    num_inputs: int = 1,
    num_states: int = 0,
    num_outputs: int = 1,
) -> FlexSNInfo:
    r"""
    **API Language:**
    :ref:`中文 <extract_info-cn>` | :ref:`English <extract_info-en>`

    ----

    .. _extract_info-cn:

    * **中文**

    从前向计算图中提取信息。前向图应具有以下签名：
    ``[*inputs, *states] -> [*outputs, *states, *intermediates]``

    提取的信息包括：

    * fwd_core_args: 核心参数
    * fwd_core_returns: 前向图的返回值名称
    * fwd_core_recipients: 接收核心返回值的变量名
    * fwd_kernel_returns: 前向 kernel 的返回值名称（无重复）
    * num_fwd_kernel_returns: fwd_kernel_returns 的长度
    * c2k_return_mapping: 中间结果与 kernel 返回值之间的映射

    :param fwd_graph: 前向计算图
    :type fwd_graph: fx.Graph
    :param num_inputs: 输入数量，默认为 1
    :type num_inputs: int
    :param num_states: 状态数量，默认为 0
    :type num_states: int
    :param num_outputs: 输出数量，默认为 1
    :type num_outputs: int
    :return: 提取的 FlexSN 元信息
    :rtype: FlexSNInfo

    ----

    .. _extract_info-en:

    * **English**

    Extract useful information from the forward graph. The forward graph
    should have the following signature:
    ``[*inputs, *states] -> [*outputs, *states, *intermediates]``

    The extracted information includes:

    * fwd_core_args: the core input argument names
    * fwd_core_returns: the return value names of the forward graph
    * fwd_core_recipients: the variable names receiving the core return values
    * fwd_kernel_returns: the forward kernel return value names (no duplicates)
    * num_fwd_kernel_returns: the length of fwd_kernel_returns
    * c2k_return_mapping: mapping from intermediate results to kernel returns

    :param fwd_graph: The forward computational graph
    :type fwd_graph: fx.Graph
    :param num_inputs: Number of inputs. Default: 1
    :type num_inputs: int
    :param num_states: Number of states. Default: 0
    :type num_states: int
    :param num_outputs: Number of outputs. Default: 1
    :type num_outputs: int
    :return: The extracted FlexSN metadata
    :rtype: FlexSNInfo
    """
    fwd_core_args = [n.name for n in fwd_graph.find_nodes(op="placeholder")]
    fwd_core_returns = []
    for n in fwd_graph.find_nodes(op="output"):
        for a in n.args[0]:
            fwd_core_returns.append(a.name)

    num_args = num_inputs + num_states
    num_outputs_states = num_outputs + num_states
    assert len(fwd_core_args) == num_args
    assert len(fwd_core_returns) >= num_outputs_states

    symbols = {}  # varname in core -> varname in kernel
    fwd_kernel_returns = []
    fwd_core_recipients = []
    for i, s in enumerate(fwd_core_returns[:num_outputs]):  # 1. outputs
        symbols[s] = f"s{i}"
        fwd_core_recipients.append(f"s{i}")
        fwd_kernel_returns.append(f"s{i}")

    for i, v in enumerate(
        fwd_core_returns[num_outputs:num_outputs_states]
    ):  # 2. states
        symbols[v] = f"v{i}"
        fwd_core_recipients.append(f"v{i}")
        fwd_kernel_returns.append(f"v{i}")  # states are also returned by kernel

    n = 0
    c2k_return_mapping = []
    for ret in fwd_core_returns[num_outputs_states:]:  # 3. intermediates
        if ret in symbols:  # duplicated core return detected
            fwd_core_recipients.append("_")  # omit the return value
            # if ret is in symbols, symbols[ret] must be in fwd_kernel_returns
        else:  # not duplicated
            symbols[ret] = f"res{n}_f"
            fwd_core_recipients.append(f"res{n}_f")
            fwd_kernel_returns.append(f"res{n}_f")
            n += 1
        idx = fwd_kernel_returns.index(symbols[ret])  # locate the symbol
        c2k_return_mapping.append(idx)

    return FlexSNInfo(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_states=num_states,
        fwd_core_args=fwd_core_args,
        fwd_core_returns=fwd_core_returns,
        fwd_core_recipients=fwd_core_recipients,
        fwd_kernel_returns=fwd_kernel_returns,
        num_fwd_kernel_returns=len(fwd_kernel_returns),
        c2k_return_mapping=c2k_return_mapping,
    )
