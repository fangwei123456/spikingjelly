from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .. import surrogate, base


__all__ = ["ILCBaseNode", "ILCIFNode", "ILCLIFNode", "ILCCUBALIFNode"]


class ILCBaseNode(nn.Module, base.MultiStepModule):
    r"""
    **API Language:**
    :ref:`中文 <ILCBaseNode-cn>` | :ref:`English <ILCBaseNode-en>`

    ----

    .. _ILCBaseNode-cn:

    * **中文**

    TODO: add Chinese description for ILCBaseNode

    :return: None
    :rtype: None

    ----

    .. _ILCBaseNode-en:

    * **English**

    TODO: add English description for ILCBaseNode

    :return: None
    :rtype: None
    """

    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.act_dim = act_dim
        self.out_pop_dim = act_dim * dec_pop_dim
        self.dec_pop_dim = dec_pop_dim

        self.conn = nn.Conv1d(act_dim, self.out_pop_dim, dec_pop_dim, groups=act_dim)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <neuronal_charge-cn>` | :ref:`English <neuronal_charge-en>`

        ----

        .. _neuronal_charge-cn:

        * **中文**

        TODO: add Chinese description for neuronal_charge

        :return: None
        :rtype: None

        ----

        .. _neuronal_charge-en:

        * **English**

        TODO: add English description for neuronal_charge

        :return: None
        :rtype: None
        """

        raise NotImplementedError

    def neuronal_fire(self):
        r"""
        **API Language:**
        :ref:`中文 <neuronal_fire-cn>` | :ref:`English <neuronal_fire-en>`

        ----

        .. _neuronal_fire-cn:

        * **中文**

        TODO: add Chinese description for neuronal_fire

        :return: None
        :rtype: None

        ----

        .. _neuronal_fire-en:

        * **English**

        TODO: add English description for neuronal_fire

        :return: None
        :rtype: None
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        r"""
        **API Language:**
        :ref:`中文 <neuronal_reset-cn>` | :ref:`English <neuronal_reset-en>`

        ----

        .. _neuronal_reset-cn:

        * **中文**

        TODO: add Chinese description for neuronal_reset

        :return: None
        :rtype: None

        ----

        .. _neuronal_reset-en:

        * **English**

        TODO: add English description for neuronal_reset

        :return: None
        :rtype: None
        """

        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1.0 - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <init_tensor-cn>` | :ref:`English <init_tensor-en>`

        ----

        .. _init_tensor-cn:

        * **中文**

        TODO: add Chinese description for init_tensor

        :return: None
        :rtype: None

        ----

        .. _init_tensor-en:

        * **English**

        TODO: add English description for init_tensor

        :return: None
        :rtype: None
        """

        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <forward-cn>` | :ref:`English <forward-en>`

        ----

        .. _forward-cn:

        * **中文**

        TODO: add Chinese description for forward

        :return: None
        :rtype: None

        ----

        .. _forward-en:

        * **English**

        TODO: add English description for forward

        :return: None
        :rtype: None
        """

        self.init_tensor(x_seq[0].data)

        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            spike_seq.append(spike)
            if t < T - 1:
                x_seq[t + 1] = x_seq[t + 1] + self.conn(
                    spike.view(-1, self.act_dim, self.dec_pop_dim)
                ).view(-1, self.out_pop_dim)

        return torch.stack(spike_seq)


class ILCIFNode(ILCBaseNode):
    r"""
    **API Language:**
    :ref:`中文 <ILCIFNode-cn>` | :ref:`English <ILCIFNode-en>`

    ----

    .. _ILCIFNode-cn:

    * **中文**

    TODO: add Chinese description for ILCIFNode

    :return: None
    :rtype: None

    ----

    .. _ILCIFNode-en:

    * **English**

    TODO: add English description for ILCIFNode

    :return: None
    :rtype: None
    """

    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <neuronal_charge-cn>` | :ref:`English <neuronal_charge-en>`

        ----

        .. _neuronal_charge-cn:

        * **中文**

        TODO: add Chinese description for neuronal_charge

        :return: None
        :rtype: None

        ----

        .. _neuronal_charge-en:

        * **English**

        TODO: add English description for neuronal_charge

        :return: None
        :rtype: None
        """

        self.v = self.v + x


class ILCLIFNode(ILCBaseNode):
    r"""
    **API Language:**
    :ref:`中文 <ILCLIFNode-cn>` | :ref:`English <ILCLIFNode-en>`

    ----

    .. _ILCLIFNode-cn:

    * **中文**

    TODO: add Chinese description for ILCLIFNode

    :return: None
    :rtype: None

    ----

    .. _ILCLIFNode-en:

    * **English**

    TODO: add English description for ILCLIFNode

    :return: None
    :rtype: None
    """

    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_decay: float = 0.75,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <neuronal_charge-cn>` | :ref:`English <neuronal_charge-en>`

        ----

        .. _neuronal_charge-cn:

        * **中文**

        TODO: add Chinese description for neuronal_charge

        :return: None
        :rtype: None

        ----

        .. _neuronal_charge-en:

        * **English**

        TODO: add English description for neuronal_charge

        :return: None
        :rtype: None
        """

        self.v = self.v * self.v_decay + x


class ILCCUBALIFNode(ILCBaseNode):
    r"""
    **API Language:**
    :ref:`中文 <ILCCUBALIFNode-cn>` | :ref:`English <ILCCUBALIFNode-en>`

    ----

    .. _ILCCUBALIFNode-cn:

    * **中文**

    TODO: add Chinese description for ILCCUBALIFNode

    :return: None
    :rtype: None

    ----

    .. _ILCCUBALIFNode-en:

    * **English**

    TODO: add English description for ILCCUBALIFNode

    :return: None
    :rtype: None
    """

    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <neuronal_charge-cn>` | :ref:`English <neuronal_charge-en>`

        ----

        .. _neuronal_charge-cn:

        * **中文**

        TODO: add Chinese description for neuronal_charge

        :return: None
        :rtype: None

        ----

        .. _neuronal_charge-en:

        * **English**

        TODO: add English description for neuronal_charge

        :return: None
        :rtype: None
        """

        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <init_tensor-cn>` | :ref:`English <init_tensor-en>`

        ----

        .. _init_tensor-cn:

        * **中文**

        TODO: add Chinese description for init_tensor

        :return: None
        :rtype: None

        ----

        .. _init_tensor-en:

        * **English**

        TODO: add English description for init_tensor

        :return: None
        :rtype: None
        """

        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)
