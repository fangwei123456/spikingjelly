from spikingjelly.clock_driven.ann2snn.modules import *
from tqdm import tqdm
from spikingjelly.clock_driven import neuron
import copy


class Converter(nn.Module):

    def __init__(self, dataloader, mode='Max'):
        """
        * :ref:`API in English <Converter.__init__-en>`

        .. _Converter.__init__-cn:

        :param dataloader: 数据加载器
        :type dataloader: Dataloader
        :param mode: 转换模式。目前支持三种模式，最大电流转换模式，99.9%电流转换模式，以及缩放转换模式
        :type mode: str, float

        ``Converter`` 用于将ReLU的ANN转换为SNN。这里实现了常见的三种模式。
        最常见的是最大电流转换模式，它利用前后层的激活上限，使发放率最高的情况能够对应激活取得最大值的情况。
        99.9%电流转换模式利用99.9%的激活分位点限制了激活上限。
        缩放转换模式下，用户需要给定缩放参数到模式中，即可利用缩放后的激活最大值对电流进行限制。

        * :ref:`中文API <VoltageScaler.__init__-cn>`

        .. _Converter.__init__-en:

        :param dataloader: Dataloader for converting
        :type dataloader: Dataloader
        :param mode: Conversion mode. Now support three mode, MaxNorm, RobustNorm(99.9%), and scaling mode
        :type mode: str, float

        ``Converter`` is used to convert ReLU's ANN to SNN. Three common methods are implemented here.
         The most common is the maximum mode, which utilizes the upper activation limits of
         the front and rear layers so that the case with the highest firing rate corresponds to the case where the
         activation achieves the maximum value.
         The 99.9% mode utilizes the 99.9% activation quantile to limit the upper activation limit.
         In the scaling conversion mode, the user needs to specify the scaling parameters into the mode, and the current
         can be limited by the activated maximum value after scaling.

        """
        super().__init__()
        self.mode = mode
        self.dataloader = dataloader
        self._check_mode()
        self.device = None
        
    def forward(self, relu_model):
        relu_model = copy.deepcopy(relu_model)
        if self.device is None:
            self.device = next(relu_model.parameters()).device
        relu_model.eval()
        model = self.set_voltagehook(relu_model, mode=self.mode).to(self.device)
        for _, (imgs, _) in enumerate(tqdm(self.dataloader)):
            model(imgs.to(self.device))
        model = self.replace_by_ifnode(model)
        return model

    def _check_mode(self):
        err_msg = 'You have used a non-defined VoltageScale Method.'
        if isinstance(self.mode, str):
            if self.mode[-1] == '%':
                try:
                    float(self.mode[:-1])
                except ValueError:
                    raise NotImplemented(err_msg)
            elif self.mode.lower() in ['max']:
                pass
            else:
                raise NotImplemented(err_msg)
        elif isinstance(self.mode, float):
            try:
                assert(self.mode <= 1 and self.mode > 0)
            except AssertionError:
                raise NotImplemented(err_msg)
        else:
            raise NotImplemented(err_msg)


    @staticmethod
    def set_voltagehook(model, mode='MaxNorm'):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = Converter.set_voltagehook(module, mode=mode)
                if module.__class__.__name__ == 'ReLU':
                    model._modules[name] = nn.Sequential(
                        nn.ReLU(),
                        VoltageHook(mode=mode)
                    )
        return model

    @staticmethod
    def replace_by_ifnode(model):
        for name,module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_by_ifnode(module)
                if module.__class__.__name__ == 'Sequential' and len(module) == 2 and \
                    module[0].__class__.__name__ == 'ReLU' and \
                    module[1].__class__.__name__ == 'VoltageHook':
                    s = module[1].scale.item()
                    model._modules[name] = nn.Sequential(
                        VoltageScaler(1.0 / s),
                        neuron.IFNode(v_threshold=1., v_reset=None),
                        VoltageScaler(s)
                    )
        return model