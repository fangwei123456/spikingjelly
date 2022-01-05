from spikingjelly.clock_driven.ann2snn.modules import *
from tqdm import tqdm
from spikingjelly.clock_driven import neuron


class Converter(nn.Module):

    def __init__(self, device, dataloader, mode='MaxNorm'):
        super().__init__()
        self.device = device
        self.mode = mode
        self.dataloader = dataloader
        
    def forward(self, relu_model):
        relu_model.eval()
        model = self.set_voltagehook(relu_model, mode=self.mode).to(self.device)
        for _, (imgs, _) in enumerate(tqdm(self.dataloader)):
            model(imgs.to(self.device))
        model = self.replace_by_ifnode(model)
        return model


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