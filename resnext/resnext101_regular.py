import torch
from torch import nn

from resnext import resnext_101_32x4d_
resnext_101_32_path = 'resnext_101_32x4d.pth'


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext_101_32_path))

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]

    def forward(self, x):
        outs = []
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        outs.append(layer0)
        outs.append(layer1)
        outs.append(layer2)
        outs.append(layer3)
        # layer4 = self.layer4(layer3)
        return layer3,outs