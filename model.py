import math
import torch.nn as nn
import torch.nn.functional as F
from config import CFG_VGG6, NUM_CLASSES

class VGG(nn.Module):
    def __init__(self, features, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.classifier(x.flatten(1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=True, activation=nn.GELU):
    layers, in_channels = [], 3
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            conv = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(value), activation()])
            else:
                layers.extend([conv, activation()])
            in_channels = value
    return nn.Sequential(*layers)

def build_model():
    return VGG(make_layers(CFG_VGG6, True, nn.GELU), NUM_CLASSES)
