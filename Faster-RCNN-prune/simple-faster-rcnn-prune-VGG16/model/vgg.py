import torch.nn as nn
from utils.config import opt
# from .utils import load_state_dict_from_url


__all__ = [
    'VGG',  'vgg16',
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        if opt.student is True:
            super(VGG, self).__init__()
            self.features = features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.cfg = opt.cfg
            self.classifier = nn.Sequential(
                nn.Linear(self.cfg[16] * 7 * 7, 2816),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2816, 2816),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2816, num_classes),
            )
            if init_weights:
                self._initialize_weights()
        else:
            super(VGG, self).__init__()
            self.features = features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.cfg = opt.tea_cfg
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            if init_weights:
                self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if cfg is None:
        cfg = cfgs['D']
    model = VGG(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model




def vgg16(pretrained=False, progress=True, cfg=None, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', cfg, False, pretrained, progress, **kwargs)