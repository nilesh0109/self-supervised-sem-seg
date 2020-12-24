from typing import List, Dict
from torch import nn
from .network_utils import get_num_features


class FCN(nn.Module):
    """ Base Class for all FCN Modules """

    def __init__(self, backbone: nn.Module, num_classes: int, model_type: str = 'fcn8s'):
        super().__init__()
        num_features = get_num_features(backbone.name, model_type)
        self.classifier = nn.ModuleList([self.upsample_head(num_features, num_classes) for num_feature in num_features])

    def upsample_head(self, in_channels: int, channels: int) -> nn.Module:
        """
        :param in_channels: Number of channels in Input
        :param channels: Desired Number of channels in Output
        :return: torch.nn.Module
        """
        inter_channels = in_channels // 8
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        """ Abstract method to be implemented by child classes"""
        pass


class FCN32s(FCN):
    """ Child FCN class that generates the output only using feature maps from last layer of the backbone """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__(backbone, num_classes, model_type='fcn32s')

    def forward(self, x):
        """ Forward pass through FCN32s"""
        h, w = x.shape[-2:]
        features = self.backbone(x)
        return self.bilinear_upsample(features, h, w)

    def bilinear_upsample(self, features: Dict, h: int, w: int):
        """
        :param features: Backbone's output feature map dict
        :param h: Desired Output Height
        :param w: Desired output Width
        :return: Upsample output of size N x C x H x W where C is the number of classes
        """
        out32s = self.classifier[-1](features['feat5'])
        upsampled_out = nn.functional.interpolate(out32s, size=(h, w), mode='bilinear', align_corners=False)
        return upsampled_out


class FCN16s(FCN):
    """ Child FCN class that generates the output only using feature maps from last two layers of the backbone """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__(backbone, num_classes, model_type='fcn16s')

    def forward(self, x):
        """ Forward pass through FCN16s"""
        h, w = x.shape[-2:]
        features = self.backbone(x)
        return self.bilinear_upsample(features, h, w)

    def bilinear_upsample(self, features: Dict, h: int, w: int):
        """
        Bilinear upsample after merging the last 2 feature maps
        :param features: Backbone's output feature map dict
        :param h: Desired Output Height
        :param w: Desired output Width
        :return: Upsample output of size N x C x H x W where C is the number of classes
        """
        out32s = self.classifier[-1](features['feat5'])
        out16s = self.classifier[-2](features['feat4'])
        upsampled_out32s = nn.functional.interpolate(out32s, size=(h//16, w//16), mode='bilinear', align_corners=False)
        out = upsampled_out32s + out16s
        upsampled_out = nn.functional.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return upsampled_out


class FCN8s(FCN):
    """ Child FCN class that generates the output only using feature maps from last three layers of the backbone """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__(backbone, num_classes, model_type='fcn8s')

    def forward(self, x):
        """ Forward pass through FCN16s"""
        h, w = x.shape[-2:]
        features = self.backbone(x)
        return self.bilinear_upsample(features, h, w)

    def bilinear_upsample(self, features: Dict, h: int, w: int):
        """
        Bilinear upsample after merging the last 3 feature maps
        :param features: Backbone's output feature map dict
        :param h: Desired Output Height
        :param w: Desired output Width
        :return: Upsample output of size N x C x H x W where C is the number of classes
        """
        out32s = self.classifier[-1](features['feat5'])
        out16s = self.classifier[-2](features['feat4'])
        out8s  = self.classifier[-3](features['feat3'])
        upsampled_out32s = nn.functional.interpolate(out32s, size=(h//16, w//16), mode='bilinear', align_corners=False)
        out = upsampled_out32s + out16s
        upsampled_out16s = nn.functional.interpolate(out, size=(h//8, w//8), mode='bilinear', align_corners=False)
        out = upsampled_out16s + out8s
        upsampled_out = nn.functional.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return upsampled_out

