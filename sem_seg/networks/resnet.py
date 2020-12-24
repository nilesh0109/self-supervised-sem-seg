from torchvision import models
import torch
from typing import List
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from pathlib import Path
from .network_utils import get_features_dict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _build_model(resnet_type: str = 'resnet50', pretrained: bool = False,
                 replace_stride_with_dilation: List = [False, False, False]) -> torch.nn.Module:
    """
    :param resnet_type: 'resnet18' | 'resnet50' | 'resnet101' | 'resnet152'
    :param pretrained: True if the network is expected to be initialized with pretrained imagenet weights
    :param replace_stride_with_dilation: List of 3 boolean values if the last 3 blocks of resnet should use dilation
    instead of stride
    :return: torch.nn.Module
    """
    if resnet_type == 'resnet18':
        base = models.resnet18(pretrained=False, replace_stride_with_dilation=replace_stride_with_dilation)
    elif resnet_type == 'resnet50':
        base = models.resnet50(pretrained=False, replace_stride_with_dilation=replace_stride_with_dilation)
    elif resnet_type == 'resnet101':
        base = models.resnet101(pretrained=False, replace_stride_with_dilation=replace_stride_with_dilation)
    elif resnet_type == 'resnet152':
        base = models.resnet152(pretrained=False, replace_stride_with_dilation=replace_stride_with_dilation)
    elif resnet_type == 'resnet50_2x':
        base = models.wide_resnet50_2(pretrained=False, replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        print(f'resent type {resnet_type} is not currently implemented')
        return

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[resnet_type], progress=True)
        base.load_state_dict(state_dict)

    network = get_features_dict(base)
    network.isDRN = any(replace_stride_with_dilation)  # Is Dilated Residual Network
    network.name = resnet_type
    return network


Resnet18 = lambda pretrained: _build_model('resnet18', pretrained)
Resnet50 = lambda pretrained: _build_model('resnet50', pretrained)
Resnet101 = lambda pretrained: _build_model('resnet101', pretrained)
Resnet152 = lambda pretrained: _build_model('resnet152', pretrained)
DRN50 = lambda pretrained: _build_model('resnet50', pretrained, replace_stride_with_dilation = [False, True, True])
Resnet50_2 = lambda pretrained: _build_model('resnet50_2x', pretrained)
Resnet101_2 = lambda pretrained: _build_model('resnet101_2x', pretrained)


