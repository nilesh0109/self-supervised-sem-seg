from typing import List
import torch
from torchvision import models

def get_num_features(backbone_name: str, model_type: str='') -> List:
    """
    Gives a List of features present in the last 3 blocks of the backbone model
    :param backbone_name: name of the backbone model e.g. 'resnet18' | 'resnet50'
    :param model_type: Type of FCN model(fcn32s | fcn16s | fcn8s)
    :return: List of number of features extracted from last 3 blocks of the backbone model
    """

    if 'resnet18' in backbone_name.lower():
        num_features = [64, 128, 256, 512]
    else:
        num_features = [256, 512, 1024, 2048]
    if 'fcn8s' in model_type.lower():
        num_features = num_features[-3:]
    elif 'fcn16s' in model_type.lower():
        num_features = num_features[-2:]
    elif 'fcn32s' in model_type.lower():
        num_features = num_features[-1:]
    return num_features


def get_features_dict(base: torch.nn.Module) -> torch.nn.Module:
    """
    This function extracts the features from various layers(hardcoded in a dictionary extract_layers) and returns
    them as a key-value pair.
    For e.g. extract_layer = {'layer2': 'feat3', 'layer3': 'feat4'} will extract 'layer2' and 'layer3' feature maps
    from the given network and assigns them to key 'feat3' and 'feat4' respectively of the output.
    :param base: backbone model
    :return: model with output layer as dictionary which provides extracted features from various layers.
    """
    extract_layers = {'layer1': 'feat1', 'layer2': 'feat3', 'layer3': 'feat4', 'layer4': 'feat5'}
    return models._utils.IntermediateLayerGetter(base, extract_layers)

