from typing import List, Tuple
import numpy as np
import torch
from torchvision import transforms as tfms
from .datautils import CityscapesUtils
from .transforms import RandomResize
from PIL import Image


def get_tfms() -> Tuple:
    """
    :return: Image and mask transforms tuple for cityscapes fine annotated images
    """
    base_size = 800
    min_size, max_size = int(0.5*base_size), int(2.0*base_size)
    img_transforms = {
        'train': tfms.Compose([
            RandomResize(min_size, max_size),
            tfms.RandomCrop(size=768, pad_if_needed=True, fill=0),
            tfms.RandomHorizontalFlip(p=0.5),
            tfms.ToTensor()
        ]),
        'val': tfms.Compose([
            tfms.ToTensor()
        ])
    }

    target_transforms = {
        'train': tfms.Compose([
            RandomResize(min_size, max_size, is_mask=True),
            tfms.RandomCrop(size=768, pad_if_needed=True, fill=0),
            tfms.RandomHorizontalFlip(p=0.5),
            mapId2TrainID
        ]),
        'val': tfms.Compose([mapId2TrainID])
    }
    return img_transforms, target_transforms

cityscapes_utils = CityscapesUtils()

def mapId2TrainID(mask: Image) -> torch.Tensor:
    """
    Redcudes the 34 labels present in gt_fine to 19 labels. Ignoring the ignore_in_eval labels.
    :param mask: Cityscapes mask(PIL Image type) with 34 classes
    :return: mask with 19 cityscapes classes of tensor type
    """
    return torch.from_numpy(cityscapes_utils.id2train_id[np.array(mask)]).long()


def get_spatial_tfms() -> tfms.Compose:
    """ Get the transformation which changes the spatial position of the object in the image"""
    return tfms.Compose([
        tfms.RandomResizedCrop(size=512, scale=(0.3, 1)),
        tfms.RandomHorizontalFlip()
    ])


def get_pixelwise_tfms() -> List[tfms.Compose]:
    """Get the transformations which only introduces local perturbation to the image."""
    tfms_list = [
        tfms.RandomApply([
            tfms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        tfms.RandomGrayscale(p=0.2)
    ]
    return tfms_list


def get_self_supervised_tfms() -> tfms.Compose:
    """ Returns the list of transformations to be used to generating self-supervised training image pairs"""
    spatial_tfms = get_spatial_tfms()
    pixelwise_tfms = get_pixelwise_tfms()

    self_supervised_tfms = tfms.Compose([
        *pixelwise_tfms,
        tfms.ToTensor(),
        spatial_tfms,
        tfms.Lambda(lambda img: img.squeeze())
    ])
    return self_supervised_tfms
