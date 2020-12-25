from typing import Callable, Union, List, Tuple
import random
import numpy as np
import functools
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset

from .augmentations import get_pixelwise_tfms


class CityscapesDataset(Dataset):
    """Custom cityscapes dataset to apply same transformation on image and mask pair"""

    def __init__(self, cityscapes_data: Union[Dataset, List], label_percent: int = 100, transform: Callable=None,
                 target_transform: Callable = None, is_test: bool = False, mode: str = 'supervised'):
        """
        :param cityscapes_data: torch loaded Cityscapes dataset from path defaults.CITYSCAPES_PATH
        :param label_percent: % of labels to use
        :param transform: List of Transformations to be applied on the input Image
        :param target_transform: List of transformations to be applied on the segmentaion mask
        :param is_test: is Test/Val dataset
        :param mode: type of dataset. 'supervised' | 'self-supervised'. 'self-supervised' mode will generate <img, transformed_img>
        pair and 'supervised' mode will generate <img, label> pair
        """

        inputs = functools.reduce(lambda a, b: a + b.images, cityscapes_data, []) \
            if type(cityscapes_data) == list else cityscapes_data.images
        total = len(inputs)
        n = int(total * label_percent) // 100
        self.imgs_path = inputs[:n]
        self.masks_path = None if is_test or mode.lower() == 'self-supervised' else cityscapes_data.targets
        self.transform = transform
        self.target_transform = target_transform
        self.isTestSet = is_test
        self.mode = mode.lower()
        assert self.mode in ['supervised', 'self-supervised'], "Invalid dataset mode. Only 'supervised', " \
                                                               "'self-supervised' is allowed"

    def __len__(self):
        return len(self.imgs_path)

    def get_random_crop(self, image: Image, crop_size: int) -> Image:
        """
        :param image: PIL Image
        :param crop_size: Size of the crop
        :return: PIL Image crop of size crop_size
        """
        crop_tfms = torchvision.transforms.RandomCrop(crop_size)
        return crop_tfms(image)

    def _apply_transformations(self, image: Image, mask: Image = None) -> Tuple:
        """

        :param image: PIL Image
        :param mask: PIL segmentation Mask
        :return: Transformed Image and Mask pair after applying exactly the same random transformations on image and
        mask
        """
        seed = np.random.randint(2147483647)            # sample a random seed
        random.seed(seed)                               # set this seed to random and numpy.random function
        np.random.seed(seed)
        torch.manual_seed(seed)                         # Needed for torchvision 0.7
        if self.transform is not None:
            image = self.transform(image)

        if mask is not None and self.target_transform is not None:
            random.seed(seed)                           # set the same seed to random and numpy.random function
            np.random.seed(seed)
            torch.manual_seed(seed)                     # Needed for torchvision 0.7
            mask = self.target_transform(mask)
        return image, mask, seed

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.imgs_path[idx]
        image = Image.open(image_path)
        if image.mode == 'P':
            image = image.convert('RGB')

        if self.mode == 'self-supervised':
            crop = self.get_random_crop(image, 512)
            tf_image1, _, seed = self._apply_transformations(crop)
            crop_tensor = torchvision.transforms.Compose([
                *get_pixelwise_tfms(),
                torchvision.transforms.ToTensor()
            ])(crop)
            return (crop_tensor, tf_image1, seed), seed
        else:
            if not self.isTestSet:
                mask_path = image_path.replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')
                mask = Image.open(mask_path)
                image, mask, seed = self._apply_transformations(image, mask)
            return image if self.isTestSet else (image, mask)

