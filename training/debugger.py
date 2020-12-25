from pathlib import Path
import numpy as np
import torch
import torchvision


from config import defaults
defaults.MATPLOTLIB_NO_GUI = False # enable GUI for matplotlib
from sem_seg.datasets.dataloader import CityscapesLoader
from sem_seg.datasets.datautils import CityscapesUtils
from sem_seg.networks import FCN8s, FCN32s, FCN16s, Resnet50
from sem_seg.models.segmentation_model import SegmentationModel
from sem_seg.models.self_supervised_model import SelfSupervisedModel
from training import utils, train


DIRNAME = Path(__file__).parents[1].resolve()
OUTDIR = DIRNAME / "semantic_seg" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


def get_dataloader(mode='supervised'):
    cityscapes = CityscapesLoader(label_percent='100%').get_cityscapes_loader(mode=mode)
    return cityscapes

def get_model():
    cityscapes = get_dataloader()
    cityscapes_utils = CityscapesUtils()
    num_classes = cityscapes_utils.num_classes + 1
    num_features = [256, 512, 512]
    base = Resnet50(pretrained=False)
    fcn8s = FCN8s(base, num_classes)
    optim = torch.optim.Adam(fcn8s.parameters())
    model = SegmentationModel(fcn8s, cityscapes, optim)
    return model

def debug_FCN():
    cityscapes = get_dataloader()
    batch_imgs, batch_targets = next(iter(cityscapes['train']))
    utils.plot_images(batch_imgs, batch_targets, title='Predictions')

def debug_self_supervised_model():
    cityscapes = get_dataloader(mode='self-supervised')
    (batch_imgs, tf_imgs, seeds), _ = next(iter(cityscapes['train']))
    preds = None
    utils.plot_images(batch_imgs, tf_imgs, preds, title='Predictions')

if __name__ == '__main__':
    debug_FCN()
    debug_self_supervised_model()
