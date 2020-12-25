from pathlib import Path
import numpy as np
from typing import Callable
from config import defaults
import torch

from sem_seg.models.base import Model
from . import eval_metrices

DIRNAME = Path(__file__).parents[1].resolve()
OUTDIR = DIRNAME / "outputs" / "class_IOUs"
OUTDIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SegmentationModel(Model):
    """Child class for semantic segmentation model from base Model"""

    def __init__(self, network: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim,
                 criterion: Callable = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 additional_identifier: str = ''):
        super().__init__(network, dataloader, optimizer, criterion, lr_scheduler, additional_identifier)

        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=defaults.CITYSCAPES_IGNORED_INDEX)

        self.metrices = ['mIOU', 'pixel_accuracy']

    def store_per_class_iou(self):
        print('Calculating per class mIOU')
        self.network.eval()
        self.network.to(device)
        c_matrix = 0
        for inputs, labels in self.dataloader['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outs = self.network(inputs)
            c_matrix += eval_metrices.get_confusion_matrix(outs, labels)
        class_iou = eval_metrices.per_class_iu(c_matrix)
        class_iou = np.round(class_iou, decimals=6)
        file_path = str(OUTDIR) + '/' + self.name + '.csv'
        np.savetxt(file_path, class_iou, delimiter=',', fmt='%.6f')
        print(f'per class IOU saved at {file_path}')
