import numpy as np
from torchvision import datasets
from config.defaults import CITYSCAPES_PATH


class CityscapesUtils:
    """CITYSCAPES Utility class that provides the mapping for training labels and their colors for visualization"""
    def __init__(self):
        cityscapes_data = datasets.Cityscapes(CITYSCAPES_PATH, split='train', mode='fine', target_type='semantic')
        self.classes = cityscapes_data.classes
        self.num_classes = self._num_classes()
        self.train_id2color = self._train_id2color()
        self.id2train_id = self._id2train_id()

    def _num_classes(self) -> int:
        """
        :return: returns the effective number of classes in cityscapes that are used in validation
        """
        train_labels = [label.id for label in self.classes if not label.ignore_in_eval]
        return len(train_labels)

    def _id2train_id(self) -> np.array:
        """
        :return: returns a list where each index is mapped to its training_id. All ignore_in_eval indexes are mapped to 0
        i.e. the unlabelled class.
        """
        train_ids = np.array([label.train_id for label in self.classes])
        train_ids[(train_ids == -1) | (train_ids == 255)] = 19   # 19 is Ignore_index(defaults.CITYSCAPES_IGNORE_INDEX)
        return train_ids

    def _train_id2color(self) -> np.array:
        """
        :return: The mapping of 20 classes (19 training classes + 1 ignore index class) to their standard color used
        in cityscapes.
        """
        return np.array([label.color for label in self.classes if not label.ignore_in_eval] + [(0, 0, 0)])

    def label2color(self, mask: np.array) -> np.array:
        """
        Given the cityscapes mask with all training id(and optionally 255 for ignored labels) as labels, returns the mask
        filled with the label's standard color.
        :param mask: np.array mask for which color mapping is required
        :return: mask with labels replaced with their standard colors
        """
        return self.train_id2color[mask]
