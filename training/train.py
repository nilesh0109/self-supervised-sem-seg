import json
import torch
import argparse
from typing import Dict
import importlib
from config import defaults
from sem_seg.datasets.datautils import CityscapesUtils
from sem_seg.datasets.dataloader import CityscapesLoader
from sem_seg import networks, models


def run_experiment(experiment_config: Dict, load_weights: bool, save_weights: bool) -> None:
    """
    Run a Training Experiment
    :param experiment_config: Dict of following form
     {
     "dataset": "cityscapes",
     "model": "SegmentationModel",   # Type of the training model(SegmentationModel, SelfSupervisedModel)
     "network": "FCN8s",             # "FCN8s" | "FCN16s" | "FCN32s" | "deeplabv3" | "deeplabv3plus"
     "network_args": {"backbone":"Resnet50",   # "Resnet50" | "Resnet18" | "Resnet101" | "DRN50" etc
                      "pretrained": false,      # Whether the backbone model is pretrained
                       "load_from_byol": true,  # Whether to load the backbone weights from self-supervised trained model.
                       "freeze_backbone": false}, # Whether the backbone weights are fixed while training
     "train_args":{"batch_size": 8,
                    "epochs": 400,
                    "labels_percent": '10%',    #percetange of supervised labels to use for training. Default 100%
                     "log_to_tensorboard": false},
    "experiment_group":{}
}'
    :param load_weights: If true, load weights for the model from last run
    :param save_weights: If true, save model weights to sem_seg/weights directory
    :return: None
    """

    DEFAULT_TRAIN_ARGS = {'epochs': defaults.NUM_EPOCHS, 'batch_size': defaults.BATCH_SIZE,
                          'num_workers': defaults.NUM_WORKERS}
    train_args = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {})
    }
    experiment_config["train_args"] = train_args
    experiment_config["experiment_group"] = experiment_config.get("experiment_group", None)
    print(f'Running experiment with config {experiment_config}')

    labels_percent = train_args.get('labels_percent', '100%')
    mode = experiment_config.get('mode', 'supervised')
    dataset_name = experiment_config["dataset"].lower()
    assert dataset_name in ['cityscapes'], "The dataloader is only implemented for cityscapes dataset"
    data_loader = CityscapesLoader(label_percent=labels_percent) \
        .get_cityscapes_loader(batch_size=train_args["batch_size"],
                               num_workers=train_args["num_workers"],
                               mode=mode)
    models_module = importlib.import_module("sem_seg.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("sem_seg.networks")
    network_args = experiment_config.get("network_args", {})

    pretrained = network_args["pretrained"]
    backbone_class_ = getattr(networks_module, network_args["backbone"])
    base = backbone_class_(pretrained=pretrained)

    num_classes = CityscapesUtils().num_classes

    network_class_ = getattr(networks_module, experiment_config["network"])
    network = network_class_(base, num_classes)
    optim = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.1,
                                                              patience=15, min_lr=1e-10, verbose=True)
    additional_identifier = get_additional_identifier(network_args["backbone"], pretrained, dataset_name,
                                                      labels_percent)
    model = model_class_(network, data_loader, optim, lr_scheduler=lr_scheduler,
                         additional_identifier=additional_identifier)

    if not train_args['log_to_tensorboard']:
        model.logToTensorboard = False
        model.add_text_to_tensorboard(json.dumps(experiment_config))

    if load_weights:
        model.load_weights()
    elif network_args.get('load_from_BYOL', ''):
        byol = networks.BYOL(base)
        ss_model = models.SelfSupervisedModel(byol, additional_identifier=additional_identifier)
        print('loading self-supervised weights from ', ss_model.weights_file_name)
        byol_state_dict = torch.load(ss_model.weights_file_name)
        backbone_dict = {}
        for key in byol_state_dict:
            if 'online_network' in key:
                new_key = key.replace('online_network.', '')
                backbone_dict[new_key] = byol_state_dict[key]
        model.network.backbone.load_state_dict(backbone_dict)

    if network_args.get('freeze_backbone', False):
        for param in model.network.backbone.parameters():
            param.requires_grad = False

    model.train(num_epochs=train_args["epochs"])
    model.store_per_class_iou()

    if save_weights:
        model.save_weights()


def get_additional_identifier(backbone: str, pretrained: bool = False, dataset_name: str = '',
                              labels_percent: str = '100%') -> str:
    """
    Returns the additional_identifier added to the model name for efficient tracking of different experiments
    :param backbone: name of the backbone
    :param pretrained: Whether the backbone is pretrained
    :param dataset_name: Name of the training dataset
    :param labels_percent: % of labels used for training
    :return: additional identifier string
    """
    additional_identifier = backbone
    additional_identifier += '_pt' if pretrained else ''
    additional_identifier += '_' + labels_percent[:-1] if labels_percent and int(labels_percent[:-1]) < 100 else ''
    additional_identifier += '_ct' if dataset_name == 'cityscapes' else '_' + dataset_name
    return additional_identifier


def _parse_args():
    """ parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, action="store_true",
                        help="If true, final weights will be stored in canonical, version-controlled location")
    parser.add_argument("--load", default=False, action="store_true",
                        help="If true, final weights will be loaded from canonical, version-controlled location")
    parser.add_argument("experiment_config", type=str,
                        help='Experiment JSON (\'{"dataset": "cityscapes", "model": "SegmentationModel",'
                             ' "network": "fcn8s"}\''
                        )
    args = parser.parse_args()
    return args


def main():
    """Run Experiment"""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.load, args.save)


if __name__ == '__main__':
    main()
