#For supervised Training
config='{"dataset": "cityscapes", "model": "SegmentationModel", "network": "FCN8s",
"network_args":{"backbone":"Resnet50", "pretrained": false, "load_from_byol": true, "freeze_backbone": false},
"train_args":{"batch_size": 8, "epochs": 400, "log_to_tensorboard": false},
"experiment_group":{}
}'

python -m training.train --save "$config"