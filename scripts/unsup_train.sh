# for unsupervised Training

config='{"dataset": "cityscapes", "model": "SelfSupervisedModel", "network": "BYOL", "mode": "self-supervised",
"network_args":{"backbone":"Resnet50", "pretrained": false, "target_momentum": 0.996},
"train_args":{"batch_size": 32, "epochs": 700, "log_to_tensorboard": false},
"experiment_group":{}
}'

python -m training.train_byol --save "$config"

#Use for distributed Training
#python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 -m training.train_byol --save "$config"