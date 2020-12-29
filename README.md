This repository is related to semantic segmentation models trainined in
self-supervised manner. In a two step training process, the backbone
of the segmentation network(say FCN8s network) is trained first
in self-supervised way and then the whole network is fine-tuned
with few semantic segmentation annotations.

![BYOL](https://github.com/nilesh0109/self-supervised-sem-seg/blob/master/byol.png)

The baseline segmentation network used is FCN8s and
various backbones are: Resnet50, Resnet101, DRN50.

To reproduce the experiments, follow the below steps.

1. Create the virtual environment with all dependencies and activate it

``` bash scripts/create_env.sh```
   
2. Train the Self-supervised Model first using unlabelled data.

``` bash scripts\unsup_train.sh ```
   
3. Train the supervised model e2e with using only labelled
data.

``` bash scripts\train.sh```

