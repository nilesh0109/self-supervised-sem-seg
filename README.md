# Self-suerpvised Semantic Segmentation 
This repository is related to semantic segmentation models trainined in
self-supervised manner. In a two step training process, the backbone
of the segmentation network(say FCN8s network) is trained first
in self-supervised way and then the whole network is fine-tuned
with few semantic segmentation annotations.

![BYOL](https://github.com/nilesh0109/self-supervised-sem-seg/blob/master/byol.png)

The baseline segmentation network used is FCN8s and
various backbones are: Resnet50, Resnet101, DRN50.

To reproduce the experiments, follow the below steps.

# How To Run The Code
1. Create the virtual environment with all dependencies and activate it

``` bash scripts/create_env.sh```
   
2. Train the Self-supervised Model first using unlabelled data.

``` bash scripts\unsup_train.sh ```
   
3. Train the supervised model e2e with using only labelled
data.

``` bash scripts\train.sh```

# Results:
Resnet50 based Semantic segmentation FCN8s network, gets 10% improvement in mIOU on cityscapes dataset if it is first pretraited in BYOL manner using 20k unlabelled images and then fine-tuned with 5k labelled cityscapes images.

X-axis in below graph shows the percetage of labels(out of 5k images) used in fine-tuning step. Top dotted line in the graph is Resnet50 trained from imagenet weights and bottom dotted line is Resnet50 trained from random weights(no pretraining). Middle two plots are Resnet50 pretrained using BYOL and another variant of Resnet50(Resnet50 with weight standarization and group normalization[2] applied) trained using BYOL[1]. Red arrows clearly indicates the success of BYOL for visual representation learning.

![BYOL gain](https://github.com/nilesh0109/self-supervised-sem-seg/blob/master/BYOL_gain.png)


# Code Walkthrough
```
config
   - defaults.py: Configuration file for setting the defaults
scripts
   - create_env.sh: script file for creating the virtual environment with dependencies listed in environment.yml
   - test.sh: script file for debugging the inputs & outputs to the FCN8s and BYOL network.
   - train.sh: script for training the segmentation network
   - unsup_train.sh: script for training the unsupervised learning(BYOL) network
sem_seg
   datasets
      - augmentations.py: Utility file for various augmentations
      - dataloader.py: Script for loading cityscapes dataloader
      - dataset.py: Custom cityscapes dataset for applying same set of augmentations to images and masks
      - datautils.py: Script for cityscapes labels formatting
      - transforms.py: Custom transformation scripts
   models
      - base.py: Base class for all models with train and evaluation pipeline
      - eval_metrices.py: Script for different evaluation metrices such as mIOU, accuracy, etc.
      - segmentation.py: Class for semantic segmentaion model inherited from base model
      - self_supervised_model.py: Self-supervised-model(BYOL) training pipeline 
   network
      - byol.py: BYOL network setup
      - fcn.py: Various FCN(Fully Convolutional Network) setup
      - network_utils.py: Utility script for vairous network configurations
      - resnet.py: Various Residual network(Resnet) setup
training
      - debugger.py: Debugger script for FCN and BYOL model
      - train.py: Training file for FCN model
      - train_byol.py: Training file for BYOL model
      - utils.py: Utility file for plotting various inputs and outputs 
- environment.yml: List of virtual environment dependencies
```
# References
1. Grill, Jean-Bastien, et al. "Bootstrap your own latent: A new approach to self-supervised learning." arXiv preprint arXiv:2006.07733 (2020).
2. Qiao, Siyuan et al. “Micro-Batch Training with Batch-Channel Normalization and Weight Standardization.” (2019).
