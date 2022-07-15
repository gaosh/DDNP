# Disentangled Differentiable Network Pruning
Pytorch Implementation of Disentangled Differentiable Network Pruning (ECCV 2022)
# Requirements
pytorch==1.7.1  
torchvision==0.8.2
# Usage
To run the full cycle of training, pruning and finetuning for ResNet-56:
```
bash bash_resnet.sh
```
To train a base model
```
CUDA_VISIBLE_DEVICES=0 python train_model.py --train_base True
```
To train the pruning algorithm
```
CUDA_VISIBLE_DEVICES=0 python resnet_topk.py --reg_w 2 --base 3.0
```
To prune the model
```
python pruning_resnet.py
```
To finetune the model 
```
python train_model.py--train_base False
```
# Citation
If you found this repository is helpful, please consider to cite our paper:
```
@InProceedings{Gao_2022_ECCV,
    author    = {Gao, Shangqian and Huang, Feihu and Zhang, Yanfu and Huang, Heng},
    title     = {Disentangled Differentiable Network Pruning},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2022},
}
