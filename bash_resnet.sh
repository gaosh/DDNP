#train the base model
CUDA_VISIBLE_DEVICES=0 nohup python train_model.py --smooth_flag False --train_base True --lr 0.1 > base_rn56.txt 2>&1 &
#prune the model
python pruning_resnet.py
#finetune the model
CUDA_VISIBLE_DEVICES=0 nohup python train_model.py --smooth_flag False --train_base False --warmup True --lr 0.1 > ft_mbv2.txt 2>&1 &

