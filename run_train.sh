for seed in 1 
do 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=12345 train.py /data/megh98/projects/dev_folder/smrai-container-documentation/src/data/imagenet --seed $seed --rand-subset-frac 0.1 --model DaViT_tiny --batch-size 512 --lr 1e-3 --native-amp --clip-grad 1.0 --output ./outputs/ --val-split val --log-wandb
done
 

 # --resume /mnt/disks/scratch0/megh98/projects/davit/outputs/20231123-003145-DaViT_tiny-224/model_best.pth.tar
