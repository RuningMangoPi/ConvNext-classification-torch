python -m torch.distributed.launch --nproc_per_node=4 \
--master_addr="192.168.1.201"  --master_port=55421 main.py \
--model convnext_tiny --drop_path 0.4 --input_size 528 \
--batch_size 4 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 100 --weight_decay 1e-8  \
--data_set image_folder \
--data_path ./datasets/val/train \
--eval_data_path ./datasets/val \
--finetune ./checkpoint-best.pth \
--output_dir ./results \
--log_dir    ./results \
--nb_classes 10 \
--use_amp  True \
--cutmix 0
# --master_addr: local host
# --nproc_per_node:  num of gpus
# --model : convnext_tiny or tf_efficientnet_b6 or .....