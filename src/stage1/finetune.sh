env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -u finetune.py \
    --data_dir ../data_for_eric/1 \
    --output_dir=./model \
    --save_top_k 80 \
    --train_batch_size=12 \
    --eval_batch_size=12 \
    --num_train_epochs 20 \
    --model_name_or_path facebook/bart-base \
    --learning_rate=1e-4 \
    --gpus 1 \
    --do_train \
    --n_val 4000 \
    --val_check_interval 1.0 \
    --overwrite_output_dir