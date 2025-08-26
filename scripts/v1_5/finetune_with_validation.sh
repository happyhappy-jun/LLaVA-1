#!/bin/bash

# Example script showing how to use multiple JSON files with image validation
# The script will validate all images before training starts
# --data_path /home/ec2-user/data/0825_sft_libero_90_full.json,/home/ec2-user/data/llava_v1_5_mix665k.json \

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /home/ec2-user/data/0825_sft_libero_90_full.json,/home/ec2-user/data/llava_v1_5_mix665k.json \
    --image_folder /home/ec2-user/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter artifacts/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --validate_images True \
    --skip_missing_images False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-0825-full-tree \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /home/ec2-user/data/0825_sft_libero_strong_only_full.json,/home/ec2-user/data/llava_v1_5_mix665k.json \
    --image_folder /home/ec2-user/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter artifacts/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --validate_images True \
    --skip_missing_images False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-0825-strong-only \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb



# Options for image validation:
# --validate_images True/False     : Enable/disable image validation (default: True)
# --skip_missing_images True/False : Continue training even with missing images (default: False)
#
# Behavior:
# - If validate_images=True and skip_missing_images=False:
#   Training will stop with an error if any images are missing
#
# - If validate_images=True and skip_missing_images=True:
#   Training will continue, but skip samples with missing images
#
# - If validate_images=False:
#   No validation is performed (original behavior)