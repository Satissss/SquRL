#!/bin/bash
# PEFT (LoRA) + Sequence Parallel Training Script

export WANDB_API_KEY=...
export WANDB_PROJECT=squRL-sft
export CUDA_VISIBLE_DEVICES=6,7

set -x

HOME=.../SquRL/data
nproc_per_node=2  # Number of GPUs to use
save_path=...

# Sequence parallel configuration
sp_size=1  # Sequence parallel size, must divide nproc_per_node
use_remove_padding=false  # Set to true when sp_size > 1

# LoRA configuration
lora_rank=32
lora_alpha=32

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/dataset.parquet \
    data.val_files=$HOME/test.parquet \
    data.train_batch_size=256 \
    data.micro_batch_size_per_gpu=2 \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.max_length=8192 \
    data.balance_dp_token=false \
    model.partial_pretrain=... \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.lora_rank=$lora_rank \
    model.lora_alpha=$lora_alpha \
    model.use_liger=false \
    optim.lr=1e-5\
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=squRL-sft-peft \
    trainer.experiment_name=qwen25_7b_lora_r${lora_rank} \
    trainer.logger=wandb \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=8 \
    trainer.total_training_steps=null \
    ulysses_sequence_parallel_size=$sp_size \
    use_remove_padding=$use_remove_padding

