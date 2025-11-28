set -x

HOME=/root/rl/SquRL
nproc_per_node=2
save_path=/mnt/yihan/checkpoint/sft_test/qwen25_0_5_B_instuct

# Shift the arguments so $@ refers to the rest


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/examples/data/train.parquet \
    data.val_files=$HOME/examples/data/test.parquet \
    optim.lr=1e-6 \
    data.micro_batch_size=4 \
    model.partial_pretrain=/mnt/yihan/model/qwen25_0_5_B_instuct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-3-0.6b-instruct-sp2 \
    trainer.logger=wandb \
    trainer.default_hdfs_dir=null\
    trainer.total_epochs=2\
    use_remove_padding=true
