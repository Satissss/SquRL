export WANDB_API_KEY=...
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=6,7

DATA_DIR_PATH=data
RUN_ID=0.5B-FSDP
GPU_ENV=2GPU
MODEL_ENV=qwen25-7B-rl-plus_api-0114
PROJECT_NAME=SquRL-new
EXPERIMENT_NAME=$MODEL_ENV

LOG_PATH=logs/$PROJECT_NAME
MODEL_PATH=/llm_jzm/yihan/model/qwen-7b-lora-1228

mkdir -p $LOG_PATH

set -x

nvidia-smi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR_PATH/rl_dataset_dynamic.parquet \
    data.val_files=$DATA_DIR_PATH/rl_dataset_dynamic.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.clip_ratio=0.6 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    algorithm.kl_ctrl.kl_coef=0.0005\
    reward_server.api_port=6517 \
    reward_server.api_timeout=1500 \
    reward_server.batch_size=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2  \
    trainer.nnodes=1 \
    trainer.default_local_dir=/llm_jzm/yihan/checkpoint/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 $@ 2>&1 | tee $LOG_PATH/$MODEL_ENV/fsdp_grpo.log