#!/bin/bash
set -x

# ================= logging setup =================
# Create timestamped log directory
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir=$PWD/logs/feedback_rl_qwen3_14b_lora_${timestamp}
mkdir -p $log_dir

echo "Logging to: $log_dir"
echo "  - Console: Real-time output in terminal"
echo "  - Full training log: $log_dir/training.log"
echo "  - Validation outputs: $log_dir/validation/*.jsonl"

# ================= data/model/tool =================
HDFS_ROOT=/workspace/verl/verl_code_base
DATA_ROOT=/workspace/verl/verl_code_base/data

feedback_sft_data=$DATA_ROOT/feedback_sft_data.parquet

# Use Qwen3-14B model (smaller than 32B)
model_path=Qwen/Qwen3-14B

# Use MATH dataset for RL training (~7500 test examples from HuggingFace)
# Use AIME 2024 for validation (harder, tests generalization)
train_files="['math']"
test_files="['aime_2024']"

# tool
tool_config_path=recipe/feedback_rl/feedback_tool_config.yaml

# wandb
project_name=feedback_rl_training
experiment_name=qwen3-14b_feedback_rl_lora8
default_local_dir=$HDFS_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

# Enable some KL to prevent model collapse when learning from feedback
use_kl_in_reward=False
kl_coef=0.1
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Multi-turn parameters
max_turns=7  # Allow up to 6 feedback rounds (initial + 6 feedback cycles)
max_prompt_length=2048
max_response_length=8192  # Shorter than DAPO since we focus on reasoning
actor_lr=5e-7  # Lower LR for feedback learning with LoRA

# Since 14B is smaller, we can use larger batch sizes
train_batch_size=32  # Can handle larger batches
ppo_mini_batch_size=8
n_resp_per_prompt=2  # Generate 2 responses per prompt
n_resp_per_prompt_val=4

# ================= LoRA parameters =================
lora_rank=8  # Minimum recommended rank
lora_alpha=16  # Standard alpha value

# ================= performance =================
infer_tp=2 # vllm tensor parallel - 14B fits well with tp=2
train_sp=2 # train sequence parallel
offload=True

# With 14B model, we have good memory headroom
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

# Run training with tee to log to both console and file
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/feedback_rl/dataset.py \
    data.custom_cls.name=FeedbackRLDataset \
    custom_reward_function.path=recipe/feedback_rl/rewards.py \
    custom_reward_function.name=compute_feedback_reward \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$lora_rank \
    actor_rollout_ref.model.lora_alpha=$lora_alpha \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.log_val_generations=50 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.default_local_dir=$default_local_dir \
    trainer.validation_data_dir=$log_dir/validation \
    trainer.test_freq=20 \
    trainer.total_epochs=5 \
    2>&1 | tee $log_dir/training.log
