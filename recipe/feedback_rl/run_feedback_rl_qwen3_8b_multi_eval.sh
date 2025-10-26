#!/bin/bash
set -x

# ================= logging setup =================
# Create timestamped log directory
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir=$PWD/logs/feedback_rl_qwen3_8b_multi_eval_${timestamp}
mkdir -p $log_dir

echo "Logging to: $log_dir"
echo "  - Console: Real-time output in terminal"
echo "  - Full training log: $log_dir/training.log"
echo "  - Validation outputs: $log_dir/validation/*.jsonl"

# ================= data/model/tool =================
HDFS_ROOT=/workspace/verl/verl_code_base
DATA_ROOT=/workspace/verl/verl_code_base/data

feedback_sft_data=$DATA_ROOT/feedback_sft_data.parquet

# Use Qwen3-8B SFT checkpoint (trained on feedback tool usage)
model_path=/workspace/verl/verl_code_base/checkpoint/feedback-sft-qwen3-8b/global_step_78/huggingface

# Training: Use 'math_train' which samples 133 problems (matches working run exactly)
# Testing hypothesis: 665 problems (math_train_train) may be too hard → low accuracy → weak learning signal
# vs 133 sampled problems which had 75% accuracy in working run
train_files="['math_train']"

# Validation: 5 separate evaluation sets
# 1. math_train_test: Sampled 1/5 of MATH training data (~133, overfitting check)
# 2. math_holdout_test: Sampled 1/5 of MATH holdout data (~133, generalization)
# 3. mmlu_pro_test: 50 samples from MMLU-Pro test set
# 4. aime_2024: Harder math problems (~30)
# 5. gpqa: 100 samples from GPQA diamond
test_files="['math_train_test', 'math_holdout_test', 'mmlu_pro_test', 'aime_2024', 'gpqa']"

# tool
tool_config_path=recipe/feedback_rl/feedback_tool_config.yaml

# wandb
project_name=feedback_rl_training
experiment_name=qwen3-8b-sft_multi_eval
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
max_prompt_length=4096  # Increased for GPQA questions with multiple choice options
max_response_length=16384  # Increased to allow full multi-turn conversations without truncation
actor_lr=5e-7  # Lower LR for feedback learning

# 8B model - balanced batch sizes
train_batch_size=12  # Reduced from 16 to save memory with long sequences
ppo_mini_batch_size=3  # Reduced from 4 to save memory with long sequences
n_resp_per_prompt=4  # More responses to see variety
n_resp_per_prompt_val=4  # Same for validation

# Validation batch size - will be split across 5 datasets
# math_train_test: ~133 problems
# math_holdout_test: ~133 problems
# mmlu_pro_test: 50 problems
# aime_2024: ~30 problems
# gpqa: 100 problems
# Total: ~446 problems
val_batch_size=200  # Process in batches

# ================= performance =================
infer_tp=1 # vllm tensor parallel - 8B easily fits on single GPU
train_sp=2 # train sequence parallel
offload=True

# With 8B model and full fine-tuning, use conservative settings
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
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.custom_cls.path=recipe/feedback_rl/dataset.py \
    data.custom_cls.name=FeedbackRLDataset \
    custom_reward_function.path=recipe/feedback_rl/rewards.py \
    custom_reward_function.name=compute_feedback_reward \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
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
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.log_val_generations=50 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$default_local_dir \
    trainer.validation_data_dir=$log_dir/validation \
    trainer.test_freq=40 \
    trainer.total_epochs=30 \
    2>&1 | tee $log_dir/training.log
