#!/bin/bash
set -x

nnodes=1
nproc_per_node=4

# For single node, use localhost
master_addr=localhost
master_port=29500
node_rank=0

experiment_name=feedback-sft-qwen3-8b
HDFS_ROOT=/workspace/verl/verl_code_base
DATA_ROOT=/workspace/verl/verl_code_base

TRAIN_DATA=$DATA_ROOT/data/feedback_sft_data.parquet
EVAL_DATA=$DATA_ROOT/data/feedback_sft_data.parquet
MODEL_PATH=Qwen/Qwen3-8B
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     --master-addr=$master_addr \
     --master-port=$master_port \
     --node-rank=$node_rank \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=feedback-sft \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=6 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
