# Feedback RL Training

Multi-turn RL training for learning to incorporate feedback, based on Self-InPerfect research.

## Files

- **convert_to_tool_sft.py**: Converts Self-InPerfect data to tool-calling SFT format
- **run_feedback_sft.sh**: SFT training script (MUST run before RL!)
- **feedback_tool.py**: Custom tool that calls vLLM server for feedback generation
- **feedback_tool_config.yaml**: Tool configuration with exact Self-InPerfect parameters
- **feedback_utils.py**: Self-contained utility functions copied from Self-InPerfect
- **dataset.py**: Dataset class that converts Self-InPerfect datasets to VERL format
- **rewards.py**: Reward function for feedback incorporation
- **run_feedback_rl.sh**: RL training script
- **test_feedback_tool.py**: Test script to verify vLLM server connection

## Requirements

1. **vLLM server** running feedback model at `http://h05:1233`
2. **Base Model**: Qwen2.5-32B-Instruct (or your base model)
3. **Self-InPerfect data**: `/scratch/dkhasha1/djiang21/Self-InPerfect_new/llama3.3_math_4.1_mini_feedback.jsonl`

## Complete Training Pipeline

### Step 1: Convert Data to SFT Format

**CRITICAL: You MUST train the model to use the feedback tool before RL!**

```bash
# Convert Self-InPerfect data to tool-calling format
python3 recipe/feedback_rl/convert_to_tool_sft.py

# Creates: data/feedback_sft_data.parquet (428 examples)
# - 374 correct first try (no feedback needed)
# - 41 one-feedback corrections
# - 13 two-feedback corrections
```

The SFT data teaches the model:
- ✅ When to call `get_feedback` tool (when wrong)
- ✅ How to incorporate feedback and revise
- ✅ When to STOP calling feedback (after correction)

### Step 2: SFT Training

```bash
# Train model on tool usage
bash recipe/feedback_rl/run_feedback_sft.sh

# Output: checkpoint/feedback-sft-qwen-2.5-32b-instruct/
```

This teaches the model the **pattern** of feedback tool usage. Without this, the model won't know it can call the tool!

### Step 3: RL Training

```bash
# Update run_feedback_rl.sh to use SFT checkpoint
# Change model_path to: checkpoint/feedback-sft-qwen-2.5-32b-instruct/global_step_XXX/huggingface

# Run RL training
bash recipe/feedback_rl/run_feedback_rl.sh
```

## Quick Start (if SFT already done)

1. **Test the feedback tool**:
```bash
python recipe/feedback_rl/test_feedback_tool.py
```

2. **Run RL training**:
```bash
bash recipe/feedback_rl/run_feedback_rl.sh
```

## How It Works

### During RL Training

1. **Initial Attempt**: Model tries to solve a problem
2. **Tool-Calling Decision**:
   - If model generates `tool_calls` → executes `get_feedback` → loops back to step 3
   - If model gives final answer (no `tool_calls`) → TERMINATED → compute reward
3. **Feedback Generation**: vLLM server generates feedback using Self-InPerfect prompts
4. **Revision**: Model revises answer based on feedback
5. **Iteration**: Steps 2-4 repeat until model stops calling tool or hits max_turns
6. **Reward**: +1 for correct final answer, -1 for incorrect
   - Small bonus for successfully using feedback
   - Small penalty for ignoring repeated feedback

### Tool-Calling Loop Mechanism

**The model decides whether to call the feedback tool!** This is different from always calling a tool.

- Agent loop checks each generation for `tool_calls` in the output
- If `tool_calls` present → state = PROCESSING_TOOLS → execute tool → back to GENERATING
- If no `tool_calls` → state = TERMINATED → done
- Hard limits: `max_assistant_turns=3`, `max_user_turns=3` prevent infinite loops

**Why SFT is essential:**
- Without SFT, the model doesn't know it CAN generate tool_calls
- SFT teaches the pattern: wrong → call tool → receive feedback → correct → DON'T call tool again
- This prevents infinite feedback loops during RL training

## Configuration

Edit `run_feedback_rl.sh` to change:
- Dataset (`math`, `mmlu`, `trivia_qa`, etc.)
- Batch size, learning rate
- Max turns for feedback
- Model checkpoint path

Edit `feedback_tool_config.yaml` to change:
- vLLM server URL
- Feedback model
- Temperature, max_tokens (currently matches Self-InPerfect exactly)

## Key Parameters (matching Self-InPerfect)

- **temperature**: 0.0 (deterministic feedback)
- **max_tokens**: 2000
- **seed**: 14
- **stop_token_ids**: [128001, 128009]