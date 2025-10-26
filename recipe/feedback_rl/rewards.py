#!/usr/bin/env python3
"""
Reward function for feedback RL training
"""

import re

from recipe.feedback_rl.feedback_utils import (
    get_normalized_answer,
    get_normalized_prediction,
    is_equivalent,
    normalize_final_answer
)


def compute_feedback_reward(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> dict:
    """
    Compute reward for feedback RL training.

    Args:
        data_source: Dataset name (math, mmlu, etc.)
        solution_str: Model's final response
        ground_truth: Correct answer
        extra_info: Additional info including number of feedback turns

    Returns:
        Dict with score, accuracy, and prediction
    """

    # Extract the model's final answer
    pred_answer = get_normalized_prediction(data_source, solution_str)

    # Normalize ground truth
    if isinstance(ground_truth, dict):
        # Handle trivia_qa format
        norm_ground_truth = ground_truth.get("value", str(ground_truth))
    else:
        norm_ground_truth = str(ground_truth)

    # Check correctness using Self-InPerfect's logic
    try:
        is_correct = is_equivalent(pred_answer, norm_ground_truth, data_source)
    except:
        # Fallback to simple string comparison
        is_correct = pred_answer.strip().lower() == norm_ground_truth.strip().lower()

    # Base reward: +1 for correct, -1 for incorrect
    if is_correct:
        base_reward = 1.0
    else:
        base_reward = -1.0

    # Feedback utilization bonus/penalty
    # Use actual tool call count instead of conversation turns
    num_feedback_turns = extra_info.get("num_tool_calls", 0)

    if is_correct and num_feedback_turns > 0:
        # Small bonus for successfully using feedback to get correct answer
        feedback_bonus = min(0.1 * num_feedback_turns, 0.3)  # Cap at 0.3
        final_reward = base_reward + feedback_bonus
    elif not is_correct and num_feedback_turns > 0:
        # Encourage tool usage even when wrong (like ReTool)
        # This helps the model learn to seek feedback during training
        tool_usage_reward = (num_feedback_turns) * 0.15  # 0.15 per feedback turn
        final_reward = base_reward + tool_usage_reward  # e.g., -1.0 + 0.3 = -0.7 for 2 turns
        final_reward = max(final_reward, -0.6)  # Cap minimum at -0.6 (like ReTool)
    else:
        final_reward = base_reward

    # Ensure reward is in reasonable range
    final_reward = max(final_reward, -1.5)
    final_reward = min(final_reward, 1.5)

    return {
        "score": float(final_reward),
        "acc": bool(is_correct),  # Convert numpy bool_ to Python bool for JSON
        "pred": str(pred_answer),
        "feedback_turns": int(num_feedback_turns),
        "base_reward": float(base_reward)
    }