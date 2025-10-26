#!/usr/bin/env python3
"""
Convert Self-InPerfect feedback data to tool-calling SFT format.
Only includes examples that succeed within 2 feedback rounds (iteration < 3).
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from omegaconf import OmegaConf


def load_tool_schema(config_path: str) -> Dict:
    """Load the get_feedback tool schema from config."""
    config = OmegaConf.load(config_path)
    tool_schema = OmegaConf.to_container(config["tools"][0]["tool_schema"])
    return tool_schema


def create_correct_first_try(problem: str, response: str, tool_schema: Dict) -> Dict:
    """
    Example where model gets it right on first try - NO tool call.
    Pattern: User → Assistant(answer, no tool_call) → DONE
    """
    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": response}  # No tool_calls!
    ]
    return {"messages": messages, "tools": [tool_schema]}


def create_one_feedback_correction(
    problem: str,
    initial_response: str,
    feedback: str,
    corrected_response: str,
    tool_schema: Dict
) -> Dict:
    """
    Example where model uses feedback once and corrects.
    Pattern: User → Assistant(wrong + tool_call) → Tool(feedback) → Assistant(corrected, no tool_call) → DONE
    """
    messages = [
        {"role": "user", "content": problem},

        # First attempt WITH tool call to request feedback
        {
            "role": "assistant",
            "content": initial_response,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_feedback",
                    "arguments": {"current_answer": initial_response, "reasoning": ""}
                }
            }]
        },

        # Tool provides feedback
        {"role": "tool", "content": feedback},

        # Corrected answer WITHOUT tool call (done)
        {"role": "assistant", "content": corrected_response}
    ]
    return {"messages": messages, "tools": [tool_schema]}


def create_two_feedback_correction(
    problem: str,
    attempt0: str, feedback0: str,
    attempt1: str, feedback1: str,
    attempt2: str,
    tool_schema: Dict
) -> Dict:
    """
    Example where model uses feedback twice before getting it right.
    Pattern: User → Assistant(wrong + tool) → Tool → Assistant(still wrong + tool) → Tool → Assistant(correct, no tool) → DONE
    """
    messages = [
        {"role": "user", "content": problem},

        # First attempt WITH tool call
        {
            "role": "assistant",
            "content": attempt0,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_feedback",
                    "arguments": {"current_answer": attempt0, "reasoning": ""}
                }
            }]
        },

        # First feedback
        {"role": "tool", "content": feedback0},

        # Second attempt WITH tool call (still not right)
        {
            "role": "assistant",
            "content": attempt1,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_feedback",
                    "arguments": {"current_answer": attempt1, "reasoning": ""}
                }
            }]
        },

        # Second feedback
        {"role": "tool", "content": feedback1},

        # Final corrected answer WITHOUT tool call (done)
        {"role": "assistant", "content": attempt2}
    ]
    return {"messages": messages, "tools": [tool_schema]}


def process_self_inperfect_data(
    input_file: str,
    output_file: str,
    tool_config_path: str
):
    """
    Convert Self-InPerfect data to tool-calling SFT format.
    Groups examples by unique_id and tracks iterations.
    """
    tool_schema = load_tool_schema(tool_config_path)

    # Group examples by unique_id
    examples_by_id = defaultdict(list)

    print("Loading and grouping examples...")
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            unique_id = data.get("unique_id", "")
            iteration = data.get("iteration", 0)

            # Only process examples with iteration < 3
            if iteration < 3:
                examples_by_id[unique_id].append(data)

    print(f"Found {len(examples_by_id)} unique problems")

    # Process and collect SFT examples
    stats = {
        "correct_first_try": 0,
        "one_feedback": 0,
        "two_feedback": 0,
        "skipped": 0
    }

    sft_examples = []

    # Iterate through all unique problems
    for unique_id, iterations in examples_by_id.items():
        # Sort by iteration
        iterations.sort(key=lambda x: x.get("iteration", 0))

        # Check patterns
        if len(iterations) == 1 and iterations[0]["iteration"] == 0:
            # Single attempt
            data = iterations[0]
            if data.get("is_correct", False):
                # Correct on first try - no feedback needed
                problem = data.get("original_question", data["problem"])
                response = data["full_response"][0] if data.get("full_response") else ""

                sft_example = create_correct_first_try(problem, response, tool_schema)
                sft_examples.append(sft_example)
                stats["correct_first_try"] += 1
            else:
                # Wrong on first try but no subsequent attempts
                stats["skipped"] += 1

        elif len(iterations) >= 2:
            # Multiple iterations - check if eventually correct
            iter0 = iterations[0]
            iter1 = iterations[1]

            # One feedback case: iteration 0 wrong, iteration 1 correct
            if (iter0["iteration"] == 0 and not iter0.get("is_correct", False) and
                iter1["iteration"] == 1 and iter1.get("is_correct", False)):

                # Use original_question to get clean problem without history
                problem = iter0.get("original_question", iter0["problem"])
                initial_response = iter0["full_response"][0] if iter0.get("full_response") else ""
                feedback = iter0.get("feedback", [""])[0] if iter0.get("feedback") else ""
                corrected_response = iter1["full_response"][0] if iter1.get("full_response") else ""

                if feedback:  # Only if we have feedback
                    sft_example = create_one_feedback_correction(
                        problem, initial_response, feedback, corrected_response, tool_schema
                    )
                    sft_examples.append(sft_example)
                    stats["one_feedback"] += 1
                else:
                    stats["skipped"] += 1

            # Two feedback case
            elif len(iterations) >= 3:
                iter2 = iterations[2]
                if (iter0["iteration"] == 0 and not iter0.get("is_correct", False) and
                    iter1["iteration"] == 1 and not iter1.get("is_correct", False) and
                    iter2["iteration"] == 2 and iter2.get("is_correct", False)):

                    # Use original_question to get clean problem without history
                    problem = iter0.get("original_question", iter0["problem"])
                    attempt0 = iter0["full_response"][0] if iter0.get("full_response") else ""
                    feedback0 = iter0.get("feedback", [""])[0] if iter0.get("feedback") else ""
                    attempt1 = iter1["full_response"][0] if iter1.get("full_response") else ""
                    feedback1 = iter1.get("feedback", [""])[0] if iter1.get("feedback") else ""
                    attempt2 = iter2["full_response"][0] if iter2.get("full_response") else ""

                    if feedback0 and feedback1:  # Only if we have both feedbacks
                        sft_example = create_two_feedback_correction(
                            problem, attempt0, feedback0, attempt1, feedback1, attempt2, tool_schema
                        )
                        sft_examples.append(sft_example)
                        stats["two_feedback"] += 1
                    else:
                        stats["skipped"] += 1
                else:
                    stats["skipped"] += 1
            else:
                stats["skipped"] += 1
        else:
            stats["skipped"] += 1

    # Convert to pandas DataFrame and save as parquet
    print(f"\nConverting {len(sft_examples)} examples to parquet format...")
    df = pd.DataFrame(sft_examples)
    df.to_parquet(output_file, index=False)

    print("\n=== Conversion Complete ===")
    print(f"Total SFT examples written: {sum(v for k, v in stats.items() if k != 'skipped')}")
    print(f"  - Correct first try (no feedback): {stats['correct_first_try']}")
    print(f"  - One feedback correction: {stats['one_feedback']}")
    print(f"  - Two feedback corrections: {stats['two_feedback']}")
    print(f"  - Skipped (incomplete): {stats['skipped']}")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    input_file = "/scratch/dkhasha1/djiang21/Self-InPerfect_new/llama3.3_math_4.1_mini_feedback.jsonl"
    output_file = "/weka/scratch/dkhasha1/djiang21/self-inperfect-verl/verl_code_base/data/feedback_sft_data.parquet"
    tool_config_path = "/weka/scratch/dkhasha1/djiang21/self-inperfect-verl/verl_code_base/recipe/feedback_rl/feedback_tool_config.yaml"

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Convert data
    process_self_inperfect_data(
        input_file=input_file,
        output_file=output_file,
        tool_config_path=tool_config_path
    )
