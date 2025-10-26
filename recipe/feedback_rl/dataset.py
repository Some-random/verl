#!/usr/bin/env python3
"""
Dataset class for feedback RL training
Adapts Self-InPerfect datasets to VERL format
"""

import datasets
import json
from typing import Dict, Any
import copy

from verl.utils.dataset import RLHFDataset
from recipe.feedback_rl.feedback_utils import setup_datalist, get_dataset_key, get_normalized_answer

# Import answer format from Self-InPerfect
answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""

# System instruction for using feedback
feedback_system_instruction = """You are a helpful assistant with access to a feedback tool called 'get_feedback'.

When you use the get_feedback tool and receive feedback:
1. Carefully read and understand the feedback
2. Re-solve the ENTIRE problem from scratch incorporating the feedback's suggestions
3. Show ALL your reasoning steps again (do NOT just output a new answer)
4. If the feedback points out errors, fix them in your new solution
5. Present your revised solution in a complete, step-by-step format

The feedback is designed to help you improve your reasoning, not to give you the answer directly."""


class FeedbackRLDataset(RLHFDataset):
    """Dataset class for feedback RL training using Self-InPerfect datasets."""

    def _read_files_and_tokenize(self):
        """Load and process datasets from Self-InPerfect."""
        dataframes = []

        for data_file in self.data_files:
            # Extract dataset name from file path or use direct dataset name
            data_file_lower = data_file.lower()

            # Check for exact dataset names first (supports math_train, math_holdout, etc.)
            if data_file in ["math_train_train", "math_train_test", "math_train", "math_holdout",
                            "math_holdout_test", "aime_2024", "gpqa", "mmlu_pro", "mmlu_pro_train",
                            "mmlu_pro_test", "mmlu", "trivia_qa", "gsm8k", "math"]:
                dataset_name = data_file
            # Then check substrings in file path
            elif "aime" in data_file_lower:
                dataset_name = "aime_2024"
            elif "math_train" in data_file_lower:
                dataset_name = "math_train"
            elif "math_holdout" in data_file_lower:
                dataset_name = "math_holdout"
            elif "math" in data_file_lower:
                dataset_name = "math"
            elif "mmlu_pro" in data_file_lower:
                dataset_name = "mmlu_pro"
            elif "mmlu" in data_file_lower:
                dataset_name = "mmlu"
            elif "trivia" in data_file_lower:
                dataset_name = "trivia_qa"
            elif "gpqa" in data_file_lower:
                dataset_name = "gpqa"
            elif "gsm8k" in data_file_lower:
                dataset_name = "gsm8k"
            else:
                dataset_name = "math"  # default

            # Load dataset using Self-InPerfect utilities
            data_list = setup_datalist(dataset_name, mode="test", random_choice=True)
            print(f"Loaded {len(data_list)} examples from {dataset_name} dataset")

            # Convert to VERL format
            processed_data = []
            for item in data_list:
                processed_item = self.convert_to_verl_format(item, dataset_name)
                if processed_item:
                    processed_data.append(processed_item)

            # Convert to HuggingFace dataset
            dataframe = datasets.Dataset.from_list(processed_data)
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"Feedback RL dataset length: {len(self.dataframe)}")

    def convert_to_verl_format(self, item: Dict, dataset_name: str) -> Dict[str, Any]:
        """Convert Self-InPerfect dataset item to VERL format."""

        # Extract question and normalized ground truth based on dataset type
        if dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                           "math_holdout", "math_holdout_test"]:
            question = item.get("problem", "")
            # Get normalized final answer for feedback prompt (e.g., "42" not full solution)
            ground_truth = get_normalized_answer(dataset_name, item)
        elif dataset_name in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test"]:
            # Format MCQ question
            question = item.get("question", "")
            choices = item.get("choices", [])
            if choices:
                choice_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                question = f"{question}\n\n{choice_text}"
            ground_truth = get_normalized_answer(dataset_name, item)
        elif dataset_name == "trivia_qa":
            question = item.get("question", "")
            ground_truth = get_normalized_answer(dataset_name, item)
        elif dataset_name == "gpqa":
            question = item.get("question", "")
            # GPQA uses "options" not "choices"
            choices = item.get("options", item.get("choices", []))
            if choices:
                choice_text = "\n".join([f"({chr(65+i)}) {choice.strip()}" for i, choice in enumerate(choices)])
                question = f"{question}\n\n{choice_text}"
            ground_truth = get_normalized_answer(dataset_name, item)
        elif dataset_name == "gsm8k":
            question = item.get("question", "")
            ground_truth = get_normalized_answer(dataset_name, item)
        elif dataset_name == "aime_2024":
            question = item.get("Problem", "")
            ground_truth = get_normalized_answer(dataset_name, item)
        else:
            return None

        if not question or not ground_truth:
            return None

        # Add answer format instruction based on dataset
        if dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                           "math_holdout", "math_holdout_test", "aime_2024"]:
            # Math problems use \boxed{} format for numerical/algebraic answers
            prompt_content = question + answer_format
        elif dataset_name in ["gpqa", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test"]:
            # Multiple choice - answer should be a single letter character
            mcq_format = """\nThe answer format must be: \\boxed{X} where X is a single letter character (A, B, C, D, E, F, G, H, I, or J)."""
            prompt_content = question + mcq_format
        elif dataset_name in ["mmlu"]:
            # MMLU still uses plain text format
            prompt_content = question + '\n\nPlease finish your answer with "The answer is (X)" where X is the correct letter choice.'
        else:
            prompt_content = question + answer_format

        # Convert item to JSON string to avoid schema conflicts when concatenating datasets
        # Different datasets (MATH, MMLU-Pro, etc.) have different field structures
        item_json = json.dumps(item)

        # VERL format with system instruction for feedback usage
        verl_item = {
            "prompt": [
                {"role": "system", "content": feedback_system_instruction},
                {"role": "user", "content": prompt_content}
            ],
            "reward_model": {"ground_truth": str(ground_truth)},
            "ability": dataset_name.upper(),
            "data_source": dataset_name,
            "agent_name": "tool_agent",
            # tools_kwargs MUST be inside extra_info for RLHFDataset.__getitem__ to extract it (line 365)
            "extra_info": {
                "tools_kwargs": {
                    "get_feedback": {
                        "create_kwargs": {
                            "ground_truth": str(ground_truth),
                            "question": prompt_content,
                            "data_json": item_json,  # Pass as JSON string to avoid schema conflicts
                            "dataset_name": dataset_name  # Pass dataset name for correct field access
                        }
                    }
                }
            },
            # Store original item for reference (as JSON string)
            "original_data": item_json,
            "dataset_type": dataset_name
        }

        return verl_item