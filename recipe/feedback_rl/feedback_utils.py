#!/usr/bin/env python3
"""
Utility functions copied from Self-InPerfect for feedback RL training
Self-contained version to avoid import errors
"""

import re
import datasets
import random

random.seed(42)

# Global cache for MATH train/holdout splits to ensure consistency
_MATH_SPLIT_CACHE = {}


def get_math_easy_split(split="train", train_ratio=0.5, seed=42):
    """
    Get MATH train/holdout split with consistent Level 1-2 problems.

    Args:
        split: "train" or "holdout"
        train_ratio: Proportion of data for training (default 0.5)
        seed: Random seed for splitting

    Returns:
        List of MATH problems
    """
    cache_key = f"{split}_{train_ratio}_{seed}"

    if cache_key in _MATH_SPLIT_CACHE:
        return _MATH_SPLIT_CACHE[cache_key]

    # Load and filter easy problems
    ds = datasets.load_dataset("lighteval/MATH", trust_remote_code=True)
    data_list = list(ds['test'])
    easy_problems = [item for item in data_list if item['level'] in ['Level 1', 'Level 2']]
    print(f"Filtered to {len(easy_problems)} easy problems (Level 1-2) from {len(data_list)} total")

    # Deterministic split
    rng = random.Random(seed)
    shuffled = easy_problems.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_split = shuffled[:split_idx]
    holdout_split = shuffled[split_idx:]

    print(f"MATH split: {len(train_split)} train, {len(holdout_split)} holdout")

    # Cache both splits
    _MATH_SPLIT_CACHE[f"train_{train_ratio}_{seed}"] = train_split
    _MATH_SPLIT_CACHE[f"holdout_{train_ratio}_{seed}"] = holdout_split

    return train_split if split == "train" else holdout_split


def setup_datalist(dataset_name, mode="test", random_choice=False):
    """Load datasets - copied from Self-InPerfect utils.py

    Special dataset names:
    - "math_train_train": Full training split of Level 1-2 MATH problems (for training)
    - "math_train_test": Sampled 1/5 of training split of Level 1-2 problems (for validation)
    - "math_train": Training split of easy MATH problems (legacy)
    - "math_holdout": Holdout split of easy MATH problems (same difficulty, different problems)
    - "math": Full MATH dataset or sampled version
    """
    # Handle special MATH splits
    if dataset_name == "math_train_train":
        # Full training set, no sampling
        data = get_math_easy_split(split="train", train_ratio=0.5, seed=42)
        print(f"Loaded {len(data)} math_train_train problems (full Level 1-2 training set)")
        return data
    elif dataset_name == "math_train_test":
        # Sampled 1/5 for validation
        data = get_math_easy_split(split="train", train_ratio=0.5, seed=42)
        random.seed(14)
        sampled = random.sample(data, len(data) // 5)
        print(f"Sampled {len(sampled)} math_train_test problems from {len(data)} Level 1-2 (for validation)")
        return sampled
    elif dataset_name == "math_holdout_test":
        # Sampled 1/5 of holdout for validation
        data = get_math_easy_split(split="holdout", train_ratio=0.5, seed=42)
        random.seed(14)
        sampled = random.sample(data, len(data) // 5)
        print(f"Sampled {len(sampled)} math_holdout_test problems from {len(data)} Level 1-2 (for validation)")
        return sampled
    elif dataset_name == "math_train":
        data = get_math_easy_split(split="train", train_ratio=0.5, seed=42)
        if random_choice:
            # Sample 1/5 for faster training
            random.seed(14)
            sampled = random.sample(data, len(data) // 5)
            print(f"Sampled {len(sampled)} math_train problems from {len(data)} for faster training")
            return sampled
        return data
    elif dataset_name == "math_holdout":
        data = get_math_easy_split(split="holdout", train_ratio=0.5, seed=42)
        if random_choice:
            # Sample 1/5 for faster validation
            random.seed(14)
            sampled = random.sample(data, len(data) // 5)
            print(f"Sampled {len(sampled)} math_holdout problems from {len(data)} for faster validation")
            return sampled
        return data
    elif dataset_name == "math":
        # Load full MATH dataset (~7500 test examples)
        ds = datasets.load_dataset("lighteval/MATH", trust_remote_code=True)
        data_list = list(ds['test'])
        if mode == "test":
            if random_choice:
                # Filter for easy problems (Level 1 and Level 2 only)
                easy_problems = [item for item in data_list if item['level'] in ['Level 1', 'Level 2']]
                print(f"Filtered to {len(easy_problems)} easy problems (Level 1-2) from {len(data_list)} total")
                # Sample 1/10 of the easy problems
                random.seed(14)
                sampled = random.sample(easy_problems, len(easy_problems) // 10)
                print(f"Sampled {len(sampled)} problems for debugging")
                return sampled
            else:
                return data_list
    elif dataset_name == "trivia_qa":
        ds = datasets.load_dataset("mandarjoshi/trivia_qa", 'rc.wikipedia.nocontext')
        data_list = list(ds['validation'])
        if mode == "test":
            if random_choice:
                random.seed(14)
                return random.sample(data_list, len(data_list) // 10)
            else:
                return data_list
    elif dataset_name == "mmlu":
        ds = datasets.load_dataset("cais/mmlu", "all")
        data_list = list(ds['test'])
        if mode == "test":
            if random_choice:
                random.seed(14)
                return random.sample(data_list, len(data_list) // 10)
            else:
                return data_list
    elif dataset_name == "mmlu_pro_train":
        # Full validation split for training (no sampling)
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        data_list = list(ds['validation'])
        print(f"Loaded {len(data_list)} mmlu_pro_train problems (full validation set for training)")
        return data_list
    elif dataset_name == "mmlu_pro_test":
        # Small sample from test split for evaluation
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        data_list = list(ds['test'])
        random.seed(14)
        sampled = random.sample(data_list, min(50, len(data_list)))
        print(f"Loaded {len(sampled)} mmlu_pro_test problems from {len(data_list)} test set (sampled 50 for validation)")
        return sampled
    elif dataset_name == "mmlu_pro":
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        data_list = list(ds['test'])
        if mode == "test":
            if random_choice:
                random.seed(14)
                return random.sample(data_list, len(data_list) // 10)
            else:
                return data_list
    elif dataset_name == "gpqa":
        original_dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        formatted_dataset = datasets.load_dataset("jeggers/gpqa_formatted", 'diamond')
        original_data_list = list(original_dataset['train'])
        formatted_data_list = list(formatted_dataset['train'])
        for item in original_data_list:
            item['question'] = item.pop('Question', None)
        for item in formatted_data_list:
            item['question'] = item.pop('Question', None)
        original_mapping = {item['question']: item.get('Explanation', None) for item in original_data_list}
        for entry in formatted_data_list:
            entry_id = entry['question']
            if entry_id in original_mapping:
                entry['Explanation'] = original_mapping[entry_id]
        # Sample 100 for validation to match other dataset sizes
        random.seed(14)
        sampled = random.sample(formatted_data_list, min(100, len(formatted_data_list)))
        print(f"Loaded {len(sampled)} gpqa problems from {len(formatted_data_list)} (sampled 100 for validation)")
        return sampled
    elif dataset_name == "gsm8k":
        ds = datasets.load_dataset("gsm8k", 'main')
        data_list = list(ds['test'])
        if mode == "test":
            return data_list
    elif dataset_name == "aime_2024":
        ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")
        data_list = list(ds['train'])
        if mode == "test":
            return data_list
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_key(dataset_name):
    """Get the key for extracting questions - copied from Self-InPerfect"""
    if dataset_name in ["arc", "ecqa", "gsm8k", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test",
                        "gsm8k_symbolic", "mmlu", "trivia_qa", "gpqa"]:
        return "question"
    elif dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                          "math_holdout", "math_holdout_test"]:
        return "problem"
    elif dataset_name == "aime_2024":
        return "Problem"
    else:
        return "question"


def last_boxed_only_string(string: str):
    """Extract last boxed answer - from dataset_specific_utils.py"""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval


def remove_boxed(s: str):
    """Remove boxed formatting - from dataset_specific_utils.py"""
    if s is None:
        return None
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def strip_string(string):
    """Strip string for comparison - from dataset_specific_utils.py"""
    string = str(string).strip()
    # Remove unit
    string = string.replace("%", "")
    string = string.replace("$", "")
    string = string.replace("\\$", "")
    # Remove comma
    string = string.replace(",", "")
    return string


def normalize_final_answer(final_answer: str) -> str:
    """Normalize answer for comparison - from dataset_specific_utils.py"""
    final_answer = final_answer.split("=")[-1]

    SUBSTITUTIONS = [
        ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
        (r"\ ", ""), (" ", ""), ("mbox", "text"),
        (",\\text{and}", ","), ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]

    REMOVED_EXPRESSIONS = [
        "square", "ways", "integers", "dollars", "mph", "inches",
        "hours", "km", "units", "\\ldots", "sue", "points",
        "feet", "minutes", "digits", "cents", "degrees", "cm",
        "gm", "pounds", "meters", "meals", "edges", "students",
        "childrentickets", "multiples", "\\text{s}", "\\text{.}",
        "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
        "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}",
        r"\;", r",\!", "{,}", '"', "\\dots",
    ]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def get_normalized_answer(dataset_name, data):
    """Extract normalized answer - copied from Self-InPerfect"""
    if dataset_name == "gsm8k":
        return data['answer'].split("####")[1].strip()
    elif dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                          "math_holdout", "math_holdout_test"]:
        solution = data['solution']
        res = remove_boxed(last_boxed_only_string(solution))
        try:
            res = strip_string(res)
        except:
            pass
        return res if res else ""
    elif dataset_name == "trivia_qa":
        return data['answer']["normalized_value"]
    elif dataset_name == "mmlu":
        number = data['answer']
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        return index_to_letter[number]
    elif dataset_name in ["mmlu_pro", "mmlu_pro_train", "mmlu_pro_test"]:
        return data['answer']
    elif dataset_name == "gpqa":
        answer = data['answer']
        # Handle both integer indices (0-4) and string answers
        if isinstance(answer, int):
            index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
            return index_to_letter.get(answer, str(answer))
        else:
            # String answer - apply strip_string to clean \text{} wrapper
            try:
                cleaned = strip_string(str(answer))
                return cleaned
            except:
                return str(answer).strip()
    elif dataset_name == "aime_2024":
        return str(data['Answer'])
    else:
        return str(data.get('answer', ''))


def get_process_answer(dataset_name, data):
    """Get process answer/solution - copied from Self-InPerfect"""
    if dataset_name == "gsm8k":
        return data['answer']
    elif dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                          "math_holdout", "math_holdout_test"]:
        return data["solution"]
    elif dataset_name == "aime_2024":
        return data['Solution']
    elif dataset_name == "trivia_qa":
        return data['answer']["normalized_value"]
    elif dataset_name == "mmlu":
        number = data['answer']
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        return index_to_letter[number]
    elif dataset_name in ["mmlu_pro", "mmlu_pro_train", "mmlu_pro_test"]:
        return data['answer']
    elif dataset_name == "gpqa":
        number = data['answer']
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        ans = index_to_letter[number]
        explanation = data.get('Explanation', '')
        return f"The answer is: {ans}. The explanation is: {explanation}"
    else:
        return str(data.get('answer', ''))


def get_normalized_prediction(dataset_name, prediction):
    """Extract normalized prediction - copied from Self-InPerfect"""
    if dataset_name == "gsm8k":
        # Extract number from prediction
        match = re.search(r'-?\d+\.?\d*', prediction)
        return match.group() if match else ""
    elif dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                          "math_holdout", "math_holdout_test", "aime_2024", "gpqa"]:
        # GPQA now uses \boxed{} format like MATH (e.g., \boxed{A})
        res = remove_boxed(last_boxed_only_string(prediction))
        try:
            res = strip_string(res) if res else ""
        except:
            res = ""
        return res
    elif dataset_name == "trivia_qa":
        # Simple normalization
        return prediction.strip().lower()
    elif dataset_name in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test"]:
        # Extract letter answer
        pattern = re.compile(
            r"(?i)"
            r"(?:answer\s*(?:is\s*:|is|:)?)\s*"
            r"(?:"
            r"\(\(\s*([A-J])\s*\)\)"
            r"|"
            r"\(\s*([A-J])\s*\)"
            r"|"
            r"([A-J])"
            r")"
        )
        matches = pattern.findall(prediction)
        if matches:
            for match in matches:
                answer = next((m for m in match if m), None)
                if answer:
                    return answer.upper()
        return ""
    else:
        return prediction.strip()


def is_equivalent(pred_answer, ground_truth, dataset_name):
    """Check if answers are equivalent - simplified from Self-InPerfect"""
    if not pred_answer or not ground_truth:
        return False

    if dataset_name in ["math", "math_train", "math_holdout", "aime_2024"]:
        try:
            a = normalize_final_answer(pred_answer)
            b = normalize_final_answer(ground_truth)
            return a.strip() == b.strip()
        except:
            return False
    elif dataset_name in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test", "gpqa"]:
        # For multiple choice, just compare letters case-insensitively
        # GPQA now outputs \boxed{A}, which gets extracted to "A"
        return pred_answer.strip().upper() == ground_truth.strip().upper()
    elif dataset_name == "trivia_qa":
        return pred_answer.strip().lower() == ground_truth.strip().lower()
    else:
        return pred_answer.strip() == ground_truth.strip()


# ========== Answer Masking Functions ==========
# Copied from Self-InPerfect to prevent answer leakage in feedback

def mask_answer_in_string_math(input_string, ground_truth):
    """Mask answer in MATH/AIME feedback - masks both \\boxed{answer} and standalone answer."""
    ground_truth_str = str(ground_truth)
    safe_ground_truth = re.escape(ground_truth_str)

    # First: mask the entire \boxed{ground_truth}
    masked_string = re.sub(rf'\\boxed\s*\{{\s*{safe_ground_truth}\s*\}}', '[masked]', input_string)

    # Then: mask standalone ground_truth
    masked_string = re.sub(rf'\b{safe_ground_truth}\b', '[masked]', masked_string)

    return masked_string


def mask_answers_in_trivia_qa(feedback_string, data):
    """
    Mask all answers (aliases and normalized_aliases) in the feedback string.
    Uses word boundaries to ensure only complete words/phrases are masked.
    """
    all_answers = data['answer']['aliases'] + data['answer']['normalized_aliases']
    all_answers = list(set(filter(None, map(str.strip, all_answers))))  # Remove empty strings
    all_answers.sort(key=len, reverse=True)  # Process longer phrases first

    masked_feedback = feedback_string
    for answer in all_answers:
        safe_answer = re.escape(answer)
        pattern = rf'\b{safe_answer}\b'
        masked_feedback = re.sub(pattern, '[masked]', masked_feedback, flags=re.IGNORECASE)

    return masked_feedback


def mask_answer_in_string_mcq_case_sensitive(input_string, ground_truth_letter):
    """Mask the correct answer letter in MCQ feedback (MMLU, GPQA)."""
    # Only allow uppercase A–E or similar; reject lowercase inputs
    if not ground_truth_letter.isupper():
        ground_truth_letter = ground_truth_letter.upper()

    safe_gt = re.escape(ground_truth_letter)

    # Mask patterns like "The answer is (A)" or "answer is A" or "(A)"
    masked = re.sub(
        rf'\b{safe_gt}\b|\({safe_gt}\)',
        '[masked]',
        input_string,
        flags=re.IGNORECASE
    )

    return masked