#!/usr/bin/env python3
"""
Compute detailed feedback usage metrics from validation logs and log to wandb.
This should be called after each validation run.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, metrics will only be printed")


def extract_boxed_answer(text):
    """Extract answer from \boxed{} format."""
    matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def categorize_example(item):
    """
    Categorize an example into one of these buckets:
    - correct_no_feedback: Got it right, didn't use feedback
    - correct_with_feedback: Got it right, used feedback anyway (reward gaming)
    - wrong_corrected: Started wrong, feedback helped correct it
    - wrong_still_wrong: Started wrong, feedback didn't help
    - wrong_no_feedback: Got it wrong, didn't use feedback
    - no_answer: Couldn't extract answer
    """
    ground_truth = item.get('gts', '')
    output = item.get('output', '')

    # Extract all answers
    all_answers = []
    segments = output.split('assistant\n')
    for segment in segments:
        answer = extract_boxed_answer(segment)
        if answer:
            all_answers.append(answer)

    if not all_answers:
        return 'no_answer'

    # Get first and final answers
    first_answer = all_answers[0]
    final_answer = all_answers[-1]

    # Count feedback calls
    num_feedback = output.count('<tool_call>')

    # Normalize for comparison (remove \text{} wrapper, trim whitespace)
    def normalize(s):
        s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
        return s.strip().lower()

    first_correct = normalize(first_answer) == normalize(ground_truth)
    final_correct = normalize(final_answer) == normalize(ground_truth)

    # Categorize
    if num_feedback == 0:
        if first_correct:
            return 'correct_no_feedback'
        else:
            return 'wrong_no_feedback'
    else:
        if first_correct and final_correct:
            return 'correct_with_feedback'  # Reward gaming
        elif not first_correct and final_correct:
            return 'wrong_corrected'  # Actual learning!
        elif not first_correct and not final_correct:
            return 'wrong_still_wrong'
        else:
            return 'correct_then_wrong'  # Rare case


def compute_metrics_for_file(jsonl_file):
    """Compute metrics for a single validation file."""
    # Overall metrics
    overall_counts = defaultdict(int)

    # Per-dataset metrics (if we can identify dataset from question format)
    dataset_counts = defaultdict(lambda: defaultdict(int))

    with open(jsonl_file) as f:
        for line in f:
            item = json.loads(line)
            category = categorize_example(item)

            overall_counts[category] += 1

            # Try to identify dataset
            # This is heuristic - you might want to add a 'dataset' field to logs
            input_text = item.get('input', '')
            if 'AIME' in input_text or 'AMC' in input_text:
                dataset = 'aime_2024'
            elif '(A)' in input_text and '(B)' in input_text and '(C)' in input_text and '(D)' in input_text:
                dataset = 'gpqa'
            else:
                dataset = 'math'  # Default to MATH

            dataset_counts[dataset][category] += 1

    return overall_counts, dataset_counts


def compute_wandb_metrics(counts):
    """Convert counts to wandb metrics."""
    total = sum(counts.values())
    if total == 0:
        return {}

    metrics = {
        # Accuracy metrics
        'accuracy': 100 * (counts['correct_no_feedback'] + counts['correct_with_feedback'] + counts['wrong_corrected']) / total,
        'accuracy_first_try': 100 * (counts['correct_no_feedback'] + counts['correct_with_feedback']) / total,

        # Feedback usage metrics
        'feedback_usage_rate': 100 * (counts['correct_with_feedback'] + counts['wrong_corrected'] + counts['wrong_still_wrong']) / total,

        # Feedback effectiveness
        'feedback_success_rate': 100 * counts['wrong_corrected'] / (counts['wrong_corrected'] + counts['wrong_still_wrong']) if (counts['wrong_corrected'] + counts['wrong_still_wrong']) > 0 else 0,

        # Category breakdown (percentages)
        'pct_correct_no_feedback': 100 * counts['correct_no_feedback'] / total,
        'pct_correct_with_feedback': 100 * counts['correct_with_feedback'] / total,
        'pct_wrong_corrected': 100 * counts['wrong_corrected'] / total,
        'pct_wrong_still_wrong': 100 * counts['wrong_still_wrong'] / total,
        'pct_wrong_no_feedback': 100 * counts['wrong_no_feedback'] / total,

        # Category breakdown (counts)
        'count_correct_no_feedback': counts['correct_no_feedback'],
        'count_correct_with_feedback': counts['correct_with_feedback'],
        'count_wrong_corrected': counts['wrong_corrected'],
        'count_wrong_still_wrong': counts['wrong_still_wrong'],
        'count_wrong_no_feedback': counts['wrong_no_feedback'],
        'count_total': total,
    }

    return metrics


def main(validation_dir, step=None, log_to_wandb=True):
    """
    Main function to compute and log metrics.

    Args:
        validation_dir: Path to validation directory containing .jsonl files
        step: Training step number (for wandb logging)
        log_to_wandb: Whether to log to wandb
    """
    validation_dir = Path(validation_dir)

    # Find validation files
    if step is not None:
        # Specific step
        jsonl_file = validation_dir / f"{step}.jsonl"
        if not jsonl_file.exists():
            print(f"Warning: {jsonl_file} not found")
            return
        files_to_process = [jsonl_file]
    else:
        # All validation files
        files_to_process = sorted(validation_dir.glob("*.jsonl"))

    for jsonl_file in files_to_process:
        step_num = int(jsonl_file.stem)
        print(f"Processing step {step_num}...")

        overall_counts, dataset_counts = compute_metrics_for_file(jsonl_file)

        # Compute overall metrics
        overall_metrics = compute_wandb_metrics(overall_counts)

        # Compute per-dataset metrics
        dataset_metrics = {}
        for dataset, counts in dataset_counts.items():
            dataset_metrics[dataset] = compute_wandb_metrics(counts)

        # Print summary
        print(f"\nStep {step_num} - Overall:")
        print(f"  Accuracy: {overall_metrics['accuracy']:.1f}%")
        print(f"  Feedback usage: {overall_metrics['feedback_usage_rate']:.1f}%")
        print(f"  Feedback success: {overall_metrics['feedback_success_rate']:.1f}%")
        print(f"  Wrong→Corrected: {overall_counts['wrong_corrected']} ({overall_metrics['pct_wrong_corrected']:.1f}%)")

        for dataset, metrics in dataset_metrics.items():
            print(f"\nStep {step_num} - {dataset}:")
            print(f"  Accuracy: {metrics['accuracy']:.1f}%")
            print(f"  Feedback success: {metrics['feedback_success_rate']:.1f}%")

        # Log to wandb
        if log_to_wandb and WANDB_AVAILABLE:
            wandb_log = {
                # Overall metrics
                'val/overall/accuracy': overall_metrics['accuracy'],
                'val/overall/accuracy_first_try': overall_metrics['accuracy_first_try'],
                'val/overall/feedback_usage_rate': overall_metrics['feedback_usage_rate'],
                'val/overall/feedback_success_rate': overall_metrics['feedback_success_rate'],
                'val/overall/pct_wrong_corrected': overall_metrics['pct_wrong_corrected'],
                'val/overall/pct_correct_with_feedback': overall_metrics['pct_correct_with_feedback'],
                'val/overall/pct_wrong_still_wrong': overall_metrics['pct_wrong_still_wrong'],
            }

            # Per-dataset metrics
            for dataset, metrics in dataset_metrics.items():
                prefix = f'val/{dataset}'
                wandb_log.update({
                    f'{prefix}/accuracy': metrics['accuracy'],
                    f'{prefix}/feedback_usage_rate': metrics['feedback_usage_rate'],
                    f'{prefix}/feedback_success_rate': metrics['feedback_success_rate'],
                    f'{prefix}/pct_wrong_corrected': metrics['pct_wrong_corrected'],
                })

            wandb.log(wandb_log, step=step_num)
            print(f"\nLogged to wandb for step {step_num}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compute_feedback_metrics.py <validation_dir> [step]")
        print("Example: python compute_feedback_metrics.py logs/my_run/validation/ 240")
        sys.exit(1)

    validation_dir = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Initialize wandb if needed
    if WANDB_AVAILABLE and wandb.run is None:
        print("Warning: No active wandb run. Metrics will only be printed, not logged.")
        log_to_wandb = False
    elif WANDB_AVAILABLE:
        log_to_wandb = True
    else:
        log_to_wandb = False

    main(validation_dir, step, log_to_wandb)
