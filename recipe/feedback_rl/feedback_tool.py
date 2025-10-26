#!/usr/bin/env python3
"""
Feedback Tool for Multi-Turn RL Training
Reuses prompts and logic from Self-InPerfect codebase
"""

import logging
import aiohttp
import asyncio
import json
import random
from typing import Any, Optional, Dict
from uuid import uuid4

from verl.tools.base_tool import BaseTool, ToolResponse
from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

# Import local utilities
from recipe.feedback_rl.feedback_utils import (
    get_normalized_answer,
    get_normalized_prediction,
    get_process_answer,
    get_dataset_key,
    is_equivalent,
    mask_answer_in_string_math,
    mask_answers_in_trivia_qa,
    mask_answer_in_string_mcq_case_sensitive
)

logger = logging.getLogger(__name__)


class SelfInPerfectFeedbackTool(BaseTool):
    """
    Feedback tool that reuses Self-InPerfect prompt templates.
    Calls vLLM server to generate feedback without revealing answers.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # vLLM server configuration - support single URL or list of URLs for load balancing
        vllm_url_config = config.get("vllm_url", "http://c002:1233/v1/chat/completions")
        if isinstance(vllm_url_config, list):
            self.vllm_urls = vllm_url_config
        else:
            self.vllm_urls = [vllm_url_config]

        logger.info(f"Initialized with {len(self.vllm_urls)} vLLM server(s): {self.vllm_urls}")

        self.feedback_model = config.get("feedback_model", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
        self.temperature = config.get("temperature", 0.0)  # feedback_temp = 0.0
        self.max_tokens = config.get("max_tokens", 2000)  # max_tokens: 2000 from Self-InPerfect
        self.timeout = config.get("timeout", 30)

        # Additional Self-InPerfect parameters
        self.best_of = config.get("best_of", 1)
        self.n = config.get("n", 1)
        self.seed = config.get("seed", 14)

        # Dataset type for proper prompt formatting
        self.dataset = config.get("dataset", "math")  # math, mmlu, trivia_qa, etc.

        # Feedback settings from Self-InPerfect
        self.use_process_feedback = config.get("use_process_feedback", True)
        self.shuffle = config.get("shuffle", False)  # For MCQ shuffling

        logger.info(f"Initialized SelfInPerfectFeedbackTool for {self.dataset} dataset")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        question: Optional[str] = None,
        data: Optional[Dict] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """Initialize instance with problem data."""
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract from create_kwargs if passed that way
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            ground_truth = create_kwargs.get("ground_truth", ground_truth)
            question = create_kwargs.get("question", question)
            # Handle both old format (data dict) and new format (data_json string)
            if "data_json" in create_kwargs:
                data = json.loads(create_kwargs["data_json"])
            else:
                data = create_kwargs.get("data", data)
            dataset_name = create_kwargs.get("dataset_name", dataset_name)

        # Use per-instance dataset_name if provided, otherwise fall back to config default
        if dataset_name is None:
            dataset_name = self.dataset

        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "question": question,
            "data": data,  # Full data dict from dataset
            "dataset_name": dataset_name,  # Store dataset name per instance
            "history": "",
            "iteration": 0,
            "all_attempts": []
        }

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Generate feedback using Self-InPerfect prompt templates.
        If answer is already correct, skip vLLM call and return success message.

        Args:
            parameters: Dict with 'current_answer' and optionally 'reasoning'
        """
        print("[TOOL CALLED] Feedback tool executed")
        instance = self._instance_dict.get(instance_id, {})
        data = instance.get("data", {})
        ground_truth = instance.get("ground_truth", "")
        question = instance.get("question", "")
        history = instance.get("history", "")
        iteration = instance.get("iteration", 0)
        dataset_name = instance.get("dataset_name", self.dataset)  # Get dataset from instance

        current_answer = parameters.get("current_answer", "")
        reasoning = parameters.get("reasoning", "")

        # Check if answer is already correct - no need for feedback!
        pred_answer = get_normalized_prediction(dataset_name, current_answer)
        is_correct = is_equivalent(pred_answer, ground_truth, dataset_name)

        if is_correct:
            # Answer is correct - return success message instead of feedback
            feedback_text = "Your answer is correct! No further feedback needed."
            logger.info(f"Answer is correct at iteration {iteration}, skipping vLLM feedback generation")
        else:
            # Answer is wrong - generate feedback from vLLM
            logger.info(f"Answer is incorrect at iteration {iteration}, generating feedback from vLLM")

            # Build feedback message using Self-InPerfect templates
            feedback_message = self._create_feedback_message(
                dataset=dataset_name,
                question=question,
                current_answer=current_answer,
                ground_truth=ground_truth,
                history=history,
                iteration=iteration,
                data=data,
                reasoning=reasoning
            )

            # Call vLLM server for feedback generation
            feedback_text = await self._call_vllm_server(feedback_message)

            # Apply dataset-specific masking to prevent answer leakage
            if feedback_text and not feedback_text.startswith("Error:"):
                if dataset_name in ["math", "math_train", "math_train_train", "math_train_test",
                                   "math_holdout", "math_holdout_test", "aime_2024"]:
                    feedback_text = mask_answer_in_string_math(feedback_text, ground_truth)
                elif dataset_name == "trivia_qa":
                    if data:
                        feedback_text = mask_answers_in_trivia_qa(feedback_text, data)
                elif dataset_name in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test", "gpqa"]:
                    feedback_text = mask_answer_in_string_mcq_case_sensitive(feedback_text, ground_truth)

        # Update history for next iteration (Self-InPerfect format)
        instance["iteration"] += 1
        instance["all_attempts"].append(current_answer)

        # Format history like Self-InPerfect
        attempt_entry = f"\n\nAttempt at (iteration={iteration}):\n{current_answer}"
        if feedback_text:
            attempt_entry += f"\n\nFeedback at (iteration={iteration}):\n{feedback_text}"

        instance["history"] += attempt_entry

        # Append instruction to re-reason after feedback (unless answer is already correct)
        if not is_correct and feedback_text:
            feedback_with_instruction = (
                feedback_text +
                "\n\n" + "="*80 + "\n"
                "IMPORTANT: Please re-solve the problem from scratch incorporating the above feedback.\n"
                "Show ALL your reasoning steps again. Do NOT just output a new answer."
            )
        else:
            feedback_with_instruction = feedback_text

        return ToolResponse(text=feedback_with_instruction), None, None

    def _create_feedback_message(
        self,
        dataset: str,
        question: str,
        current_answer: str,
        ground_truth: str,
        history: str,
        iteration: int,
        data: Dict,
        reasoning: str = ""
    ) -> list:
        """
        Create feedback message using Self-InPerfect prompt templates.
        Directly adapted from openai_async_process.py
        """

        # Combine question with reasoning if available
        full_response = current_answer
        if reasoning:
            full_response = f"{reasoning}\n\n{current_answer}"

        # Different templates based on dataset and iteration
        if iteration == 0:
            # First attempt - no history
            if dataset in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test", "gpqa"]:
                content = (
                    "There was a mistake in answering the following question:\n\n"
                    + question
                    + "\n\nMost Recent Answer:\n" + full_response
                    + "\n\nThe correct final answer is: " + ground_truth
                )

                if self.use_process_feedback and data:
                    # Add process answer if available
                    process_answer = get_process_answer(dataset, data) if data else ""
                    if process_answer:
                        content += "\nThe correct reasoning process that leads to this answer is: " + process_answer

                content += (
                    "\n\nPlease provide feedback identifying which step(s) were incorrect or how to get to the correct answer "
                    "WITHOUT revealing the correct final answer or the content of the correct option."
                )
            else:
                # Non-MCQ datasets (math, trivia_qa, etc.)
                content = (
                    "There was a mistake in answering this question.\n\n"
                    + question
                    + "\n\nMost Recent Answer: " + full_response
                    + "\n\nThe correct final answer is: " + ground_truth
                )

                if self.use_process_feedback and data:
                    process_answer = get_process_answer(dataset, data) if data else ""
                    if process_answer:
                        content += "\nThe correct reasoning process that leads to this answer is: " + process_answer

                content += (
                    "\n\nPlease provide feedback identifying which step(s) were incorrect or how to get to the correct answer "
                    "**WITHOUT PROVIDING THE CORRECT FINAL ANSWER**:"
                )
        else:
            # Subsequent attempts - include history with clear structure
            base_content = (
                "There was a mistake in answering this question.\n\n"
                + "QUESTION:\n"
                + question
                + "\n\n" + "="*80 + "\n"
                + "You are provided with the full history of previous attempts made by a separate model, along with corresponding feedback.\n"
            )

            if self.shuffle and dataset in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test", "gpqa"]:
                base_content += "\nNote that the options in previous questions might have been switched in each different attempt.\n"

            base_content += (
                "\n" + "="*80 + "\n"
                + "PREVIOUS ATTEMPTS AND FEEDBACK:\n"
                + "="*80 + "\n\n"
                + history
                + "\n\n" + "="*80 + "\n"
                + "CURRENT ATTEMPT (iteration=" + str(iteration) + "):\n"
                + "="*80 + "\n"
                + full_response
                + "\n\n" + "="*80 + "\n"
                + "GROUND TRUTH:\n"
                + "The correct final answer is: " + ground_truth
            )

            if self.use_process_feedback and data:
                process_answer = get_process_answer(dataset, data) if data else ""
                if process_answer:
                    base_content += "\nThe correct reasoning process that leads to this answer is: " + process_answer

            if dataset in ["mmlu", "mmlu_pro", "mmlu_pro_train", "mmlu_pro_test", "gpqa"]:
                base_content += (
                    "\n\nPlease provide feedback identifying which step(s) were incorrect or how to get to the correct answer "
                    "WITHOUT revealing the correct final answer or the content of the correct option."
                )
            else:
                base_content += (
                    "\n\nPlease provide feedback identifying which step(s) were incorrect or how to get to the correct answer "
                    "**WITHOUT PROVIDING THE CORRECT FINAL ANSWER**:"
                )

            content = base_content

        # Return in chat format for vLLM
        return [{"role": "user", "content": content}]

    async def _call_vllm_server(self, messages: list) -> str:
        """Call vLLM server using exact Self-InPerfect parameters.

        If multiple vLLM URLs are configured, randomly select one for load balancing.
        If the selected server fails (500 error, timeout, connection error),
        automatically failover to the other servers.
        """
        # Match exact parameters from Self-InPerfect utils.py call_vllm_server
        payload = {
            "model": self.feedback_model,
            "messages": messages,
            "max_tokens": self.max_tokens,  # 2000 from Self-InPerfect
            "temperature": self.temperature,  # 0.0 from Self-InPerfect feedback_temp
            "stop_token_ids": [128001, 128009],  # Match Self-InPerfect (no tokenizer here)
            "best_of": getattr(self, 'best_of', 1),
            "n": getattr(self, 'n', 1),
            "seed": getattr(self, 'seed', 14)  # seed: 14 from Self-InPerfect
        }

        # Shuffle URLs for random load balancing, then try each in order
        shuffled_urls = self.vllm_urls.copy()
        random.shuffle(shuffled_urls)

        last_error = None
        for i, vllm_url in enumerate(shuffled_urls):
            is_last_attempt = (i == len(shuffled_urls) - 1)
            logger.debug(f"Trying vLLM server {i+1}/{len(shuffled_urls)}: {vllm_url}")

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        vllm_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                # Extract content from chat completion format
                                choice = result["choices"][0]
                                if "message" in choice:
                                    logger.info(f"Successfully got feedback from {vllm_url}")
                                    return choice["message"]["content"].strip()
                                elif "text" in choice:
                                    logger.info(f"Successfully got feedback from {vllm_url}")
                                    return choice["text"].strip()
                            last_error = "Invalid response format from feedback server."
                            logger.warning(f"{vllm_url}: {last_error}")
                            if is_last_attempt:
                                return f"Error: {last_error}"
                            continue  # Try next server
                        else:
                            # Server error (4xx, 5xx) - try next server
                            error_text = await response.text()
                            last_error = f"Server returned status {response.status}: {error_text[:100]}"
                            logger.warning(f"{vllm_url}: {last_error}")
                            if is_last_attempt:
                                return f"Error: All feedback servers failed. Last error: {last_error}"
                            continue  # Try next server

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {self.timeout}s"
                    logger.warning(f"{vllm_url}: {last_error}")
                    if is_last_attempt:
                        return f"Error: All feedback servers timed out."
                    continue  # Try next server

                except Exception as e:
                    last_error = f"Connection error: {str(e)}"
                    logger.warning(f"{vllm_url}: {last_error}")
                    if is_last_attempt:
                        return f"Error: Failed to connect to all feedback servers. Last error: {str(e)}"
                    continue  # Try next server

        # Should never reach here, but just in case
        return f"Error: Failed to generate feedback - {last_error}"

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Not used in standard flow - reward computed based on final answer."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up instance data."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]