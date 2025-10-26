#!/usr/bin/env python3
"""
Test script to verify feedback tool works with your vLLM server
"""

import asyncio
import sys

# Add paths
sys.path.append('/weka/scratch/dkhasha1/djiang21/self-inperfect-verl/verl_code_base')
sys.path.append('/scratch/dkhasha1/djiang21/Self-InPerfect_new')

from recipe.feedback_rl.feedback_tool import SelfInPerfectFeedbackTool
from verl.tools.base_tool import OpenAIFunctionToolSchema


async def test_feedback_tool():
    """Test the feedback tool with a sample math problem."""

    # Tool configuration
    config = {
        "vllm_url": "http://h05:1233/v1/chat/completions",
        "feedback_model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "temperature": 0.0,
        "max_tokens": 2000,
        "dataset": "math",
        "use_process_feedback": True
    }

    # Tool schema
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "get_feedback",
            "description": "Get feedback on your answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_answer": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["current_answer"]
            }
        }
    )

    # Create tool
    tool = SelfInPerfectFeedbackTool(config, tool_schema)

    # Test problem
    question = "What is 15 × 23?"
    ground_truth = "345"

    # Create instance
    instance_id, _ = await tool.create(
        ground_truth=ground_truth,
        question=question,
        data={"problem": question, "solution": ground_truth}
    )

    print(f"Testing feedback tool with question: {question}")
    print(f"Ground truth: {ground_truth}")

    # Test with wrong answer
    wrong_answer = "340"
    print(f"\n1. Testing with wrong answer: {wrong_answer}")

    response, score, metrics = await tool.execute(
        instance_id,
        {"current_answer": wrong_answer, "reasoning": "I multiplied 15 × 23 = 340"}
    )

    print(f"Feedback: {response.text}")

    # Test with another attempt
    print(f"\n2. Testing second attempt...")

    response2, score2, metrics2 = await tool.execute(
        instance_id,
        {"current_answer": "345", "reasoning": "Let me recalculate: 15 × 23 = 345"}
    )

    print(f"Feedback: {response2.text}")

    # Clean up
    await tool.release(instance_id)
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_feedback_tool())