# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Augment dataset with backward reasoning."""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional

from prompt import consistency_check_prompt_math
from prompt import consistency_check_prompt_mcq
from prompt import gen_reasoning_prompt
from prompt import icl_samples
from prompt import prompt_for_backward_question
import tqdm
from utils import get_alphabet_choice
from utils import get_gemini_output
from utils import get_true_false
from utils import get_yes_no
from utils import parse_math_boxed
from utils import parse_number
from utils import remove_backward_answer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Validate that a sample has required fields."""
    required_fields = ["question", "gold_answer"]
    return all(field in sample for field in required_fields)


def save_checkpoint(results: List[Dict], task: str, checkpoint_name: str) -> None:
    """Save intermediate results as checkpoint."""
    checkpoint_path = f"./checkpoints/{task}_{checkpoint_name}.json"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    try:
        with open(checkpoint_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(task: str, checkpoint_name: str) -> Optional[List[Dict]]:
    """Load checkpoint if it exists."""
    checkpoint_path = f"./checkpoints/{task}_{checkpoint_name}.json"
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    return None


def generate_backward_questions(train_samples: List[Dict], task: str, 
                              start_idx: int = 0) -> List[Dict]:
    """Generate backward questions with error handling and checkpointing."""
    logger.info("Starting backward question generation...")
    
    # Try to load existing checkpoint
    results = load_checkpoint(task, "backward_questions")
    if results:
        logger.info(f"Loaded {len(results)} samples from checkpoint")
        start_idx = len(results)
    else:
        results = []

    failed_samples = []
    
    for idx, sample in enumerate(tqdm.tqdm(train_samples[start_idx:], 
                                          desc="Backward questions"), start_idx):
        if not validate_sample(sample):
            logger.warning(f"Sample {idx} missing required fields, skipping")
            continue
            
        try:
            tmp = {
                "sample_id": idx,
                "question": sample["question"],
                "gold_answer": sample["gold_answer"]
            }

            q = f"{sample['question']} The correct answer is {sample['gold_answer']}."
            prompt = prompt_for_backward_question.format(
                icl_samples=icl_samples[task],
                input_question=q
            )
            
            backward_question = get_gemini_output(prompt, model="pro")
            if backward_question:
                tmp["backward_question"] = remove_backward_answer(backward_question)
                results.append(tmp)
            else:
                logger.warning(f"Empty response for sample {idx}")
                failed_samples.append(idx)
                
        except Exception as e:
            logger.error(f"Error in backward question generation for sample {idx}: {e}")
            failed_samples.append(idx)
            continue
        
        # Save checkpoint every 100 samples
        if (idx + 1) % 100 == 0:
            save_checkpoint(results, task, "backward_questions")
    
    # Final checkpoint save
    save_checkpoint(results, task, "backward_questions")
    
    if failed_samples:
        logger.warning(f"Failed to process {len(failed_samples)} samples: {failed_samples}")
    
    return results


def generate_forward_reasoning(results: List[Dict], task: str, 
                             answer_extraction) -> List[Dict]:
    """Generate forward reasoning with error handling."""
    logger.info("Starting forward reasoning generation...")
    
    failed_indices = []
    
    for i, sample in enumerate(tqdm.tqdm(results, desc="Forward reasoning")):
        try:
            if "question" not in sample:
                logger.warning(f"Sample {i} missing question field")
                continue
                
            prompt = sample["question"] + gen_reasoning_prompt[task]
            forward_reasoning = get_gemini_output(prompt, model="pro")
            
            if forward_reasoning:
                sample["forward_reasoning"] = forward_reasoning
                sample["forward_pred"] = answer_extraction(forward_reasoning)
            else:
                logger.warning(f"Empty forward reasoning for sample {i}")
                failed_indices.append(i)
                
        except Exception as e:
            logger.error(f"Error in forward reasoning generation for sample {i}: {e}")
            failed_indices.append(i)
            continue
    
    # Save checkpoint
    save_checkpoint(results, task, "forward_reasoning")
    
    if failed_indices:
        logger.warning(f"Failed forward reasoning for {len(failed_indices)} samples")
    
    return results


def generate_backward_reasoning(results: List[Dict], task: str, 
                              answer_extraction) -> List[Dict]:
    """Generate backward reasoning with error handling."""
    logger.info("Starting backward reasoning generation...")
    
    failed_indices = []
    
    for i, sample in enumerate(tqdm.tqdm(results, desc="Backward reasoning")):
        try:
            if "backward_question" not in sample:
                logger.warning(f"Sample {i} missing backward_question field")
                continue
                
            prompt = sample["backward_question"] + gen_reasoning_prompt[task]
            backward_reasoning = get_gemini_output(prompt, model="pro")
            
            if backward_reasoning:
                sample["backward_reasoning"] = backward_reasoning
                sample["backward_pred"] = answer_extraction(backward_reasoning)
            else:
                logger.warning(f"Empty backward reasoning for sample {i}")
                failed_indices.append(i)
                
        except Exception as e:
            logger.error(f"Error in backward reasoning generation for sample {i}: {e}")
            failed_indices.append(i)
            continue
    
    # Save checkpoint
    save_checkpoint(results, task, "backward_reasoning")
    
    if failed_indices:
        logger.warning(f"Failed backward reasoning for {len(failed_indices)} samples")
    
    return results


def validate_consistency(results: List[Dict], task: str, 
                        consistency_check_prompt) -> List[Dict]:
    """Validate consistency with error handling."""
    logger.info("Starting consistency validation...")
    
    failed_indices = []
    
    for i, sample in enumerate(tqdm.tqdm(results, desc="Consistency check")):
        try:
            required_fields = ["question", "gold_answer", "backward_question", "backward_pred"]
            if not all(field in sample for field in required_fields):
                logger.warning(f"Sample {i} missing required fields for consistency check")
                sample["consistency_reasoning"] = "N/A"
                sample["is_consistent"] = "false"
                continue
            
            prompt = consistency_check_prompt.format(
                question=sample["question"],
                gold_answer=sample["gold_answer"],
                backward_question=sample["backward_question"],
                backward_pred=sample["backward_pred"]
            )
            
            consistency = get_gemini_output(prompt, model="pro")
            
            if consistency:
                sample["consistency_reasoning"] = consistency
                sample["is_consistent"] = get_true_false(consistency)
            else:
                logger.warning(f"Empty consistency response for sample {i}")
                sample["consistency_reasoning"] = "N/A"
                sample["is_consistent"] = "false"
                failed_indices.append(i)
                
        except Exception as e:
            logger.error(f"Error in consistency check for sample {i}: {e}")
            sample["consistency_reasoning"] = "N/A"
            sample["is_consistent"] = "false"
            failed_indices.append(i)
    
    if failed_indices:
        logger.warning(f"Failed consistency check for {len(failed_indices)} samples")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Augment dataset with backward reasoning")
    parser.add_argument("--task", default="SQA", type=str, 
                       help="Task name (SQA, ANLI, ARC, Date, CSQA, ESNLI, GSM8K, GSM8K-Rev, TabMWP, MATH)")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from checkpoint if available")
    parser.add_argument("--output_dir", default="./training_data", type=str,
                       help="Output directory for results")
    
    args = parser.parse_args()

    # Validate task
    supported_tasks = ["SQA", "ANLI", "ARC", "Date", "CSQA", "ESNLI", 
                      "GSM8K", "GSM8K-Rev", "TabMWP", "MATH"]
    if args.task not in supported_tasks:
        raise ValueError(f"Unsupported task: {args.task}. Supported tasks: {supported_tasks}")

    # Load training data
    input_path = f"./training_data/{args.task}.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Training data not found: {input_path}")
    
    try:
        with open(input_path, "r", encoding='utf-8') as f:
            train_samples = json.load(f)
        logger.info(f"Loaded {len(train_samples)} training samples")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return

    # Configure task-specific settings
    if args.task == "SQA":
        answer_extraction = get_yes_no
        consistency_check_prompt = consistency_check_prompt_mcq
    elif args.task in ["ANLI", "ARC", "Date", "CSQA", "ESNLI"]:
        answer_extraction = get_alphabet_choice
        consistency_check_prompt = consistency_check_prompt_mcq
    elif args.task in ["GSM8K", "GSM8K-Rev"]:
        answer_extraction = parse_number
        consistency_check_prompt = consistency_check_prompt_math
    elif args.task in ["TabMWP", "MATH"]:
        answer_extraction = parse_math_boxed
        consistency_check_prompt = consistency_check_prompt_math

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Generate backward questions
        results = generate_backward_questions(train_samples, args.task)
        logger.info(f"Generated backward questions for {len(results)} samples")

        # Step 2: Generate forward reasoning
        results = generate_forward_reasoning(results, args.task, answer_extraction)
        logger.info(f"Generated forward reasoning for {len(results)} samples")

        # Step 3: Generate backward reasoning  
        results = generate_backward_reasoning(results, args.task, answer_extraction)
        logger.info(f"Generated backward reasoning for {len(results)} samples")

        # Step 4: Validate consistency
        results = validate_consistency(results, args.task, consistency_check_prompt)
        logger.info(f"Completed consistency validation for {len(results)} samples")

        # Save final results
        output_path = f"{args.output_dir}/{args.task}_augmented.json"
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final results saved to: {output_path}")
        
        # Print summary statistics
        consistent_samples = sum(1 for r in results if r.get("is_consistent") == "true")
        logger.info(f"Summary: {len(results)} total samples, {consistent_samples} consistent samples")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
