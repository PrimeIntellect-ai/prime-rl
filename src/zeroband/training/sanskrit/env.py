"""Sanskrit meter generation environment for Prime-RL."""
from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import numpy as np
from transformers import AutoTokenizer

from zeroband.training.sanskrit.reward import calculate_reward
from zeroband.data.sanskrit_meters import get_data_path

class PrimeSanskritMeterEnv:
    """Environment for generating Sanskrit poetry in specific meters."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        max_length: int = 256,
        **kwargs
    ):
        # Load the prompts
        data_path = get_data_path() / "prompts.jsonl"
        self.prompts = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.prompts.append(json.loads(line))
        
        # Setup tokenizer for decoding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.return_scale = 1.0  # Scale rewards to be roughly in [-1, 1]
        
        # Episode state
        self.current_prompt = None
        self.current_generated = []
        self.current_step = 0
        self.current_logprobs = []
        
        # Episode tracking
        self._episode_rewards = []
        self._episode_lengths = []
    
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset the environment."""
        # Select a random prompt
        self.current_prompt = self.prompts[0] if seed == 0 else self.prompts[-1]  # Simple deterministic selection
        self.current_generated = []
        self.current_step = 0
        self.current_logprobs = []
        
        # Encode prompt
        prompt_encoding = self.tokenizer(
            self.current_prompt["prompt"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True
        )
        
        return {
            # Required fields for training
            "input_ids": prompt_encoding["input_ids"],
            "position_ids": torch.arange(self.max_length).unsqueeze(0),
            "attention_mask": prompt_encoding["attention_mask"],
            "loss_mask": torch.ones(self.max_length),  # All tokens contribute to loss
            "rewards": torch.zeros(1),  # Will be updated during episode
            "task_rewards": torch.zeros(1),  # Task-specific rewards
            "seq_lens": torch.tensor([0]),  # Current sequence length
            "length_penalties": torch.zeros(1),  # Length-based penalties
            "target_lengths": torch.tensor([self.max_length]),  # Target sequence length
            "advantages": torch.zeros(1),  # Will be computed from rewards
            "logprobs": torch.zeros(1),  # Log probabilities of chosen actions
            
            # Additional info
            "prompt": self.current_prompt["prompt"],
            "meter": self.current_prompt["meter"],
            "topic": self.current_prompt.get("topic"),
            "keywords": self.current_prompt.get("keywords", [])
        }
    
    def step(self, token_id: int, logprob: Optional[float] = None) -> tuple[Dict, float, bool, Dict]:
        """Take a step in the environment."""
        # Add token and logprob to sequences
        self.current_generated.append(token_id)
        if logprob is not None:
            self.current_logprobs.append(logprob)
        self.current_step += 1
        
        # Convert to text and calculate reward
        generated_text = self.tokenizer.decode(self.current_generated)
        reward_info = calculate_reward(
            generated_text,
            self.current_prompt["meter"],
            topic=self.current_prompt.get("topic"),
            topic_keywords=set(self.current_prompt.get("keywords", []))
        )
        
        # Scale reward to be roughly in [-1, 1]
        reward = reward_info["reward"] * self.return_scale
        
        # Create tensors for training
        input_ids = torch.tensor(self.current_generated).unsqueeze(0)
        seq_len = len(self.current_generated)
        
        # Episode is done if we hit max length or generate EOS token
        done = (self.current_step >= self.max_length or
                token_id == self.tokenizer.eos_token_id)
        
        # Add episode stats
        if done:
            self._episode_rewards.append(reward)
            self._episode_lengths.append(self.current_step)
            reward_info["episode"] = {
                "r": sum(self._episode_rewards[-100:]) / len(self._episode_rewards[-100:]),
                "l": sum(self._episode_lengths[-100:]) / len(self._episode_lengths[-100:])
            }
        
        # Return step info with all required fields
        obs = {
            # Required fields for training
            "input_ids": input_ids,
            "position_ids": torch.arange(seq_len).unsqueeze(0),
            "attention_mask": torch.ones(1, seq_len),
            "loss_mask": torch.ones(seq_len),
            "rewards": torch.tensor([reward]),
            "task_rewards": torch.tensor([reward_info.get("task_reward", reward)]),
            "seq_lens": torch.tensor([seq_len]),
            "length_penalties": torch.tensor([reward_info.get("length_penalty", 0.0)]),
            "target_lengths": torch.tensor([self.max_length]),
            "advantages": torch.tensor([reward]),  # Simple advantage = reward
            "logprobs": torch.tensor(self.current_logprobs) if self.current_logprobs else torch.zeros(1),
            
            # Additional info
            "prompt": self.current_prompt["prompt"],
            "generated_text": generated_text,
            "step": self.current_step,
            "meter_score": reward_info.get("meter_score", 0.0),
            "semantic_score": reward_info.get("semantic_score", 0.0)
        }
        
        return obs, reward, done, reward_info
    
    @property
    def unwrapped(self):
        """Get the base environment."""
        return self
    
    def get_normalized_score(self, reward: float) -> float:
        """Convert raw reward to normalized score (0 to 1)."""
        return max(0.0, min(1.0, (reward + 1.0) / 2.0))
