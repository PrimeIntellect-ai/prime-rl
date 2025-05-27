import json
import random
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from zeroband.inference.genesys.sanskript_library import compute_sanskript_library_reward
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

class SanskritLibraryEnv:
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 1,
        seed: Optional[int] = None,
        model_name: str = "Qwen/Qwen3-8B",
        max_model_len: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self._load_data()
        self.current_batch: List[Dict[str, Any]] = []

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def _load_data(self) -> None:
        self.data: List[Dict[str, Any]] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "quote" in item and "solution" in item:
                    self.data.append(item)

    def reset(self) -> List[str]:
        if not self.data:
            raise ValueError("Dataset is empty. Please check dataset_path.")
        self.current_batch = random.sample(self.data, k=self.batch_size)
        return [item["quote"] for item in self.current_batch]

    def step(self):
        prompts = [item["quote"] for item in self.current_batch]
        outputs = self.llm.generate(prompts, self.sampling_params)
        completions = [output.outputs[0].text for output in outputs]
        
        rewards = []
        for comp, item in zip(completions, self.current_batch):
            verification_info = {"ground_truth": item["solution"]}
            reward = compute_sanskript_library_reward(comp, verification_info)
            rewards.append(reward)
        reward_array = np.array(rewards, dtype=np.float32)
        advantages = (reward_array - reward_array.mean()) / (reward_array.std(ddof=1) + 1e-6)
        
        #TBD train loop
        records = []
        for output, reward, advantage in zip(outputs, rewards, advantages):
            records.append({
                "input_tokens": output.prompt_token_ids,
                "output_tokens": output.outputs[0].token_ids,
                "advantages": float(advantage),
                "rewards": float(reward),
                "task_rewards": float(reward),
                "length_penalties": 0.0,
                "proofs": b"",
                "step": 0,
                "target_lengths": len(output.outputs[0].token_ids)
            })

        table = pa.Table.from_pylist(records, schema=pa_schema)
        pq.write_table(table, f"outputs/step_{self.step_count}/data.parquet")
        
        return rewards, True, {}