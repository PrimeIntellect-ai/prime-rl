import json
import random
from typing import List, Dict, Any, Optional
from zeroband.inference.genesys.sanskript_library import compute_sanskript_library_reward

class SanskritQuoteEnv:
    def __init__(
        self,
        dataset_path: str,
        sample_size: int = 1,
        seed: Optional[int] = None
    ):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self._load_data()
        self.current_batch: List[Dict[str, Any]] = []

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
        self.current_batch = random.sample(self.data, k=self.sample_size)
        return [item["quote"] for item in self.current_batch]

    def step(self, completions: List[str]) -> (List[float], bool, Dict[str, Any]):
        if len(completions) != len(self.current_batch):
            raise ValueError(f"Number of completions ({len(completions)}) does not match batch size ({len(self.current_batch)}).")
        rewards: List[float] = []
        info: Dict[str, Any] = {}
        for comp, item in zip(completions, self.current_batch):
            verification_info = {"ground_truth": item["solution"]}
            reward = compute_sanskript_library_reward(comp, verification_info)
            rewards.append(reward)
        done = True
        return rewards, done, info