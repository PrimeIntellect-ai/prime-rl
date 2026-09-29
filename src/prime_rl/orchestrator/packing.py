from transformers import AutoConfig

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.trainer.batch import build_bin_cost, prepare_batch
from prime_rl.transports.batch.types import MicroBatch, TrainingSample
from prime_rl.utils.logger import get_logger


class BatchPacker:
    """Bin-packs a step's training samples into one micro-batch list per trainer DP rank."""

    def __init__(self, config: OrchestratorConfig, *, for_value: bool = False):
        self.seq_len = config.value_seq_len if for_value else config.seq_len
        self.num_train_workers = config.value_num_train_workers if for_value else config.num_train_workers
        self.pad_to_multiple_of = config.value_pad_to_multiple_of if for_value else config.pad_to_multiple_of
        self.for_value = for_value
        if self.seq_len is None or self.num_train_workers is None or self.pad_to_multiple_of is None:
            raise ValueError("Value packing requires critic sequence length, DP count, and CP padding")
        try:
            model_config = AutoConfig.from_pretrained(
                config.model.name, trust_remote_code=config.tokenizer.trust_remote_code
            )
        except Exception as e:
            get_logger().warning(
                f"Could not load model config for {config.model.name} ({e}) - "
                "packing balances by token count instead of estimated FLOPs"
            )
            model_config = None
        self.bin_cost = build_bin_cost(model_config)

    def pack(self, samples: list[TrainingSample]) -> list[list[MicroBatch]]:
        return prepare_batch(
            rollouts=samples,
            seq_len=self.seq_len,
            num_train_workers=self.num_train_workers,
            bin_cost=self.bin_cost,
            pad_to_multiple_of=self.pad_to_multiple_of,
            for_value=self.for_value,
        )
