from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.sft import DataConfig
from prime_rl.trainer.sft.data import dataset as datasets
from prime_rl.trainer.sft.data.broker import PackedDataLoader


def setup_dataloader(
    tokenizer,
    config: DataConfig,
    cp_size: int = 1,
    timeout_seconds: int = 300,
    *,
    validation: bool = False,
    **dataset_kwargs,
):
    dataset = datasets.setup_dataset(tokenizer, config, cp_size, **dataset_kwargs)
    if isinstance(dataset, datasets.SFTDataset) and not dataset.multimodal:
        return PackedDataLoader(dataset, config, cp_size, timeout_seconds=timeout_seconds, validation=validation)
    return datasets.setup_local_dataloader(dataset, config)


def get_dataset_state(dataloader: PackedDataLoader | StatefulDataLoader) -> dict:
    if isinstance(dataloader, PackedDataLoader):
        return {"position": dataloader.dataset_progress["step"]}
    return datasets.get_dataset_state(dataloader)


def get_dataset_progress(dataloader: PackedDataLoader | StatefulDataLoader) -> dict:
    if isinstance(dataloader, PackedDataLoader):
        return dataloader.dataset_progress
    return datasets.get_dataset_progress(dataloader)
