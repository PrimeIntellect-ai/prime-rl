from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.sft import DataConfig
from prime_rl.trainer.sft.data import dataset as datasets
from prime_rl.trainer.sft.data.broker import PackedDataLoader
from prime_rl.trainer.sft.data.dataset import (
    Batch as Batch,
)
from prime_rl.trainer.sft.data.dataset import (
    CatDataset as CatDataset,
)
from prime_rl.trainer.sft.data.dataset import (
    FakeDataset as FakeDataset,
)
from prime_rl.trainer.sft.data.dataset import (
    RendererResolver as RendererResolver,
)
from prime_rl.trainer.sft.data.dataset import (
    Sample as Sample,
)
from prime_rl.trainer.sft.data.dataset import (
    SFTDataset as SFTDataset,
)
from prime_rl.trainer.sft.data.dataset import (
    StatefulIterableDataset as StatefulIterableDataset,
)
from prime_rl.trainer.sft.data.dataset import (
    _drop_null_fields as _drop_null_fields,
)
from prime_rl.trainer.sft.data.dataset import (
    cat_collate as cat_collate,
)
from prime_rl.trainer.sft.data.dataset import (
    load_sft_dataset as load_sft_dataset,
)
from prime_rl.trainer.sft.data.dataset import (
    pre_download_data as pre_download_data,
)
from prime_rl.trainer.sft.data.dataset import (
    setup_dataset as setup_dataset,
)
from prime_rl.trainer.sft.data.dataset import (
    setup_local_dataloader as setup_local_dataloader,
)


def setup_dataloader(tokenizer, config: DataConfig, cp_size: int = 1, timeout_seconds: int = 300, **dataset_kwargs):
    dataset = setup_dataset(tokenizer, config, cp_size, **dataset_kwargs)
    if isinstance(dataset, SFTDataset) and not dataset.multimodal:
        return PackedDataLoader(dataset, config, cp_size, timeout_seconds=timeout_seconds)
    return setup_local_dataloader(dataset, config)


def get_dataset_state(dataloader: PackedDataLoader | StatefulDataLoader) -> dict:
    if isinstance(dataloader, PackedDataLoader):
        return {"position": dataloader.dataset_progress["step"]}
    return datasets.get_dataset_state(dataloader)


def get_dataset_progress(dataloader: PackedDataLoader | StatefulDataLoader) -> dict:
    if isinstance(dataloader, PackedDataLoader):
        return dataloader.dataset_progress
    return datasets.get_dataset_progress(dataloader)
