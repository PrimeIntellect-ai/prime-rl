"""
This script is a drop in replacement for the train.py that only run the data, useful for debugging the data pipeline.

Example usage:

lets assume your training run with this command :
    uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/prime_rl/trainer/sft/train.py @ configs/debug/sft/train.toml
    
and that you want to debug the data pipeline, you can run this command to only run the data:

    uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/prime_rl/trainer/sft/data_debug.py @ configs/debug/sft/train.toml
    
notice how the only change is the script name. The API is the name but only tests the data.
"""

import time
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from loguru import logger
from prime_rl.trainer.ckpt import Progress
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.model import setup_tokenizer

from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.utils import setup_torch_distributed

from prime_rl.trainer.world import get_world
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit


@clean_exit
@logger.catch(reraise=True)
def train(config: SFTTrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
    )
    logger.info(f"Starting SFT trainer in only data debug mode{world}")

  
    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Set precision
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))


    tokenizer = setup_tokenizer(config.model)
  
    # Set up the dataset and dataloader
    logger.info(f"Initializing data ({config.data})")
    dataset = setup_dataset(tokenizer, config.data, config.model.cp * config.model.tp)
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # Check that the world size and batch configuration is compatible
    num_micro_batches = config.data.batch_size // config.data.micro_batch_size
    if world.world_size > num_micro_batches:
        raise ValueError(
            f"There must be at least one micro batch per rank, but only have {num_micro_batches} micro batches for {world.world_size} ranks."
        )
    if num_micro_batches % world.world_size != 0:
        raise ValueError(
            f"The number of micro batches ({num_micro_batches}) must be divisible by the world size ({world.world_size})."
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()

    while True:

       
        grad_accum_steps = (
            config.data.batch_size
            * config.model.cp
            * config.model.tp
            // (config.data.micro_batch_size * world.world_size)
        )
        
                # Break if we have reached the maximum number of steps
        if config.max_steps is not None and progress.step >= config.max_steps:
            break
        
        step_start_time = time.time()

        for micro_step in range(grad_accum_steps):
            micro_batch = next(dataiter)
            

            logger.debug(f"Micro Step {micro_step}/{grad_accum_steps} input_ids: {micro_batch['input_ids'].shape}")
          

        # Compute step metrics
        num_tokens = config.data.batch_size * config.data.seq_len
        progress.total_tokens += num_tokens
        progress.total_samples = dataset.step
    
        step_time = time.time() - step_start_time


        # Log step metrics
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | total_tokens: {progress.total_tokens} | total_samples: {progress.total_samples}%)"
    
        logger.success(step_message)

        # Log progress metrics
        progress_metrics = {
            "progress/epoch": dataset.epoch,
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "step": progress.step,
        }
        # At least two subsets/splits
       
        monitor.log(progress_metrics)

        # Log performance metrics
        progress.step += 1



def main():
    train(parse_argv(SFTTrainerConfig))


if __name__ == "__main__":
    main()
