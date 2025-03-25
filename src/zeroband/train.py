import os
from pathlib import Path
import shutil
import time
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # type: ignore
import wandb

from zeroband.models import AttnImpl, ModelName, ModelType, get_model_and_tokenizer
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.data import DataConfig, get_dataloader
from zeroband.training.loss import grpo_loss, selective_log_softmax, entropy_loss
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import PerfCounter, apply_ac_ckpt

from zeroband.logger import get_logger

from pydantic_config import BaseConfig, parse_argv
from jaxtyping import Float

from zeroband.training.world_info import WorldInfo, get_world_info

from pydantic import model_validator

from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
import logging

import json


class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.01
    betas1: float = 0.9
    betas2: float = 0.99


class OptimConfig(BaseConfig):
    optim: AdamConfig = AdamConfig()
    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    step_per_rollout: int = 1


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    memory_profile: str | None = None
    torch_compile: bool = True
    liger_qwen: bool = False

    attn_impl: AttnImpl = "flex_attention"


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    resume: str | None = None

    rollout_path: str | None = None  # if rollout path is set we saved at each step


class Config(BaseConfig):
    name_model: ModelName = "150M"

    ckpt: CkptConfig = CkptConfig()

    project: str = "prime_simple"
    wandb: bool = False

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    gpus_ids: list[int] | None = None

    temperature: float = 0.6  # todo remove this and add this to the data
    grpo_epsilon: float = 0.2
    entropy_loss_coeff: float = 0.001

    on_policy_log_prob: bool = True

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.name_model, "train.liger_qwen can only be applied to Qwen2 models."
        return self


def get_gradient_accumulation_steps(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    print(f"batch_size: {batch_size}, micro_bs: {micro_bs}, data_workers: {data_workers}")
    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size // micro_bs


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=None)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank

    if world_info.local_rank >= len(gpus_ids):
        raise ValueError(f"Local rank {world_info.local_rank} is greater than the number of available GPUs ({len(gpus_ids)})")

    return gpus_ids[world_info.local_rank]


def train(config: Config):
    if "ZERO_BAND_DEV" not in os.environ:
        torch._logging.set_logs(dynamo=logging.CRITICAL)  # silent flex attn error
        torch_log.setLevel(logging.CRITICAL)  #

    logger = get_logger()
    world_info = get_world_info()

    logger.info(f"start training on {world_info.world_size}")

    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    # batch_size is the total batch size for all GPUs

    gradient_accumulation_steps = get_gradient_accumulation_steps(
        config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info
    )

    model, tokenizer = get_model_and_tokenizer(config.name_model, config.train.attn_impl)

    model = model.cuda()

    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        micro_batch_size=config.train.micro_bs,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
    )

    train_dataloader_iterator = iter(train_dataloader)

    if config.train.liger_qwen:
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            model=model,
        )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    apply_fsdp(model, config.train.reshard_after_forward)

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=config.optim.optim.lr,weight_decay=config.optim.optim.weight_decay,betas=(config.optim.optim.betas1, config.optim.optim.betas2))  # fmt: skip

    scheduler = get_scheduler(sched_type=config.optim.sched_type,optimizer=optimizer,num_warmup_steps=config.optim.warmup_steps,num_stable_steps=config.optim.stable_steps,num_training_steps=config.optim.total_steps)  # fmt: skip

    training_progress = TrainingProgress(total_tokens=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    # if config.train.torch_compile:
    #    model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume:
        load_checkpoint_fsdp_state(model, [optimizer], training_progress, train_dataloader, scheduler, config.ckpt.resume)

    perf_counter = PerfCounter(window_size=10, model=model, seq_len=config.data.seq_length)

    previous_ckpt_rollout = []
    
    exit()

    while True:
        time_start = time.time()

        data_loader = iter(get_batch_2())
        
        

        # here we want to pre-compute the logprobs with the model before update
        with torch.no_grad():
            if config.on_policy_log_prob:
                data = []

                for rollout_step in range(1):
                    for grad_acc_step in range(6):
                        batch = next(data_loader)
                        # batch = next(train_dataloader_iterator)
                        # for k in batch.keys():
                        #    print(batch[k].shape)

                        # print("batch", batch.keys())
                        input_ids = batch["input_ids"].to("cuda")
                        attention_mask = batch["full_attention_mask"].to("cuda")
                        position_ids = batch["full_position_ids"].to("cuda")

                        logits = model(input_ids=input_ids).logits  # .contiguous()

                        input_ids = input_ids[:, 1:]
                        logits = logits[:, :-1, :] / config.temperature

                        per_token_logps = selective_log_softmax(logits, input_ids)

                        batch["logprobs"] = per_token_logps.to("cpu")

                        del logits, per_token_logps
                        data.append(batch)

                logprobs_aware_iterator = iter(data)
            else:
                logprobs_aware_iterator = train_dataloader_iterator
        
                
        print("LEN BATCH", len(batch["input_ids"]))
                
        print("gradient accumulation", gradient_accumulation_steps)

        for rollout_step in range(config.optim.step_per_rollout):
            loss_batch = 0
            pg_loss_batch = 0
            entropy_loss_batch = 0
            clip_ratio_batch = 0
            seq_lens_batch = 0

            rewards_sum = torch.tensor(0.0)
            rewards_token_count = torch.tensor(0.0)

            if config.train.memory_profile and world_info.rank == 0:
                torch.cuda.memory._record_memory_history()

            for grad_acc_step in range(6):
                batch = next(logprobs_aware_iterator)
                
                is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                model.set_requires_gradient_sync(not is_accumulating)  # no sync if we are accumulating gradients

                # Load args
                # batch = next(logprobs_aware_iterator)
                input_ids = batch["input_ids"].to("cuda")
                loss_mask = batch["loss_mask"]

                # rewards = batch["rewards"][loss_mask.bool()]
                # rewards_sum += rewards.sum()
                # rewards_token_count += rewards.numel()

                seq_lens_batch += batch["seq_lens"].float().mean() / gradient_accumulation_steps

                # Forward
                logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids).logits.contiguous()

                # Gather args for grpo loss
                advantages = batch["advantages"].to("cuda")
                loss_mask = loss_mask.to("cuda")
                original_logprobs = batch["logprobs"].to("cuda")
                if not config.on_policy_log_prob:
                    original_logprobs = original_logprobs[:, 1:]

                # Loss
                #print("target logprob", target_logp[:20])
                # diff = (target_logp-original_logprobs[:,79:]).abs()
                # print("diff mean", diff.mean())
                # print("diff max", diff.max())
                #print("all close???", torch.allclose(target_logp, original_logprobs[:, 44:]))

                pg_loss, clip_ratio = grpo_loss(
                    logits, input_ids, advantages, original_logprobs, loss_mask, config.temperature, config.grpo_epsilon
                )
                entropy = entropy_loss(logits, loss_mask, config.temperature)

                loss = pg_loss - config.entropy_loss_coeff * entropy
                loss = loss / gradient_accumulation_steps
                clip_ratio = clip_ratio / gradient_accumulation_steps

                del batch, logits, input_ids, advantages, loss_mask, original_logprobs

                # Backward
                print("pg loss", pg_loss)
                print("gradient accumulation steps", gradient_accumulation_steps)
                print("loss a", loss*gradient_accumulation_steps)
                loss.backward()
                loss_batch += loss.detach().clone()
                pg_loss_batch += (pg_loss / gradient_accumulation_steps).detach().clone()
                entropy_loss_batch += (entropy / gradient_accumulation_steps).detach().clone()
                clip_ratio_batch += clip_ratio.detach().clone()
                del loss, clip_ratio, pg_loss, entropy

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=pg_loss_batch, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=entropy_loss_batch, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=clip_ratio_batch, op=dist.ReduceOp.AVG)

            seq_lens_batch = seq_lens_batch / world_info.world_size
            dist.all_reduce(tensor=seq_lens_batch, op=dist.ReduceOp.SUM)

            # dist.all_reduce(rewards_sum, op=dist.ReduceOp.SUM)
            # dist.all_reduce(rewards_token_count, op=dist.ReduceOp.SUM)
            # average_rewards = rewards_sum / rewards_token_count

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore (is a dtensor)

            print("grad_norm", grad_norm)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # logging
            training_progress.step += 1
            inner_lr = [group["lr"] for group in optimizer.param_groups][0]

            # syncing loss across all data parallel rank within a nodes
            new_tokens = config.data.seq_length * config.optim.batch_size
            perf_counter.count_tokens(new_tokens)
            training_progress.total_tokens += new_tokens

            padding_proportion = (config.data.seq_length - seq_lens_batch.item() - 1) / config.data.seq_length

            metrics = {
                "Loss": loss_batch.item(),
                "pg_loss": pg_loss_batch.item(),
                "entropy_loss": entropy_loss_batch.item(),
                "step": training_progress.step,
                "rollout_step": rollout_step,
                "seq_lens": seq_lens_batch.item(),
                "inner_lr": inner_lr,
                "Perplexity": torch.exp(loss_batch).item(),
                "total_tokens": training_progress.total_tokens,
                "time": time.time(),
                "grad_norm": grad_norm.item(),
                # "average_rewards": average_rewards.item(),
                "clip_ratio": clip_ratio_batch.item(),
                "padding_proportion": padding_proportion,
            }

            log = f"step: {training_progress.step}, rollout_step: {training_progress.step // config.optim.step_per_rollout}, loss: {loss_batch.item():.4f}"  # , average_rewards: {average_rewards.item():.4f}"

            del loss_batch, grad_norm, pg_loss_batch, entropy_loss_batch  # , average_rewards

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = perf_counter.get_mfu()
                log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

            if world_info.rank == 0 and config.wandb:
                wandb.log(metrics)

            logger.info(log)

            if config.train.memory_profile and (training_progress.step == 2) and world_info.rank == 0:
                logger.info("Dumping memory snapshot.")
                pickle_path: str = config.train.memory_profile
                if not pickle_path.endswith(".pickle"):
                    pickle_path += ".pickle"
                torch.cuda.memory._dump_snapshot(pickle_path)
                torch.cuda.memory._record_memory_history(enabled=False)

            if config.ckpt.interval is not None and training_progress.step % config.ckpt.interval == 0:
                save_checkpoint_fsdp_state(model, [optimizer], training_progress, train_dataloader, scheduler, config.ckpt.path)

            if config.ckpt.rollout_path is not None and training_progress.step % config.optim.step_per_rollout == 0:
                rollout_step = training_progress.step // config.optim.step_per_rollout
                path = Path(config.ckpt.rollout_path) / f"step_{rollout_step}"
                previous_ckpt_rollout.append(path)
                save_ckpt_for_rollout(model, path)

                if len(previous_ckpt_rollout) > 2:
                    path_to_delete = previous_ckpt_rollout.pop(0)
                    if path_to_delete.exists():
                        logger.info(f"Removing past rollout ckpt at {path_to_delete}")
                        shutil.rmtree(path_to_delete, ignore_errors=True)


        logger.info(f"Finished rollout {rollout_step} step {training_progress.step}")
        if world_info.rank == 0 and config.wandb:
            wandb.log({"rollout_step": rollout_step, "step": training_progress.step, "time_rollout_step": time.time() - time_start})

        if training_progress.step >= config.optim.total_steps:
            break

        print("DONE WITH STEP")
        exit()

    if prefetcher is not None:
        prefetcher.shutdown()

    logger.info("Training finished, exiting ...")
    logger.info(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def get_num_leading_eos(input_ids, eos_id=151643):
    # Find the index of the first token that is not the eos token.
    non_eos = (input_ids != eos_id).nonzero(as_tuple=True)[0]
    if non_eos.numel() == 0:
        # In case all tokens are eos_id, return an empty tensor.
        return input_ids[:0]
    first_non_eos_idx = non_eos[0].item()
    return first_non_eos_idx


def get_batch():
    KEYS = ["old_log_prob", "log_prob", "advantages", "pg_losses", "pg_losses2", "old_log_probs"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    with open("grouped_metrics_2.json", "r") as f:
        data = json.load(f)

    for k in data.keys():
        batch = dict()
        for key in data[k][4].keys():
            print("key", key)
            try:
                print(torch.Tensor(data[k][4][key]).shape)
            except Exception:
                pass
        input_ids = torch.LongTensor(data[k][4]["input_ids"])
        advantages = torch.LongTensor([])
        non_eos_idx = get_num_leading_eos(input_ids)
        batch["input_ids"] = input_ids[non_eos_idx:].unsqueeze(0)
        batch["advantages"] = torch.Tensor([data[k][4]["advantages"][513]] * batch["input_ids"].shape[1]).unsqueeze(0)
        batch["loss_mask"] = torch.LongTensor(data[k][4]["attention_mask"][non_eos_idx:]).unsqueeze(0)
        batch["seq_lens"] = torch.LongTensor(data[k][4]["attention_mask"]).sum().unsqueeze(0)
        batch["full_input_ids"] = torch.LongTensor(input_ids.unsqueeze(0))
        batch["full_attention_mask"] = torch.LongTensor(data[k][4]["attention_mask"]).unsqueeze(0)
        batch["full_position_ids"] = torch.LongTensor(data[k][4]["position_ids"]).unsqueeze(0)

        print("full attention mask", batch["full_attention_mask"][:, :20])
        print("full attention mask", batch["full_attention_mask"][:, 350:500])
        print("full attention mask", batch["full_attention_mask"][:, 2980:3000])

        print("INPUT IDS SHAPE", batch["full_input_ids"].shape)
        print("LOG PROB SHAPE", torch.Tensor(data[k][4]["log_prob"]).shape)

        print("decoded", tokenizer.decode(batch["full_input_ids"][0, 512:550]))
        print("\n\n")
        print("text", tokenizer.decode(batch["input_ids"][0, 44:600]))

        print(batch)

        print("Losses:")
        for key in KEYS:
            print("key", key)
            print(torch.BFloat16Tensor(data[k][4][key]))
            print("\n- - - - - - - - - -\n")
            
        print("key loss", k)

        # batch["returns"] = [data[k][4]["returns"][513]]*len(input_ids)
        # batch["rewards"] = [batch["rewards"][0]]*len(input_ids)

        logp = torch.BFloat16Tensor(data[k][4]["old_log_probs"]).cuda()
        
        exit()

        return batch, logp
    
    
def get_batch():
    KEYS = ["old_log_prob", "log_prob", "advantages", "pg_losses", "pg_losses2", "old_log_probs"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    with open("grouped_metrics_2.json", "r") as f:
        data = json.load(f)

    for k in data.keys():
        batch = dict()
        for key in data[k][4].keys():
            print("key", key)
            try:
                print(torch.Tensor(data[k][4][key]).shape)
            except Exception:
                pass
            
        input_ids = torch.LongTensor(data[k][4]["input_ids"])
        advantages = torch.LongTensor([])
        non_eos_idx = get_num_leading_eos(input_ids)
        batch["input_ids"] = input_ids[non_eos_idx:].unsqueeze(0)
        batch["advantages"] = torch.Tensor([data[k][4]["advantages"][513]] * batch["input_ids"].shape[1]).unsqueeze(0)
        batch["loss_mask"] = torch.LongTensor(data[k][4]["attention_mask"][non_eos_idx:]).unsqueeze(0)
        batch["seq_lens"] = torch.LongTensor(data[k][4]["attention_mask"]).sum().unsqueeze(0)
        batch["full_input_ids"] = torch.LongTensor(input_ids.unsqueeze(0))
        batch["full_attention_mask"] = torch.LongTensor(data[k][4]["attention_mask"]).unsqueeze(0)
        batch["full_position_ids"] = torch.LongTensor(data[k][4]["position_ids"]).unsqueeze(0)

        print("full attention mask", batch["full_attention_mask"][:, :20])
        print("full attention mask", batch["full_attention_mask"][:, 350:500])
        print("full attention mask", batch["full_attention_mask"][:, 2980:3000])

        print("INPUT IDS SHAPE", batch["full_input_ids"].shape)
        print("LOG PROB SHAPE", torch.Tensor(data[k][4]["log_prob"]).shape)

        print("decoded", tokenizer.decode(batch["full_input_ids"][0, 512:550]))
        print("\n\n")
        print("text", tokenizer.decode(batch["input_ids"][0, 44:600]))

        print(batch)

        print("Losses:")
        for key in KEYS:
            print("key", key)
            print(torch.BFloat16Tensor(data[k][4][key]))
            print("\n- - - - - - - - - -\n")
            
        print("key loss", k)

        # batch["returns"] = [data[k][4]["returns"][513]]*len(input_ids)
        # batch["rewards"] = [batch["rewards"][0]]*len(input_ids)

        logp = torch.BFloat16Tensor(data[k][4]["old_log_probs"]).cuda()
        
        exit()

        return batch, logp



def get_batch_2(json_path="grouped_metrics_2.json"):
    from transformers import AutoTokenizer
    """
    This function builds a single batch from *all* entries
    in the JSON file, rather than just the entry at index [4].
    """

    KEYS = ["old_log_prob", "log_prob", "advantages", "pg_losses", "pg_losses2", "old_log_probs"]
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    with open(json_path, "r") as f:
        data = json.load(f)

    # Prepare python lists to collect tensors for each field
    all_input_ids = []
    all_advantages = []
    all_loss_masks = []
    all_seq_lens = []
    all_full_input_ids = []
    all_full_attention_masks = []
    all_full_position_ids = []
    
    # (Optional) also collect any of the "log_prob" fields you might want to batch
    all_log_probs = []

    # If you want to look at all (k, i) pairs in data
    # data[k] is presumably a list; we iterate over each element
    for k in data.keys():
        for i, item in enumerate(data[k]):
            # item should be the dictionary that used to be data[k][i]
            # Check if the necessary fields exist
            if not isinstance(item, dict):
                # If item is not a dict, you may want to skip or handle differently
                continue
            
            print("ACTUAL LOSS", k)
            print("pg_losses", item["pg_losses"][600:610])
            print("pg losses 2", item["pg_losses2"][600:610])

            # Make sure your item has the fields you need
            if "input_ids" not in item:
                continue  # skip if missing required data

            # Extract input_ids, attention_mask, etc.
            input_ids = torch.LongTensor(item["input_ids"])

            # Optionally compute how many leading EOS tokens to skip
            non_eos_idx = get_num_leading_eos(input_ids)
            print("NUM EOS", non_eos_idx)

            # Crop input_ids and attention_mask to skip leading EOS
            cropped_input_ids = input_ids[non_eos_idx:]
            cropped_attn_mask = torch.LongTensor(item["attention_mask"][non_eos_idx:])
            
            prefix_zero_count = 512 - non_eos_idx
            if prefix_zero_count > 0 and prefix_zero_count < cropped_attn_mask.shape[0]:
                cropped_attn_mask[:prefix_zero_count] = 0

            # Here you used the 513th advantage previously. 
            # Make sure that your sequence is long enough for index 513
            # or adapt to the actual length you need. 
            # For example, you can clamp the index or skip if too short:
            adv_index = min(513, len(item["advantages"]) - 1)
            advantage_val = item["advantages"][adv_index]
            
            # For the entire sequence, we replicate that advantage across all tokens:
            advantages = torch.full(
                (cropped_input_ids.shape[0],),  # shape across seq_len
                fill_value=advantage_val,
                dtype=torch.float
            )

            # We can store these in lists to stack later
            all_input_ids.append(cropped_input_ids)
            all_advantages.append(advantages)
            all_loss_masks.append(cropped_attn_mask)
            all_seq_lens.append(torch.LongTensor([torch.LongTensor(item["attention_mask"]).sum()]))

            # full versions (no cropping)
            full_input_ids = torch.LongTensor(item["input_ids"])
            full_attention_mask = torch.LongTensor(item["attention_mask"])
            full_position_ids = torch.LongTensor(item["position_ids"])

            all_full_input_ids.append(full_input_ids)
            all_full_attention_masks.append(full_attention_mask)
            all_full_position_ids.append(full_position_ids)

            # If you need to track log probabilities or other fields in a batch:
            if "log_prob" in item:
                all_log_probs.append(torch.Tensor(item["log_prob"]))
            else:
                # Or append a placeholder if missing
                all_log_probs.append(torch.zeros_like(cropped_input_ids, dtype=torch.float))

        # Now we have lists of tensors, one per (k, i). We can stack or pad them.
        # If all sequences are the *same* length, you can do a simple stack:
        #   stacked_input_ids = torch.stack(all_input_ids, dim=0)
        #
        # If sequences differ in length, you may want to pad them:
        #   stacked_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=0)
        #
        # For demonstration, let's assume they are the same length or we want to pad.

        # from torch.nn.utils.rnn import pad_sequence  # Make sure to import at the top if needed.
        

        
        eos_id = 151643
        stacked_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=eos_id)
        stacked_advantages = torch.nn.utils.rnn.pad_sequence(all_advantages, batch_first=True, padding_value=0)
        stacked_loss_masks = torch.nn.utils.rnn.pad_sequence(all_loss_masks, batch_first=True, padding_value=0)
        stacked_seq_lens = torch.cat(all_seq_lens, dim=0)

        # For full input_ids, we do the same:
        stacked_full_input_ids = torch.nn.utils.rnn.pad_sequence(all_full_input_ids, batch_first=True, padding_value=eos_id)
        stacked_full_attention_masks = torch.nn.utils.rnn.pad_sequence(all_full_attention_masks, batch_first=True, padding_value=eos_id)
        stacked_full_position_ids = torch.nn.utils.rnn.pad_sequence(all_full_position_ids, batch_first=True, padding_value=eos_id)

        # Similarly for log_probs if you want to store them:
        # (They may also require padding if their lengths differ.)
        stacked_log_probs = torch.nn.utils.rnn.pad_sequence(all_log_probs, batch_first=True, padding_value=eos_id)

        # Build final batch dict
        batch = {
            "input_ids": stacked_input_ids,
            "advantages": stacked_advantages,
            "loss_mask": stacked_loss_masks,
            "seq_lens": stacked_seq_lens,
            "full_input_ids": stacked_full_input_ids,
            "full_attention_mask": stacked_full_attention_masks,
            "full_position_ids": stacked_full_position_ids,
            "log_prob": stacked_log_probs,  # optional
        }
        
        
        all_data_batch = []
        for i in range(batch["input_ids"].shape[0]):
            w = dict()
            for key in batch.keys():
                
                if key == "seq_lens":
                    w[key] = batch[key][i]
                    continue
                
                w[key] = batch[key][i:i+1,:]
            
            all_data_batch.append(w)
        
        print("actual batch loss", k)


                        
            
        for k, v in batch.items():
            print(k, v.shape if isinstance(v, torch.Tensor) else type(v))
            if "loss" in k:
                print(v)
                
        # Return the entire batch (and anything else you might need)
        return all_data_batch



# batch dict_keys(['input_ids', 'advantages', 'rewards', 'loss_mask', 'logprobs', 'seq_lens'])

if __name__ == "__main__":
    #get_batch_2()

    """
    import json
    
    KEYS = [
        'old_log_prob', 
        'log_prob', 
        'advantages', 
        'eos_mask', 
        'cliprange', 
        'negative_approx_kl', 
        'ratio', 
        'ppo_kl', 
        'pg_losses', 
        'pg_losses2', 
        'pg_loss', 
        'pg_clipfrac', 
        'responses', 
        'position_ids', 
        'attention_mask', 
        'input_ids', 
        'prompts', 
        'token_level_scores', 
        'old_log_probs', 
        'ref_log_prob', 
        'token_level_rewards', 
        'returns'
    ]
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    with open("grouped_metrics.json", "r") as f:
        data = json.load(f)
            
    for k in data.keys():
        input_ids = torch.LongTensor(data[k][4]["input_ids"])
        print(list(data[k][4].keys()))
        print("\n\n- - - - - - - - - - -\n\n")
        print(input_ids.shape)
        print(input_ids[432:450])
        print(tokenizer.decode(input_ids[432:470]))
        #print(input_ids[600:620])
        #print(input_ids[-20:])
        break
    
    """

    train(Config(**parse_argv()))
