import torch
from muon_fsdp2 import Muon
from torch.optim import Optimizer

from prime_rl.trainer.config import OptimizerConfig
from prime_rl.trainer.model import Model


def create_optimizer(config: OptimizerConfig, model: Model) -> Optimizer:
    match config.name:
        case "AdamW":
            return torch.optim.AdamW(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )
        case "Muon":

            def muon_enabled(n, p):
                if p.ndim < 2:
                    return False
                if "lm_head" in n:
                    return False
                if "embed_tokens" in n:
                    return False
                return True

            muon_params = [p for n, p in model.named_parameters() if p.requires_grad and muon_enabled(n, p)]
            adamw_params = [p for n, p in model.named_parameters() if p.requires_grad and not muon_enabled(n, p)]

            optimizer = Muon(
                [
                    dict(params=muon_params, lr=config.lr, weight_decay=config.weight_decay, use_muon=True),
                    dict(
                        params=adamw_params,
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                        betas=(config.betas1, config.betas2),
                        use_muon=False,
                    ),
                ]
            )

            return optimizer
