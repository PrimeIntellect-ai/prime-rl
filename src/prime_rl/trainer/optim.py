from dion import Muon
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.trainer.config import OptimizerConfigType


def setup_optimizer(config: OptimizerConfigType, model: nn.Module, device_mesh: DeviceMesh) -> Optimizer:
    match config.type:
        case "sgd":
            return SGD(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )
        case "muon":

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
                    dict(
                        params=muon_params,
                        algorithm="muon",
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                        adjust_lr="rms_norm",
                    ),
                    dict(params=adamw_params, algorithm="adamw", lr=config.lr, weight_decay=config.weight_decay),
                ],
                lr=config.lr,
                weight_decay=config.weight_decay,
                adjust_lr="rms_norm",
                distributed_mesh=device_mesh,
            )

            return optimizer
