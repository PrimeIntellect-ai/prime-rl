def main() -> None:
    import torch
    from speculators.models import attention
    from speculators.train.cli import main as train
    from speculators.train.config import TrainConfig

    from prime_rl.specdecode.training import add_verifier_weight_aliases

    # Checkpoint recomputation can call attention outside the compiled model.
    attention.flex_attention = torch.compile(attention.flex_attention, dynamic=False)
    add_verifier_weight_aliases()
    train(TrainConfig.resolve())


if __name__ == "__main__":
    main()
