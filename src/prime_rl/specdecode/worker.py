def main() -> None:
    import torch
    from speculators.models import attention
    from speculators.train.cli import main as train
    from speculators.train.config import TrainConfig

    # Checkpoint recomputation can call attention outside the compiled model.
    attention.flex_attention = torch.compile(attention.flex_attention, dynamic=False)
    train(TrainConfig.resolve())


if __name__ == "__main__":
    main()
