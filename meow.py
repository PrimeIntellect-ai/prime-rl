from pathlib import Path

from transformers import AutoTokenizer

from prime_rl.trainer.rl.packer import Packer
from prime_rl.trainer.runs import setup_runs


def main():
    setup_runs(Path("multi_out_dev"), 2)
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT")

    packer = Packer(dp_world_size=2, seq_len=2048, tokenizer=tokenizer)

    while True:
        rollouts = packer.get_batch()
        print(len(rollouts))
        packer.pack()
        input("Press Enter to continue...")

    # print(len(packer.get_batch()))
    # input("Press Enter to continue...")
    # print(len(packer.get_batch()))


if __name__ == "__main__":
    main()
