# "http://151.185.32.59:8000/v1",
# "http://151.185.33.12:8000/v1",
import argparse
import asyncio
import itertools
import time

from openai import AsyncOpenAI
from openai.resources import AsyncChat, AsyncCompletions
from verifiers import load_environment

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.client import setup_client


class RoundRobinAsyncOpenAI(AsyncOpenAI):
    def __init__(self, clients: list[AsyncOpenAI]):
        self.clients = clients
        self.cycle = itertools.cycle(clients)

    def _next_client(self) -> AsyncOpenAI:
        return next(self.cycle)

    @property
    def chat(self) -> AsyncChat:
        """Round-robin chat requests"""
        return AsyncChat(self._next_client())

    @property
    def completions(self) -> AsyncCompletions:
        """Round-robin completion requests"""
        return AsyncCompletions(self._next_client())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-urls", "-b", nargs="+", type=str)
    parser.add_argument("--num-examples", "-n", type=int, default=5)
    parser.add_argument("--rollouts-per-example", "-r", type=int, default=3)
    parser.add_argument("--max-tokens", "-t", type=int, default=128)
    parser.add_argument("--max-concurrent", "-c", type=int, default=-1)
    args = parser.parse_args()

    env = load_environment("math500", use_think=True)
    clients = [setup_client(ClientConfig(base_url=base_url)) for base_url in args.base_urls]
    client = RoundRobinAsyncOpenAI(clients=clients)

    print(
        f"Evaluating {args.num_examples=}, {args.rollouts_per_example=}, {args.max_tokens=}, {args.max_concurrent=}..."
    )
    start_time = time.time()
    sampling_args = dict(max_tokens=args.max_tokens)
    env.evaluate(
        client=client,
        model="zai-org/GLM-4.5-Air",
        sampling_args=sampling_args,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
    )
    end_time = time.time()
    time_taken = end_time - start_time
    tokens_processed = args.num_examples * args.rollouts_per_example * args.max_tokens
    print(f"Throughput: {tokens_processed / time_taken:.1f} tok/sec ({tokens_processed} in {time_taken:.1f} seconds)")


if __name__ == "__main__":
    asyncio.run(main())
