#!/usr/bin/env python3
"""
Simple bottleneck reproduction script.
This script simulates the weight update bottleneck without needing a full training loop.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from openai import AsyncOpenAI
from prime_rl.orchestrator.client import update_weights, reload_weights, _admin_client, _trace_id
from prime_rl.utils.logger import setup_logger

logger = setup_logger("INFO")

async def simulate_streaming_load(client: AsyncOpenAI, duration: int = 60):
    """Simulate many concurrent streaming completions to saturate the server."""
    logger.info(f"Starting streaming load simulation for {duration} seconds...")

    tasks = []
    start_time = time.time()

    while time.time() - start_time < duration:
        # Create multiple streaming requests
        for _ in range(10):  # 10 concurrent streams
            task = asyncio.create_task(
                client.completions.create(
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    prompt="Write a long story about",
                    max_tokens=500,
                    stream=True,
                )
            )
            tasks.append(task)

        # Let them run for a bit
        await asyncio.sleep(2)

    logger.info(f"Streaming load completed. Created {len(tasks)} streaming requests.")

async def test_weight_updates_under_load(client: AsyncOpenAI):
    """Test weight update operations while server is under streaming load."""
    logger.info("Starting weight update test under load...")

    # Start streaming load in background
    load_task = asyncio.create_task(simulate_streaming_load(client, duration=30))

    # Wait for load to build up
    await asyncio.sleep(5)

    # Now try weight updates while server is saturated
    logger.info("Server should now be saturated with streaming requests. Testing weight updates...")

    for i in range(5):
        logger.info(f"\n=== Weight Update Test {i+1}/5 ===")

        # Try reload_weights (simpler operation)
        try:
            logger.info("Testing reload_weights...")
            await reload_weights(client)
            logger.info("reload_weights completed successfully")
        except Exception as e:
            logger.error(f"reload_weights failed: {e}")

        await asyncio.sleep(3)

    # Wait for streaming load to finish
    await load_task
    logger.info("Weight update test under load completed.")

async def test_weight_updates_no_load(client: AsyncOpenAI):
    """Test weight update operations without any load (baseline)."""
    logger.info("Starting baseline weight update test (no load)...")

    for i in range(5):
        logger.info(f"\n=== Baseline Weight Update {i+1}/5 ===")

        try:
            logger.info("Testing reload_weights (no load)...")
            await reload_weights(client)
            logger.info("reload_weights completed successfully")
        except Exception as e:
            logger.error(f"reload_weights failed: {e}")

        await asyncio.sleep(2)

    logger.info("Baseline weight update test completed.")

async def main():
    logger.info("=" * 80)
    logger.info("Bottleneck Reproduction Test")
    logger.info("=" * 80)

    # Setup client
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1/",  # Note trailing slash to avoid double-slash in _server_base_from_oai
        api_key="EMPTY",
    )

    # Wait for server to be ready
    logger.info("Waiting for server to be ready...")
    for i in range(30):
        try:
            models = await client.models.list()
            logger.info(f"Server is ready. Models: {[m.id for m in models.data]}")
            break
        except Exception as e:
            if i == 29:
                logger.error(f"Server did not become ready after 30 attempts: {e}")
                return
            await asyncio.sleep(2)

    # Test 1: Baseline (no load)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Baseline Weight Updates (No Load)")
    logger.info("=" * 80)
    await test_weight_updates_no_load(client)

    await asyncio.sleep(5)

    # Test 2: Under load (should show bottleneck)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Weight Updates Under Streaming Load")
    logger.info("Expected: High queue_ms due to shared connection pool saturation")
    logger.info("=" * 80)
    await test_weight_updates_under_load(client)

    logger.info("\n" + "=" * 80)
    logger.info("Test completed! Check logs for [weights] entries.")
    logger.info("Look for queue_ms values:")
    logger.info("  - Baseline: should be <100ms")
    logger.info("  - Under load: may be >1000ms if bottleneck exists")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
