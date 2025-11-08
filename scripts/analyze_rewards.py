#!/usr/bin/env python3
"""
Analyze rewards from RL training run logs and rollouts.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def parse_log_rewards(log_file: Path):
    """Extract reward metrics from orchestrator logs."""
    rewards = []
    steps = []
    
    # Pattern to match: SUCCESS Step X | Time: ... | Reward: Y | ...
    pattern = r'SUCCESS Step (\d+).*?Reward: ([\d.]+)'
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    step = int(match.group(1))
                    reward = float(match.group(2))
                    steps.append(step)
                    rewards.append(reward)
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None
    
    return steps, rewards

def analyze_rollout_rewards(rollout_dir: Path):
    """Extract rewards from saved rollout files."""
    import torch
    
    all_rewards = []
    step_rewards = defaultdict(list)
    
    if not rollout_dir.exists():
        return None, None
    
    # Find all step directories
    step_dirs = sorted([d for d in rollout_dir.iterdir() if d.is_dir() and d.name.startswith('step_')])
    
    for step_dir in step_dirs:
        step_num = int(step_dir.name.split('_')[1])
        
        # Find all rank files
        rank_files = sorted([f for f in step_dir.iterdir() if f.name.startswith('rank_')])
        
        for rank_file in rank_files:
            try:
                data = torch.load(rank_file, map_location='cpu')
                # Rollouts are stored as batches, extract rewards
                if isinstance(data, list):
                    for batch in data:
                        if isinstance(batch, dict) and 'rewards' in batch:
                            batch_rewards = batch['rewards']
                            if isinstance(batch_rewards, list):
                                all_rewards.extend(batch_rewards)
                                step_rewards[step_num].extend(batch_rewards)
            except Exception as e:
                print(f"Error loading {rank_file}: {e}")
    
    return all_rewards, step_rewards

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_rewards.py <output_dir>")
        print("Example: python analyze_rewards.py outputs/dakota-v12-1000steps")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    log_file = output_dir / "logs" / "orchestrator.stdout"
    rollout_dir = output_dir / "rollouts"
    
    print("=" * 60)
    print("Reward Analysis")
    print("=" * 60)
    
    # Analyze logs
    print("\n1. Analyzing orchestrator logs...")
    steps, rewards = parse_log_rewards(log_file)
    
    if steps and rewards:
        print(f"   Found {len(rewards)} logged steps")
        print(f"   Step range: {min(steps)} - {max(steps)}")
        print(f"   Mean reward: {sum(rewards) / len(rewards):.4f}")
        print(f"   Min reward: {min(rewards):.4f}")
        print(f"   Max reward: {max(rewards):.4f}")
        print(f"   Latest reward (step {max(steps)}): {rewards[-1]:.4f}")
        
        # Show recent rewards
        if len(rewards) >= 10:
            print(f"\n   Last 10 rewards:")
            for step, reward in zip(steps[-10:], rewards[-10:]):
                print(f"     Step {step}: {reward:.4f}")
    else:
        print("   No rewards found in logs")
    
    # Analyze rollouts
    print("\n2. Analyzing rollout files...")
    all_rewards, step_rewards = analyze_rollout_rewards(rollout_dir)
    
    if all_rewards:
        print(f"   Found {len(all_rewards)} total rollouts")
        print(f"   Mean reward: {sum(all_rewards) / len(all_rewards):.4f}")
        print(f"   Min reward: {min(all_rewards):.4f}")
        print(f"   Max reward: {max(all_rewards):.4f}")
        
        if step_rewards:
            print(f"\n   Per-step statistics:")
            for step in sorted(step_rewards.keys())[-10:]:  # Last 10 steps
                step_reward_list = step_rewards[step]
                if step_reward_list:
                    print(f"     Step {step}: mean={sum(step_reward_list)/len(step_reward_list):.4f}, "
                          f"count={len(step_reward_list)}")
    else:
        print("   No rollout files found or unable to parse")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

