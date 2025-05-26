"""Generate training data for Sanskrit meter generation."""
from __future__ import annotations

import json
import itertools
import random
from pathlib import Path

from zeroband.data.sanskrit_meters import get_data_path
from zeroband.data.sanskrit_meters.constants import TOPICS, METERS, METER_PATTERNS

def format_prompt(entry: dict) -> str:
    """Format a prompt entry into a complete instruction."""
    meter = entry['meter']
    topic = entry['topic']
    pattern = METER_PATTERNS[meter]['pattern']
    syllables = METER_PATTERNS[meter]['syllables_per_line']
    lines = METER_PATTERNS[meter]['lines_per_verse']
    
    return f"""\
Please write a Sanskrit poem in {meter} meter on the theme of "{topic}".
The meter pattern is {pattern} with {syllables} syllables per line and {lines} lines.
Return only the poem in Devanagari script, without any explanation."""

def generate_dataset(output_path: Path | None = None) -> list[dict]:
    """
    Generate the complete dataset of Sanskrit meter prompts.
    
    Args:
        output_path: Optional path to save the dataset. If None, uses default location.
        
    Returns:
        List of prompt entries
    """
    # Create all combinations of topics and meters
    base_entries = [{"topic": t, "meter": m}
                   for t, m in itertools.product(TOPICS, METERS)]
    random.shuffle(base_entries)
    
    # Add formatted prompts and meter information
    entries = []
    for entry in base_entries:
        meter = entry['meter']
        entries.append({
            **entry,
            "prompt": format_prompt(entry),
            "pattern": METER_PATTERNS[meter]['pattern'],
            "syllables_per_line": METER_PATTERNS[meter]['syllables_per_line'],
            "lines_per_verse": METER_PATTERNS[meter]['lines_per_verse']
        })
    
    # Save if path provided
    if output_path is None:
        output_path = get_data_path() / "prompts.jsonl"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✔ Generated {len(entries)} prompts and saved to {output_path}")
    return entries

def main() -> None:
    """Generate and save the dataset."""
    generate_dataset()

if __name__ == "__main__":
    main()
