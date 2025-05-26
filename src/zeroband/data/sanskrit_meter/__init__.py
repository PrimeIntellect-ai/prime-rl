"""Sanskrit meter data generation and loading utilities."""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "resources"
DATA_DIR.mkdir(exist_ok=True)

def get_data_path() -> Path:
    """Get the path to the Sanskrit meter data directory."""
    return DATA_DIR
