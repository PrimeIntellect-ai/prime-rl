import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, "tests/unit/train")
import test_compile as tc

cases = [dataclasses.replace(c, fullgraph=False, known_failure=None) for c in tc.select_cases(sys.argv[2:])]
tc.main(Path(sys.argv[1]), cases)
