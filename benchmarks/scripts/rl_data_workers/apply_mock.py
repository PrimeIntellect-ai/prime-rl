import shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
shutil.copy(Path(__file__).with_name("mock_mm.py"), root / "src/prime_rl/trainer/rl/mock_mm.py")
path = root / "src/prime_rl/trainer/rl/data.py"
text = path.read_text()
edits = [
    ("        self.generate_samples = config.generate_samples\n",
     "        self.generate_samples = config.generate_samples\n"
     "        self.mock_mm = mock_mm.MockMMMicroBatches(seq_len) if mock_mm.enabled() else None\n"),
    ("            get_micro_batch_fn = self._get_sample_micro_batch\n",
     "            get_micro_batch_fn = self._get_sample_micro_batch\n"
     "        if self.mock_mm is not None:\n"
     "            get_micro_batch_fn = self.mock_mm.micro_batch\n"),
    ("from prime_rl.trainer.world import get_world\n",
     "from prime_rl.trainer.rl import mock_mm\nfrom prime_rl.trainer.world import get_world\n"),
]
for old, new in edits:
    if text.count(old) != 1:
        sys.exit(f"anchor not unique in {path}: {old!r}")
    text = text.replace(old, new)
path.write_text(text)
print("patched", root.name)
