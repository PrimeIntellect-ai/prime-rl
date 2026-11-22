#!/bin/bash
# Usage: stage_data.sh DATA_DIR. Downloads the chat and if splits of INTELLECT-3-SFT-10K for the CP arms, which run with the HF hub offline.
set -euo pipefail
hf download PrimeIntellect/INTELLECT-3-SFT-10K --repo-type dataset \
  --include "data/chat-*" --include "data/if-*" --local-dir "$1/INTELLECT-3-SFT-10K"
