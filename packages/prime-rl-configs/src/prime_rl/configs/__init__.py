import os
import sys

# Model Factory forwards no environment variables to pods. A run whose volume carries Python packages
# (/volume/packages) or a Hugging Face cache (/volume/hf/hub) gets them here, before any config names a plugin.
if os.path.isdir("/volume/packages") and "/volume/packages" not in sys.path:
    sys.path.append("/volume/packages")
if os.path.isdir("/volume/hf/hub"):
    os.environ["HF_HUB_CACHE"] = "/volume/hf/hub"
    if "huggingface_hub.constants" in sys.modules:
        sys.modules["huggingface_hub.constants"].HF_HUB_CACHE = "/volume/hf/hub"
