
from importlib.metadata import distribution as _vllm_distribution
__path__.append(str(_vllm_distribution("vllm").locate_file("vllm/distributed/device_communicators")))
