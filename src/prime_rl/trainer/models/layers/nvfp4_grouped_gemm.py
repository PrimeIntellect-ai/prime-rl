from prime_rl.trainer.distributed.expert_parallel import set_token_group_alignment_size_m
from prime_rl.utils.logger import get_logger

_NVFP4_TOKEN_GROUP_ALIGNMENT = 32


def prepare_nvfp4_grouped_gemm() -> None:
    from prime_rl_kernels.nvfp4 import prepare_for_compile

    prepare_for_compile()
    set_token_group_alignment_size_m(_NVFP4_TOKEN_GROUP_ALIGNMENT)

    get_logger().info(f"Prepared NVFP4 grouped GEMM (token_group_alignment={_NVFP4_TOKEN_GROUP_ALIGNMENT})")
