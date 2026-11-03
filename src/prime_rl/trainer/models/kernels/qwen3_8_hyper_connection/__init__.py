"""Fused hyper-connection read and write for Qwen3.8 Flash Next: `kernels` holds the Triton kernels, `launch` their
launch configurations and host wrappers, `ops` the custom ops with their autograd (`fused_hc_mix`,
`fused_hc_combine_mix`)."""
