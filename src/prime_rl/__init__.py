try:
    import prime_rl._compat  # noqa: F401 — must run before ring_flash_attn is imported
except ImportError:
    pass  # transformers not installed (slim/configs-only install)
