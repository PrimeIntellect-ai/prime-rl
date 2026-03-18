"""HMAC signing and verification for vLLM usage data.

The vLLM ASGI middleware signs token usage before the response leaves the
inference server. The orchestrator verifies the signature to detect tampering
by untrusted verifiers environments.
"""

import hashlib
import hmac


def sign_usage(
    secret: str,
    run_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    msg = f"{run_id}:{prompt_tokens}:{completion_tokens}"
    return hmac.new(
        secret.encode(),
        msg.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify_usage(
    secret: str,
    run_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    signature: str,
) -> bool:
    expected = sign_usage(secret, run_id, prompt_tokens, completion_tokens)
    return hmac.compare_digest(expected, signature)
