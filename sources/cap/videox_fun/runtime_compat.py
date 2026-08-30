import os
import sys


_ATTENTION_RUNTIME = None


def _disable_transformers_flash_attention():
    """Make Transformers treat a present but unusable flash-attn as unavailable."""
    import transformers
    from transformers.utils import import_utils

    def unavailable(*_args, **_kwargs):
        return False

    function_names = (
        "is_flash_attn_2_available",
        "is_flash_attn_greater_or_equal",
        "is_flash_attn_greater_or_equal_2_10",
    )
    for function_name in function_names:
        setattr(import_utils, function_name, unavailable)
        setattr(transformers.utils, function_name, unavailable)


def configure_attention_runtime():
    """Select FlashAttention when it imports cleanly, otherwise use PyTorch SDPA."""
    global _ATTENTION_RUNTIME
    if _ATTENTION_RUNTIME is not None:
        return _ATTENTION_RUNTIME

    requested_backend = os.environ.get("VIDEOX_ATTENTION_TYPE", "").strip().upper()
    flash_error = None
    if requested_backend == "SDPA":
        flash_available = False
        reason = "VIDEOX_ATTENTION_TYPE=SDPA"
    else:
        try:
            import flash_attn  # noqa: F401

            flash_available = True
            reason = "flash-attn import succeeded"
        except Exception as error:
            flash_available = False
            flash_error = error
            reason = f"flash-attn import failed: {type(error).__name__}: {error}"

    if not flash_available:
        os.environ["VIDEOX_ATTENTION_TYPE"] = "SDPA"
        _disable_transformers_flash_attention()

    _ATTENTION_RUNTIME = {
        "backend": "FLASH_ATTENTION" if flash_available else "SDPA",
        "flash_attn_available": flash_available,
        "reason": reason,
    }

    rank = os.environ.get("RANK", "0")
    if rank in {"", "0"}:
        message = f"CAP attention runtime: backend={_ATTENTION_RUNTIME['backend']} reason={reason}"
        print(message.replace("\n", " "), file=sys.stderr, flush=True)

    return _ATTENTION_RUNTIME
