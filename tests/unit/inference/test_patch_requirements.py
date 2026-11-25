from prime_rl.inference import patches
from prime_rl.inference.patches import _missing_in_vllm, requires_vllm


def test_missing_in_vllm_checks_modules_and_attributes():
    assert not _missing_in_vllm("vllm.v1.core.sched.scheduler")
    assert not _missing_in_vllm("vllm.v1.core.sched.scheduler:Scheduler")
    assert _missing_in_vllm("vllm.v1.core.sched.scheduler:NoSuchName")
    # A missing parent package must not raise, and neither must a parent that is a module.
    assert _missing_in_vllm("vllm.no_such_package.module")
    assert _missing_in_vllm("vllm.no_such_package.module:name")
    assert _missing_in_vllm("vllm.v1.kv_cache_interface.no_such_submodule")


def test_patch_is_skipped_only_when_a_requirement_is_missing():
    applied = []

    @requires_vllm("vllm.v1.core.sched.scheduler:Scheduler", "vllm.no_such_package.module")
    def patch_of_a_moved_internal():
        applied.append("moved")

    @requires_vllm("vllm.v1.core.sched.scheduler:Scheduler")
    def patch_of_a_present_internal():
        applied.append("present")

    patch_of_a_moved_internal()
    patch_of_a_present_internal()
    assert applied == ["present"]


def test_no_patch_is_skipped_on_the_pinned_vllm():
    """The pinned release has everything the guarded patches name, so every one of them applies there."""
    guarded = {
        name: patch.requirements
        for name, patch in vars(patches).items()
        if callable(patch) and hasattr(patch, "requirements")
    }
    assert guarded
    missing = {name: [r for r in requirements if _missing_in_vllm(r)] for name, requirements in guarded.items()}
    assert not any(missing.values()), missing
