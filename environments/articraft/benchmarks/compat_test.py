"""
Articraft + prime-rl 依赖兼容性测试。

在 KAOLA debug pod 上 setup 完成后运行：
  uv run python environments/articraft/benchmarks/compat_test.py \
      --articraft-dir /data/work/articraft

测试:
  1. 关键 import 不冲突（torch, vllm, manifold3d, trimesh, fcl, sdk）
  2. articraft compile 在 prime-rl venv 中正常工作
  3. scaffold compile 基线
  4. 从 records 采样 compile（如有数据）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def test_imports() -> dict[str, str]:
    """Test that all critical packages can be imported without conflict."""
    results: dict[str, str] = {}
    packages = [
        ("torch", "torch.__version__"),
        ("vllm", "vllm.__version__"),
        ("numpy", "numpy.__version__"),
        ("manifold3d", "'OK'"),
        ("trimesh", "trimesh.__version__"),
        ("fcl", "'OK'"),
        ("scipy", "scipy.__version__"),
        ("networkx", "networkx.__version__"),
        ("pydantic", "pydantic.__version__"),
        ("sdk", "'OK'"),
        ("sdk.v0", "'OK'"),
        ("agent.compiler", "'OK'"),
        ("agent.feedback", "'OK'"),
        ("agent.models", "'OK'"),
    ]

    for pkg, ver_expr in packages:
        try:
            mod = __import__(pkg)
            version = str(eval(ver_expr, {pkg.split(".")[0]: mod}))
            results[pkg] = f"OK ({version})"
        except Exception as e:
            results[pkg] = f"FAIL: {type(e).__name__}: {e}"

    return results


def test_torch_cuda() -> str:
    """Test CUDA availability and basic tensor ops."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "WARN: CUDA not available (expected on CPU debug pod)"
        device = torch.device("cuda:0")
        x = torch.randn(100, 100, device=device)
        y = x @ x.T
        return f"OK (device={torch.cuda.get_device_name(0)}, test matmul passed)"
    except Exception as e:
        return f"FAIL: {e}"


def test_scaffold_compile(articraft_dir: Path) -> str:
    """Compile the empty scaffold model.py."""
    try:
        sys.path.insert(0, str(articraft_dir))
        os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "0"

        from agent.compiler import compile_urdf_report

        scaffold_code = (articraft_dir / "scaffold.py").read_text()
        work_dir = Path(tempfile.mkdtemp(prefix="compat_scaffold_"))
        script_path = work_dir / "model.py"
        script_path.write_text(scaffold_code)

        t0 = time.perf_counter()
        try:
            report = compile_urdf_report(script_path, sdk_package="sdk")
            elapsed = time.perf_counter() - t0
            return f"OK ({elapsed:.2f}s, status={report.signal_bundle.status})"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            err_type = type(e).__name__
            msg = str(e)[:100]
            return f"EXPECTED FAIL ({elapsed:.2f}s, {err_type}: {msg})"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"


def test_record_compile(articraft_dir: Path) -> str:
    """Try compiling a real record's model.py if data is available."""
    records_dir = articraft_dir / "data" / "records"
    if not records_dir.exists():
        return "SKIP: no records directory"

    candidates = []
    for rec_dir in sorted(records_dir.iterdir()):
        if not rec_dir.name.startswith("rec_"):
            continue
        for rev_dir in sorted((rec_dir / "revisions").iterdir()) if (rec_dir / "revisions").exists() else []:
            mp = rev_dir / "model.py"
            if mp.exists():
                candidates.append(mp)
                break
        if len(candidates) >= 3:
            break

    if not candidates:
        return "SKIP: no model.py found in records"

    os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "0"
    from agent.compiler import compile_urdf_report

    results = []
    for mp in candidates:
        work_dir = Path(tempfile.mkdtemp(prefix="compat_rec_"))
        dest = work_dir / "model.py"
        shutil.copy2(mp, dest)

        rec_name = mp.parent.parent.parent.name[:50]
        t0 = time.perf_counter()
        try:
            report = compile_urdf_report(dest, sdk_package="sdk")
            elapsed = time.perf_counter() - t0
            status = report.signal_bundle.status
            n_signals = len(report.signal_bundle.signals)
            results.append(f"  {rec_name}: OK ({elapsed:.2f}s, {status}, {n_signals} signals)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append(f"  {rec_name}: FAIL ({elapsed:.2f}s, {type(e).__name__})")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    return "\n".join(results)


def test_signal_bundle_serialization() -> str:
    """Test CompileSignalBundle serialization round-trip."""
    try:
        from agent.models import CompileSignal, CompileSignalBundle
        bundle = CompileSignalBundle(
            status="success",
            summary="test",
            signals=(
                CompileSignal(
                    severity="warning", kind="test_warning", code="TEST",
                    summary="test signal", blocking=False,
                ),
            ),
        )
        d = bundle.to_dict()
        restored = CompileSignalBundle.from_dict(d)
        assert restored.status == "success"
        assert len(restored.signals) == 1
        assert restored.signals[0].code == "TEST"
        return "OK (round-trip passed)"
    except Exception as e:
        return f"FAIL: {e}"


def test_render_compile_signals() -> str:
    """Test render_compile_signals produces readable output."""
    try:
        from agent.feedback import render_compile_signals
        from agent.models import CompileSignal, CompileSignalBundle
        bundle = CompileSignalBundle(
            status="failure",
            summary="test failure",
            signals=(
                CompileSignal(
                    severity="failure", kind="compile_runtime", code="RUNTIME_ERROR",
                    summary="SyntaxError in model.py", blocking=True, source="compiler",
                    group="build",
                ),
            ),
        )
        text = render_compile_signals(bundle)
        assert "<compile_signals>" in text
        assert "RUNTIME_ERROR" in text or "failure" in text.lower()
        return f"OK ({len(text)} chars)"
    except Exception as e:
        return f"FAIL: {e}"


def main():
    parser = argparse.ArgumentParser(description="Articraft + prime-rl compatibility test")
    parser.add_argument("--articraft-dir", type=Path,
                        default=Path("/data/work/articraft"),
                        help="Path to articraft source directory")
    args = parser.parse_args()

    articraft_dir = args.articraft_dir.resolve()
    if not (articraft_dir / "sdk").is_dir():
        print(f"ERROR: {articraft_dir}/sdk not found. Is --articraft-dir correct?")
        sys.exit(1)

    if str(articraft_dir) not in sys.path:
        sys.path.insert(0, str(articraft_dir))

    print("=" * 60)
    print("  Articraft + prime-rl Compatibility Test")
    print("=" * 60)

    # 1. Imports
    print("\n--- 1. Import compatibility ---")
    import_results = test_imports()
    all_ok = True
    for pkg, result in import_results.items():
        status = "PASS" if result.startswith("OK") else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] {pkg}: {result}")

    # 2. CUDA
    print("\n--- 2. CUDA test ---")
    cuda_result = test_torch_cuda()
    print(f"  {cuda_result}")

    # 3. Signal bundle serialization
    print("\n--- 3. CompileSignalBundle serialization ---")
    serial_result = test_signal_bundle_serialization()
    print(f"  {serial_result}")

    # 4. render_compile_signals
    print("\n--- 4. render_compile_signals ---")
    render_result = test_render_compile_signals()
    print(f"  {render_result}")

    # 5. Scaffold compile
    print("\n--- 5. Scaffold compile ---")
    scaffold_result = test_scaffold_compile(articraft_dir)
    print(f"  {scaffold_result}")

    # 6. Record compile
    print("\n--- 6. Record compile (first 3) ---")
    record_result = test_record_compile(articraft_dir)
    print(f"{record_result}")

    # Summary
    print("\n" + "=" * 60)
    fail_count = sum(1 for r in import_results.values() if "FAIL" in r)
    if fail_count == 0:
        print("  RESULT: All import checks passed. No dependency conflicts detected.")
        print("  方案 A' (直接安装) 可行。")
    else:
        print(f"  RESULT: {fail_count} import(s) failed. Review above for conflicts.")
        print("  可能需要 fallback 到方案 C-Lite (服务化)。")
    print("=" * 60)


if __name__ == "__main__":
    main()
