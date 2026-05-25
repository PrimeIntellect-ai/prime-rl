# bpy 技术选型

> 2026-04-30 决策记录：为什么用 Infinigen Blender 而不是 PyPI bpy。

## 结论

| 方案 | 安装 | OPTIX | .blend 加载 | 推荐 |
|------|------|-------|------------|------|
| **A** Infinigen Blender bin + socket | 已有 | OK (生产验证) | OK | **YES** |
| **B** PyPI bpy in 3.12 venv | FAIL (无 cp312 wheel) | — | — | no |
| **B'** PyPI bpy in 3.13 隔离 venv | OK | **FAIL** (OptiX 7804) | OK (70ms) | no |

> 决策：使用 Path A（Infinigen Blender 4.2 binary）。

---

## 验证过程

### Path B: 主 venv 安装 (Python 3.12)

```bash
uv add bpy
# → No solution: bpy 只有 cp313 wheel，项目锁定 ~=3.12.0
```

死路。除非升级 Python 或等 bpy 出 cp312 wheel。

### Path B': 隔离 3.13 venv

```bash
uv venv --python 3.13 /tmp/bpy-poc-venv
VIRTUAL_ENV=/tmp/bpy-poc-venv uv pip install bpy  # 380MB
```

OPTIX probe 结果:
```
cycles | WARNING OptiX initialization failed with error code 7804
# 静默回退到 CUDA，无 OPTIX 设备
```

> Error 7804 = `OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH`：PyPI bpy 5.1.1 内置的 OPTIX header 版本与系统 NVIDIA driver (560.35.03) 的 OPTIX runtime 不匹配。

`.blend` 加载正常（70ms），但渲染只能走 CUDA。

### Path A: Infinigen Blender 4.2 (当前生产)

```
Blender 4.2.0 (hash a51f293548ad built 2024-07-16)
[blendergym] cycles samples=16 denoiser=OPENIMAGEDENOISE compute=OPTIX
Time: 00:03.35
```

8x H20 集群上，OPTIX 健康，16spp 512x512 渲染 ~3.4s。

---

## 决策理由

- **性能差距**: OPTIX 比 CUDA 快 2-3x（indoor scenes）
- **修复成本**: 在 PyPI bpy 中恢复 OPTIX 需要自定义 runtime 或降级 bpy — 无底洞
- **Path A 优势**: 已验证 OPTIX + .blend 兼容性，daemon 化只需加 socket 协议 + 进程管理
- **无需迁移**: 不涉及 Python 版本升级、wheel 重编译、driver 协调
