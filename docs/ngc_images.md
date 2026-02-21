# NGC Container Image Inventory — DGX SPARK GB10

This document catalogs all NVIDIA NGC container images available for ML workloads
on the DGX SPARK GB10 (Blackwell, SM 121, aarch64).

> [!IMPORTANT]
> **25.11** images match the host driver (no forward-compatibility layer needed).
> **26.01** images use CUDA 13.1 but require forward-compatibility drivers.

---

## Quick Reference

| Image | Tag | CUDA | Size | PyTorch | TensorRT | ORT | Python | Use Case |
|-------|-----|------|------|---------|----------|-----|--------|----------|
| **pytorch** | 25.11-py3 | 13.0 | 19.5 GB | 2.10.0a0 | 10.14.1 | ❌ | 3.12 | Full ML training & inference |
| **pytorch** | 26.01-py3 | 13.1 | 19.9 GB | 2.10.0a0 | 10.14.1 | ❌ | 3.12 | Full ML training & inference |
| **tensorrt** | 25.11-py3 | 13.0 | 10.6 GB | ❌ | 10.14.1 | ❌ | 3.12 | TensorRT-only inference |
| **tensorrt** | 26.01-py3 | 13.1 | 10.8 GB | ❌ | 10.14.1 | ❌ | 3.12 | TensorRT-only inference |
| **cuda-dl-base** | 25.11-cuda13.0-devel | 13.0 | 9.4 GB | ❌ | ✅ (dev) | ❌ | ❌ | Custom builds (full toolchain) |
| **cuda-dl-base** | 25.11-cuda13.0-runtime | 13.0 | 8.4 GB | ❌ | ❌ | ❌ | ❌ | Minimal runtime |
| **cuda-dl-base** | 26.01-inference-devel | 13.1 | 5.2 GB | ❌ | ✅ (dev) | ❌ | ❌ | Inference builds (smallest devel) |
| **cuda-dl-base** | 26.01-inference-runtime | 13.1 | 4.7 GB | ❌ | ❌ | ❌ | ❌ | Minimal inference runtime |
| **vllm** | 25.11-py3 | 13.0 | — | ✅ | ✅ | — | 3.12 | LLM serving (vLLM engine) |
| **vllm** | 26.01-py3 | 13.1 | — | ✅ | ✅ | — | 3.12 | LLM serving (vLLM engine) |

---

## Choosing an Image

| Workload | Recommended Image |
|----------|-------------------|
| **Model training / finetuning** | `pytorch:26.01-py3` |
| **Model optimization (ModelOpt)** | `pytorch:26.01-py3` |
| **LLM serving** | `vllm:26.01-py3` |
| **TensorRT engine inference** | `tensorrt:26.01-py3` |
| **Custom lean inference** | `cuda-dl-base:26.01-inference-devel` |
| **Minimal production deploy** | `cuda-dl-base:26.01-inference-runtime` |
| **ComfyUI / pip-based stack** | `cuda13.0-torch2.10-ubuntu24.04` (custom base) |

> [!CAUTION]
> Both 25.11 and 26.01 ship the **same PyTorch 2.10.0a0**. The `torch.compile`
> `KernelMetadata` crash and `torch_tensorrt` failures are **PyTorch bugs**, not
> CUDA version issues. Fixed in PyTorch 2.11+.

---

*Last updated: 2026-02-21. Probed on DGX SPARK GB10 (aarch64).*
