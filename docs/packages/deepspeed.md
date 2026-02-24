# DeepSpeed

| | |
|---|---|
| **Repo** | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| **Version** | 0.16.x |
| **Status** | 🔍 Evaluating |

## What it does

Deep learning optimization library for training and inference. ZeRO optimizer for distributed training, inference kernels, MoE support.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ❓ Needs testing |
| SM 121 | ❓ CUDA kernels may need arch flags |
| aarch64 | ❓ Has had aarch64 issues historically |
| torch 2.10 | ❓ |
| torch 2.11 | ❓ |

## Considerations

- Primarily a **training** library — less relevant if only doing inference
- Heavy build (~2-3 hours) with many CUDA extension modules
- May require custom `DS_CUDA_ARCH` flags for SM 121
- Single-GPU DGX Spark limits the value of distributed training features
