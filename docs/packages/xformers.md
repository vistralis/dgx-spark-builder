# xformers

| | |
|---|---|
| **Repo** | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) |
| **Version** | 0.0.30+ |
| **Status** | 🔍 Blocked — SM 121 compile errors |

## What it does

Memory-efficient attention and transformer building blocks from Meta. Used by diffusers, ComfyUI, and many training pipelines for reduced VRAM usage.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ❓ Untested |
| SM 121 | ❌ Compile errors in CUTLASS kernels |
| aarch64 | ❓ No prebuilt wheels |
| torch 2.10 | ❓ |
| torch 2.11 | ❓ |

## Blocker

SM 121 triggers compile errors in xformers' CUTLASS-based attention kernels. The Dockerfile exists at `dockerfiles/builders/xformers/` but the build is disabled. Needs upstream CUTLASS updates for SM 12.x support.

## Next Steps

- Monitor xformers releases for SM 12.x / CUDA 13.0 support
- Test with `--gencode arch=compute_120,code=sm_120` only
