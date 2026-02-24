# llama.cpp

| | |
|---|---|
| **Repo** | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **Version** | latest |
| **Status** | 🔍 Evaluating |

## What it does

Efficient LLM inference in C/C++ with GGUF model format. CPU + GPU hybrid execution, quantized model support (Q4, Q5, Q8, etc.). Python bindings via `llama-cpp-python`.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ Works with `-DGGML_CUDA=ON` |
| SM 121 | ✅ Verified (see DGX Spark knowledge base) |
| aarch64 | ✅ Native support |
| torch 2.10 | N/A (independent of PyTorch) |
| torch 2.11 | N/A |

## Considerations

- **Different ecosystem** — GGUF models, not HuggingFace format
- Already works well via Ollama on DGX Spark (verified: ~45W GPU, 2418 MHz sustained)
- Python bindings (`llama-cpp-python`) would need a wheel build
- VMM support enables ~160 GB GGUF models on 128 GB system
- Low priority — Ollama already provides a convenient interface
