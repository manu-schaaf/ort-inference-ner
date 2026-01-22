# ONNX Runtime Downstream Task Benchmarks

## Setup

This benchmark evaluates the running time for a simple NER sequence labeling task using a fine-tuned DistilBERT model.
The time of the following three Python and one Rust implementation are compared:

1. `baseline.py`: "naive" implementation, using regular batching and HuggingFace `transformers`.
2. `pipelines.py`: using the `transformers` sequence labeling pipeline.
3. `ort-hf.py`: using `onnxruntime` with Python bindings within the frame of the `baseline` implementation.
4. `onnxruntime-ner`: Rust implementation using `ort` and `rust_tokenizers`.

Prior to running the benchmark, the model must be [exported to ONNX](https://huggingface.co/docs/transformers/serialization#exporting-a--transformers-model-to-onnx-with-cli) and be placed in the `data/` folder.

Requires ONNX Runtime [v1.19.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.0).

## Benchmark

Benchmarks are conducted using hyperfine on a small corpus of 1.000 sentences:

```shell
hyperfine --warmup 1 -L device cpu,cuda -L bs 1,8,32,64,128 \
'./target/release/onnxruntime-ner -d {device} -b {bs} data/test-1k.txt' \
'python src/python/baseline.py -d {device} -b {bs} data/test-1k.txt' \
'python src/python/pipelines.py -d {device} -b {bs} data/test-1k.txt' \
'python src/python/ort-hf.py -d {device} -b {bs} data/test-1k.txt'
```

### Results

#### CPU

| Method | Time | Range |
| ------ | ---- | ----- |
| `baseline.py -b 8` | 12.292 s ±  0.165 s | 11.910 s … 12.437 s |
| `pipelines.py -b 8` | 12.616 s ±  0.100 s | 12.494 s … 12.848 s |
| `ort-hf.py -b 1` | 11.129 s ±  0.895 s | 10.421 s … 12.553 s |
| `onnxruntime-ner -b 128`[^1] | 19.066 s ±  0.129 s | 18.864 s … 19.258 s |

#### CUDA

| Method | Time | Range |
| ------ | ---- | ----- |
| `baseline.py -b 8` | 6.209 s ±  0.041 s | 6.170 s …  6.291 s |
| `pipelines.py -b 8` | 7.170 s ±  0.043 s | 7.112 s …  7.247 s |
| `ort-hf.py -b 32` | 4.597 s ±  0.011 s | 4.581 s … 4.619 s |
| `onnxruntime-ner -b 128`[^1] | 2.473 s ±  0.014 s | 2.458 s …  2.496 s |

#### Notes

The benchmark was run on a small workstation with an Intel i7-8700 CPU and a Nvidia 1660 Ti GPU.
Results above are given for batch inference with batch size `-b N`, where `-b 1` indicates no batching.
Batching usually reduces inference time by at least 1s, regardless of the backend choice.
However, different batch sizes have an inconsistent effect on `ort-hf.py`: for the CPU backend only, larger batch sizes result in a substantial _runtime increase_ of about 5s.

[^1]: `transformers` times have been updated in 2026, but `onnxruntime-ner` times are ~2y old and were measured with `ort=2.0.0-rc.8`.
