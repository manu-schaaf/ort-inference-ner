# ONNX Runtime Downstream Task Benchmarks

## Setup

This benchmark evaluates the running time for a simple NER sequence labeling task using a fine-tuned BERT model.
The time of the following three Python and one Rust implementation are compared:

1. `baseline.py`: "naive" implementation, using regular batching and HuggingFace `transformers`.
2. `pipelines.py`: using the `transformers` sequence labeling pipeline.
3. `ort-hf.py`: using `onnxruntime` within the frame of the `baseline` implementation.
4. `onnxruntime-ner`: Rust implementation using `ort` and `rust_tokenizers`.

Prior to running the benchmark, the model must be [exported to ONNX](https://huggingface.co/docs/transformers/serialization#exporting-a--transformers-model-to-onnx-with-cli) and be placed in the `data/` folder.

Requires ONNX Runtime [v1.19.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.0).

## Benchmark

Benchmarks are conducted using hyperfine on a small corpus of 1.000 sentences:

```shell
hyperfine --warmup 1 -L device cpu,cuda \
'./target/release/onnxruntime-ner -d {device} data/test-1k.txt' \
'python src/python/baseline.py -d {device} data/test-1k.txt' \
'python src/python/pipelines.py -d {device} data/test-1k.txt' \
'python src/python/ort-hf.py -d {device} data/test-1k.txt'
```

## Results

### CPU

| Method | Time | Range |
| ------ | ---- | ----- |
| `baseline` | 14.348 s ±  0.107 s | 14.268 s … 14.577 s |
| `pipelines` | 15.610 s ±  0.071 s | 15.498 s … 15.731 s |
| `ort-hf` | 15.981 s ±  1.043 s | 15.168 s … 18.254 s |
| `onnxruntime-ner` | 19.066 s ±  0.129 s | 18.864 s … 19.258 s |

### CUDA

| Method | Time | Range |
| ------ | ---- | ----- |
| `baseline` | 5.601 s ±  0.095 s | 5.507 s …  5.834 s |
| `pipelines` | 6.773 s ±  0.051 s | 6.718 s …  6.889 s |
| `ort-hf` | 17.240 s ±  1.178 s | 15.729 s … 19.796 s |
| `onnxruntime-ner` | 2.473 s ±  0.014 s | 2.458 s …  2.496 s |

