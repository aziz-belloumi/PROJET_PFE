# Ressources benchmark notes

This benchmark mirrors the resource-monitoring approach used in the Testing_Models project: it measures CPU and GPU memory growth, load time, and average inference latency per document for each task/model/language pair.

## What is measured
- `load_time_sec`: time needed to initialize the model or pipeline
- `total_inf_time_sec`: total time spent processing all texts in a language bucket
- `avg_ms_per_doc`: average latency per document
- `peak_cpu_mb` / `peak_gpu_mb`: memory increase during inference relative to the baseline

## Best practice
- Run the benchmark on CPU first to estimate a stable baseline.
- Re-run on GPU when CUDA is available to compare acceleration.
- Keep sample counts limited during exploratory runs; the full benchmark can be expensive for large datasets.

## Important caveat
The Qwen benchmark is intentionally lightweight: the script starts an Ollama-style generation request and measures local Python-side resource consumption. In practice, the actual model lifecycle can be handled externally, so the reported GPU footprint may appear lower than the full server-side footprint.

## Suggested usage
```bash
python scripts/ressources/master_benchmark.py
```

The generated CSV is stored in this directory as `resource_usage_report.csv`.
