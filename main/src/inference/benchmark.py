"""Rigorous Edge AI Inference Benchmarking for NXP Cup Autonomous Driving.

Measures:
- Warm-up latency (50 iterations)
- High-precision inference latency (500 timed runs via perf_counter_ns)
- Statistical metrics: P50 (Median), P95, P99, Mean, Std, Throughput (FPS)
- Compares PyTorch vs ONNX Runtime across DAVE-2, PalhanoCNN, and MicroViT
- Generates Violin plots and Throughput bar charts saved to main/training_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.src.inference.strategy import create_inference_engine  # noqa: E402
from main.src.models import ModelFactory, count_parameters  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.benchmark")


def benchmark_engine(
    engine: Any,
    dummy_input: np.ndarray,
    warmup_iters: int = 50,
    measure_iters: int = 500,
) -> dict[str, Any]:
    """Runs high-precision timing benchmark on an inference engine."""
    # 1. Warm-up
    for _ in range(warmup_iters):
        _ = engine.infer_batch(dummy_input)

    # 2. Timed measurements
    latencies_ms = np.empty(measure_iters, dtype=np.float64)

    for i in range(measure_iters):
        t0 = time.perf_counter_ns()
        _ = engine.infer_batch(dummy_input)
        t1 = time.perf_counter_ns()
        latencies_ms[i] = (t1 - t0) / 1_000_000.0

    mean_lat = float(np.mean(latencies_ms))
    median_p50 = float(np.median(latencies_ms))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    std_lat = float(np.std(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(median_p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "std_ms": round(std_lat, 3),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
        "fps": round(fps, 1),
        "raw_latencies": latencies_ms.tolist(),
    }


def plot_benchmark_results(
    results: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generates comparison violin plot and throughput bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Latency Violin Plot
    violin_path = output_dir / "latency_violin_plot.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    data_to_plot = []
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    for key, stats in results.items():
        labels.append(key)
        data_to_plot.append(stats["raw_latencies"])

    parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
    bodies: Any = parts["bodies"]
    for i, pc in enumerate(bodies):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(
        "NXP Cup Inference Latency Distribution (P50 / P95 / P99)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(violin_path, dpi=300)
    plt.close()

    # 2. Throughput & Parameters Bar Chart
    throughput_path = output_dir / "throughput_comparison.png"
    fig, ax1 = plt.subplots(figsize=(10, 5))

    names = list(results.keys())
    fps_values = [results[k]["fps"] for k in names]

    bars = ax1.bar(names, fps_values, color="#2ca02c", width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_ylabel(
        "Throughput (Frames Per Second)", color="#2ca02c", fontsize=12, fontweight="bold"
    )
    ax1.set_title("Inference Throughput (FPS) by Engine Backend", fontsize=14, fontweight="bold")
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.0,
            f"{yval:.1f} FPS",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(throughput_path, dpi=300)
    plt.close()

    return violin_path, throughput_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NXP Cup models across inference backends"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=300, help="Benchmark measurement iterations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MAIN_DIR / "training_results"),
        help="Directory to save benchmark reports and plots",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_batch = np.random.randn(1, 3, 66, 200).astype(np.float32)

    configurations = [
        ("micro_vit_smoothed", "onnx", out_dir / "micro_vit" / "micro_vit.onnx"),
        ("micro_vit", "onnx", out_dir / "micro_vit" / "micro_vit.onnx"),
        ("micro_vit", "pytorch", out_dir / "micro_vit" / "best_model.pt"),
        ("palhano_cnn", "onnx", out_dir / "palhano_cnn" / "palhano_cnn.onnx"),
        ("palhano_cnn", "pytorch", out_dir / "palhano_cnn" / "best_model.pt"),
        ("dave2", "onnx", out_dir / "dave2" / "dave2.onnx"),
        ("dave2", "pytorch", out_dir / "dave2" / "best_model.pt"),
    ]

    benchmark_summary: dict[str, Any] = {}

    logger.info(
        "Starting edge inference benchmarks (%d warmup, %d runs)...", args.warmup, args.iters
    )

    for m_name, backend, model_file in configurations:
        label = f"{m_name}_{backend.upper()}"
        logger.info("Benchmarking: %s...", label)

        try:
            apply_smooth = "smoothed" in m_name
            if backend == "onnx":
                if not model_file.exists():
                    logger.warning("ONNX file %s missing, skipping.", model_file)
                    continue
                engine = create_inference_engine(
                    backend="onnx",
                    model_name_or_path=str(model_file),
                    apply_smoothing=apply_smooth,
                )
            else:
                ckpt = str(model_file) if model_file.exists() else None
                engine = create_inference_engine(
                    backend="pytorch",
                    model_name_or_path=m_name,
                    checkpoint_path=ckpt,
                    apply_smoothing=apply_smooth,
                )

            stats = benchmark_engine(
                engine, dummy_batch, warmup_iters=args.warmup, measure_iters=args.iters
            )
            # Count params
            dummy_m = ModelFactory.create(m_name)
            stats["params"] = count_parameters(dummy_m)["total"]
            stats["backend"] = backend.upper()
            stats["model"] = m_name

            benchmark_summary[label] = stats
            logger.info(
                "  -> %s: P50 = %.2f ms | P95 = %.2f ms | Mean = %.2f ms | FPS = %.1f",
                label,
                stats["p50_ms"],
                stats["p95_ms"],
                stats["mean_ms"],
                stats["fps"],
            )
        except Exception as err:
            logger.error("Failed to benchmark %s: %s", label, err)

    # Plot results
    v_path, t_path = plot_benchmark_results(benchmark_summary, out_dir)

    # Clean raw latencies from saved JSON for compact storage
    json_summary = {}
    for k, v in benchmark_summary.items():
        item = dict(v)
        item.pop("raw_latencies", None)
        json_summary[k] = item

    json_path = out_dir / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    logger.info("✅ Benchmark complete! Artifacts saved:")
    logger.info("  Report JSON : %s", json_path)
    logger.info("  Violin Plot : %s", v_path)
    logger.info("  Throughput  : %s", t_path)


if __name__ == "__main__":
    main()
