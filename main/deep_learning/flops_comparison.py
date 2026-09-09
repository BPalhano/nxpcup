import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def get_flops(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    input_shape = [1 if s is None else s for s in input_shape]

    @tf.function
    def forward(x):
        return model(x)

    concrete_func = forward.get_concrete_function(
        tf.TensorSpec(input_shape, model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph = frozen_func.graph

    with graph.as_default():
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd="op",
            options=opts
        )

    return flops.total_float_ops if flops is not None else 0


def get_model_size_mb(model_path):
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)

    total = 0
    for root, _, files in os.walk(model_path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def analyze_efficiency_tf(
    model_path,
    warmup_iters=20,
    measure_iters=300
):
    model = tf.keras.models.load_model(model_path, compile=False)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    input_shape = [1 if s is None else s for s in input_shape]
    dummy_input = tf.random.uniform(input_shape)

    # -------------------------
    # Warm-up
    # -------------------------
    for _ in range(warmup_iters):
        _ = model(dummy_input, training=False)

    latencies_ms = []

    for _ in range(measure_iters):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    latencies_ms = np.array(latencies_ms)

    stats = {
        "Params": model.count_params(),
        "FLOPs": get_flops(model),
        "Size (MB)": get_model_size_mb(model_path),
        "Latency mean (ms)": float(np.mean(latencies_ms)),
        "Latency median (ms)": float(np.median(latencies_ms)),
        "Latency std (ms)": float(np.std(latencies_ms)),
        "Latency p95 (ms)": float(np.percentile(latencies_ms, 95)),
        "Latency p99 (ms)": float(np.percentile(latencies_ms, 99)),
        "Latencies (ms)": latencies_ms
    }

    return stats

stats_orig = analyze_efficiency_tf("dave2_original.h5")
stats_lite = analyze_efficiency_tf("proposed_nn_512.h5")

print("Original:", stats_orig)
print("Proposed:", stats_lite)
print("\nCompare FLOPs e Latency para estimar ganho energético.")

flops_orig = stats_orig["FLOPs"]
flops_lite = stats_lite["FLOPs"]

print(f"FLOPs gain: {(flops_orig / flops_lite - 1) * 100:.2f}% reduction")

latency_orig = stats_orig["Latency mean (ms)"]
latency_lite = stats_lite["Latency mean (ms)"]

print(f"Latency gain: {(latency_orig / latency_lite - 1) * 100:.2f}% reduction")

print(f"Latency p95 gain: {(stats_orig['Latency p95 (ms)'] / stats_lite['Latency p95 (ms)'] - 1) * 100:.2f}% reduction")
print(f"Latency p99 gain: {(stats_orig['Latency p99 (ms)'] / stats_lite['Latency p99 (ms)'] - 1) * 100:.2f}% reduction")

print("\n latency std:{:.2f} ms vs {:.2f} ms".format(stats_orig["Latency std (ms)"], stats_lite["Latency std (ms)"]))

print("\nn p95 frames per second: {:.2f} vs {:.2f}".format(1000 / stats_orig["Latency p95 (ms)"], 1000 / stats_lite["Latency p95 (ms)"]))
print("\nn p99 frames per second: {:.2f} vs {:.2f}".format(1000 / stats_orig["Latency p99 (ms)"], 1000 / stats_lite["Latency p99 (ms)"]))

import matplotlib.pyplot as plt

plt.figure()
plt.hist(stats_orig["Latencies (ms)"], bins=40, alpha=0.6, label="DAVE-2")
plt.hist(stats_lite["Latencies (ms)"], bins=40, alpha=0.6, label="Proposed architecture")
plt.xlabel("Latency [ms]")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.boxplot(
    [stats_orig["Latencies (ms)"], stats_lite["Latencies (ms)"]],
    labels=["DAVE-2", "Proposed architecture"],
    showfliers=False
)

plt.ylabel("Latency (ms)")
plt.grid(True)
plt.show()

