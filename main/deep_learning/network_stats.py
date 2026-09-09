import time
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from contextlib import redirect_stdout

OUTPUT_FILE = "analysis_latency.txt"

def analyze_model(model_path, model_name):
    print(f"\n\n==============================")
    print(f"ANÁLISE DO MODELO: {model_name}")
    print(f"Arquivo: {model_path}")
    print(f"==============================\n")

    # ===============================
    # Carrega modelo
    # ===============================
    model = load_model(model_path, compile=False)

    # ===============================
    # 1. Número de parâmetros
    # ===============================
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )

    total_connections = trainable_params

    print("========== PARÂMETROS ==========")
    print(f"Total de parâmetros:        {total_params:,}")
    print(f"Parâmetros treináveis:      {trainable_params:,}")
    print(f"Parâmetros não-treináveis:  {non_trainable_params:,}")
    print(f"Total de conexões (pesos):  {total_connections:,}")
    print("================================\n")

    # ===============================
    # 2. Resumo do modelo
    # ===============================
    print("========== RESUMO DO MODELO ==========\n")
    model.summary()
    print("\n======================================\n")

    # ===============================
    # 3. Latência por camada
    # ===============================
    INPUT_SHAPE = (1, 200, 66, 3)  # batch=1
    N_RUNS = 100
    x_dummy = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    print("=== MEDIÇÃO DE LATÊNCIA POR CAMADA ===\n")

    prev_time = 0.0

    for i, layer in enumerate(model.layers):
        sub_model = Model(inputs=model.inputs, outputs=layer.output)

        # Warm-up
        for _ in range(10):
            _ = sub_model.predict(x_dummy, verbose=0)

        times = []

        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = sub_model.predict(x_dummy, verbose=0)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)

        layer_time = mean_time - prev_time
        prev_time = mean_time

        print(
            f"[{i:02d}] {layer.name:<25} "
            f"Total: {mean_time:7.2f} ms | "
            f"Layer: {layer_time:7.2f} ms | "
            f"Std: {std_time:6.2f} ms"
        )

    print("\n=== FIM DA ANÁLISE ===\n")


# ===============================
# EXECUÇÃO COM REDIRECIONAMENTO
# ===============================
with open(OUTPUT_FILE, "w") as f:
    with redirect_stdout(f):
        analyze_model("proposed_nn.h5", "REDE PROPOSTA")
        analyze_model("dave2_original.h5", "DAVE-2 ORIGINAL")
        analyze_model("proposed_nn_512.h5", "REDE PROPOSTA (512)")

print(f"\nAnálise concluída. Resultados salvos em: {OUTPUT_FILE}")
