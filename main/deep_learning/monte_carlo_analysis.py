import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from contextlib import redirect_stdout

# =====================================
# FUNÇÃO PRINCIPAL DE ANÁLISE
# =====================================
def analyze_model(
    model_path,
    model_name,
    output_prefix,
    n_seeds=1000,
    n_warmup=20,
    input_shape=(1, 200, 66, 3)
):
    print(f"\nAnalisando modelo: {model_name}")

    # -------------------------------
    # Carrega modelo
    # -------------------------------
    model = load_model(model_path, compile=False)

    # -------------------------------
    # Warm-up
    # -------------------------------
    x_warm = np.random.rand(*input_shape).astype(np.float32)
    for _ in range(n_warmup):
        _ = model.predict(x_warm, verbose=0)

    # -------------------------------
    # Monte Carlo
    # -------------------------------
    latencies = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x = np.random.rand(*input_shape).astype(np.float32)

        t0 = time.perf_counter()
        _ = model.predict(x, verbose=0)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)

    # -------------------------------
    # Salva vetor
    # -------------------------------
    npy_file = f"{output_prefix}.npy"
    np.save(npy_file, latencies)

    # -------------------------------
    # Estatísticas
    # -------------------------------
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    # -------------------------------
    # Salva TXT
    # -------------------------------
    txt_file = f"{output_prefix}.txt"

    with open(txt_file, "w") as f:
        with redirect_stdout(f):
            print("=======================================")
            print(" ANÁLISE DE LATÊNCIA — MONTE CARLO")
            print("=======================================\n")

            print(f"Modelo: {model_name}")
            print(f"Arquivo do modelo: {model_path}\n")

            print(f"Número de execuções (seeds): {n_seeds}")
            print(f"Formato de entrada: {input_shape}")
            print(f"Batch size: 1\n")

            print("====== RESULTADOS ======")
            print(f"Delay médio total:      {mean_latency:.2f} ms")
            print(f"Desvio padrão:          {std_latency:.2f} ms")
            print(f"Delay mínimo:           {min_latency:.2f} ms")
            print(f"Delay máximo:           {max_latency:.2f} ms")
            print("========================\n")

            print("Observações:")
            print("- Inferência em batch=1")
            print("- Sem quantização, pruning ou otimizações")
            print("- Latência medida com time.perf_counter()")
            print("- Warm-up aplicado antes do Monte Carlo")
            print("- Vetor completo salvo para análise posterior\n")

            print("Arquivos gerados:")
            print(f"- {npy_file}")
            print(f"- {txt_file}")

    print(f"Vetor salvo em: {npy_file}")
    print(f"Relatório salvo em: {txt_file}")


# =====================================
# MAIN
# =====================================
def main():
    analyze_model(
        "dave2_original.h5",
        "DAVE-2 ORIGINAL",
        "latency_mc_dave2"
    )

    #analyze_model(
    #    "proposed_nn.h5",
    #    "REDE PROPOSTA",
    #    "latency_mc_proposed"
    #s)

    analyze_model(
        "proposed_nn_512.h5",
        "REDE PROPOSTA (512)",
        "latency_mc_proposed_512"
    )


if __name__ == "__main__":
    main()
