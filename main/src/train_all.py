"""Orchestrator to train all 3 NXP Cup models sequentially and generate a consolidated report.

Outputs are cleanly organized in:
  main/training_results/
    ├── micro_vit/
    │   ├── best_model.pt
    │   ├── evaluation_summary.json
    │   ├── learning_curves.png
    │   ├── steer_scatter.png
    │   └── residual_distribution.png
    ├── palhano_cnn/
    │   └── ...
    ├── dave2/
    │   └── ...
    ├── model_comparison_report.json
    └── COMPARISON_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.train_all")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = PROJECT_ROOT / "main"


def generate_comparison_summary(
    results: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generates a clean Markdown comparison summary table."""
    md_path = output_dir / "COMPARISON_SUMMARY.md"
    content = [
        "# 🏎️ NXP Cup — Consolidated Model Training & Evaluation Summary",
        "",
        f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Directory:** `{output_dir.relative_to(PROJECT_ROOT)}`",
        "",
        "---",
        "",
        "## 📊 Benchmark Comparativo de Desempenho",
        "",
        "| Modelo | Parâmetros | Época de Melhor Val | Val MAE (°) | Test MAE (°) | Test RMSE (°) | Test $R^2$ Score | Max Error (°) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, data in results.items():
        if "error" in data:
            content.append(f"| **{model_name}** | - | - | ERRO | ERRO | ERRO | ERRO | ERRO |")
            continue

        params = data.get("parameters", {}).get("total", 0)
        best_epoch = data.get("best_epoch", "-")
        best_val_mae = data.get("best_val_mae_deg", 0.0)
        test_m = data.get("test_metrics", {})
        test_mae = test_m.get("mae_deg", 0.0)
        test_rmse = test_m.get("rmse_deg", 0.0)
        test_r2 = test_m.get("r2", 0.0)
        test_max_err = test_m.get("max_err_deg", 0.0)

        content.append(
            f"| **{model_name}** | {params:,} | {best_epoch} | {best_val_mae:.2f}° | "
            f"**{test_mae:.2f}°** | {test_rmse:.2f}° | {test_r2:.3f} | {test_max_err:.2f}° |"
        )

    content.extend(
        [
            "",
            "---",
            "",
            "## 📂 Estrutura de Arquivos Gerada",
            "",
            "```text",
            "main/training_results/",
            "├── micro_vit/",
            "│   ├── best_model.pt               # Pesos ótimos PyTorch",
            "│   ├── evaluation_summary.json     # Métricas JSON",
            "│   ├── learning_curves.png         # Evolução da Huber Loss & MAE",
            "│   ├── steer_scatter.png           # Dispersão Predição vs Ground Truth",
            "│   └── residual_distribution.png   # Histograma de resíduos",
            "├── palhano_cnn/",
            "│   └── [mesma estrutura]",
            "├── dave2/",
            "│   └── [mesma estrutura]",
            "├── model_comparison_report.json    # Relatório JSON agregado",
            "└── COMPARISON_SUMMARY.md           # Este relatório",
            "```",
        ]
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")

    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all NXP Cup models sequentially")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per model")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MAIN_DIR / "training_results"),
        help="Root training results directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["micro_vit", "palhano_cnn", "dave2"]
    results: dict[str, Any] = {}

    logger.info("Starting sequential training for models: %s", models)
    logger.info("Target directory: %s", output_dir)

    for model_name in models:
        logger.info("==================================================")
        logger.info(">>> STARTING TRAINING: %s (%d epochs)", model_name, args.epochs)
        logger.info("==================================================")

        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--output-dir",
            str(output_dir),
        ]

        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(MAIN_DIR), capture_output=False)
        elapsed = time.time() - t0

        if res.returncode != 0:
            logger.error(
                "Training failed for model %s (return code: %d)", model_name, res.returncode
            )
            results[model_name] = {"error": f"Process exited with code {res.returncode}"}
            continue

        logger.info(">>> COMPLETED %s in %.1f minutes", model_name, elapsed / 60.0)

        # Load generated summary
        summary_file = output_dir / model_name / "evaluation_summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                results[model_name] = json.load(f)
        else:
            results[model_name] = {"warning": "evaluation_summary.json not found"}

    # Save consolidated JSON report
    consolidated_json = output_dir / "model_comparison_report.json"
    with open(consolidated_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown report
    summary_md = generate_comparison_summary(results, output_dir)
    logger.info("Consolidated comparison report saved to: %s", summary_md)
    logger.info("All model trainings completed successfully!")


if __name__ == "__main__":
    main()
