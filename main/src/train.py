"""Unified Training & Evaluation Pipeline for NXP Cup Autonomous Driving.

Supports:
- Model architectures: DAVE-2, PalhanoCNN, MicroViT via ModelFactory.
- Huber Loss on normalized steering angles with metric conversion to physical degrees.
- AdamW optimizer with Cosine Annealing Learning Rate scheduler.
- Comprehensive MLflow tracking (hyperparameters, epoch metrics, best model, evaluation plots).
- Evaluation artifacts: Residual distribution, GT vs. Pred scatter, Loss & MAE learning curves.
- Fast dry-run mode for CI/CD and pipeline verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Configure MLflow environment defaults
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Add workspace and main to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.data.dataset import get_dataloaders  # noqa: E402
from main.src.models import (  # noqa: E402
    ModelFactory,
    count_parameters,
    get_model_summary_str,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.train")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    scale_deg: float = 30.0,
) -> dict[str, float]:
    """Calculates regression metrics in normalized units and physical degrees.

    Args:
        predictions: Model predictions in [-1.0, 1.0].
        targets: Ground truth target angles in [-1.0, 1.0].
        scale_deg: Multiplier to convert normalized units to degrees.

    Returns:
        Dictionary containing MAE, RMSE, Max Error in degrees, and R^2 score.
    """
    preds = np.asarray(predictions).flatten()
    gts = np.asarray(targets).flatten()

    diff_norm = preds - gts
    mae_norm = float(np.mean(np.abs(diff_norm)))
    rmse_norm = float(np.sqrt(np.mean(diff_norm**2)))

    preds_deg = preds * scale_deg
    gts_deg = gts * scale_deg
    diff_deg = preds_deg - gts_deg

    mae_deg = float(np.mean(np.abs(diff_deg)))
    rmse_deg = float(np.sqrt(np.mean(diff_deg**2)))
    max_err_deg = float(np.max(np.abs(diff_deg)))

    # R2 Score
    ss_res = np.sum(diff_norm**2)
    ss_tot = np.sum((gts - np.mean(gts)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "mae_norm": mae_norm,
        "rmse_norm": rmse_norm,
        "mae_deg": mae_deg,
        "rmse_deg": rmse_deg,
        "max_err_deg": max_err_deg,
        "r2": r2,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device)
        targets = batch["angle"].unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    """Evaluates the model on validation or test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds: list[float] = []
    all_targets: list[float] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device)
        targets = batch["angle"].unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

        all_preds.extend(outputs.cpu().numpy().flatten().tolist())
        all_targets.extend(targets.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    preds_arr = np.array(all_preds, dtype=np.float32)
    targets_arr = np.array(all_targets, dtype=np.float32)
    metrics = calculate_metrics(preds_arr, targets_arr)

    return avg_loss, metrics, preds_arr, targets_arr


def plot_learning_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Plots and saves loss and MAE learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Huber Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Huber Loss", color="#ff7f0e", lw=2)
    ax1.set_title("Huber Loss Evolution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # MAE (deg) curve
    ax2.plot(epochs, history["val_mae_deg"], label="Val MAE (°)", color="#2ca02c", lw=2)
    ax2.set_title("Validation MAE (°)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error (°)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_evaluation_artifacts(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Generates scatter and residual distribution plots for model evaluation."""
    preds_deg = predictions * 30.0
    gts_deg = targets * 30.0
    residuals = preds_deg - gts_deg

    # 1. Ground Truth vs Predicted Scatter Plot
    scatter_path = output_dir / "steer_scatter.png"
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gts_deg, preds_deg, alpha=0.4, color="#1f77b4", edgecolors="none", s=25)
    ax.plot([-30, 30], [-30, 30], color="#d62728", linestyle="--", lw=2, label="Identity (y=x)")
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_xlabel("Ground Truth Angle (°)", fontsize=11)
    ax.set_ylabel("Predicted Angle (°)", fontsize=11)
    ax.set_title(f"{model_name}: Predictions vs. Ground Truth", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # 2. Residual Distribution Plot
    residual_path = output_dir / "residual_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="#2ca02c", alpha=0.7, edgecolor="black", density=True)
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    ax.axvline(mean_res, color="#d62728", linestyle="--", lw=2, label=f"Mean: {mean_res:+.2f}°")
    ax.axvline(
        mean_res - std_res, color="#ff7f0e", linestyle=":", lw=1.5, label=f"±1σ: {std_res:.2f}°"
    )
    ax.axvline(mean_res + std_res, color="#ff7f0e", linestyle=":", lw=1.5)
    ax.set_xlabel("Residual Error (°)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"{model_name}: Residual Steering Angle Distribution", fontsize=13, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(residual_path, dpi=300)
    plt.close()

    return scatter_path, residual_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NXP Cup steering models with MLflow tracking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="micro_vit",
        choices=["dave2", "palhano_cnn", "micro_vit"],
        help="Model architecture name",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Execution device (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run 2 epochs with 3 batches for CI validation"
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=str(PROJECT_ROOT / "main" / "configs" / "train_config.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=str(PROJECT_ROOT / "main" / "configs" / "models_config.yaml"),
        help="Path to models config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "main" / "runs"),
        help="Directory to save checkpoints and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing NXP Cup training pipeline for model: %s", args.model)

    train_cfg = load_yaml(args.train_config)
    models_cfg = load_yaml(args.models_config)

    # Resolve device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info("Using compute device: %s", device)

    # Hyperparameters
    seed = int(train_cfg["dataset"].get("seed", 42))
    set_seed(seed)

    epochs = args.epochs or (2 if args.dry_run else int(train_cfg["training"]["epochs"]))
    batch_size = args.batch_size or int(train_cfg["training"]["batch_size"])
    learning_rate = args.lr or float(train_cfg["training"]["learning_rate"])
    weight_decay = float(train_cfg["training"].get("weight_decay", 1e-4))
    huber_delta = float(train_cfg["training"].get("huber_delta", 0.1))
    patience = int(train_cfg["training"].get("early_stopping_patience", 8))
    num_workers = 0 if args.dry_run else int(train_cfg["training"].get("num_workers", 2))

    # Resolve paths
    csv_path = Path(train_cfg["dataset"]["csv_path"])
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    images_dir = Path(train_cfg["dataset"]["data_dir"])
    if not images_dir.is_absolute():
        images_dir = PROJECT_ROOT / images_dir

    logger.info("Loading dataset from CSV: %s", csv_path)
    logger.info("Images directory: %s", images_dir)

    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=csv_path,
        images_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=float(train_cfg["dataset"].get("train_split", 0.70)),
        val_ratio=float(train_cfg["dataset"].get("val_split", 0.15)),
        test_ratio=float(train_cfg["dataset"].get("test_split", 0.15)),
        seed=seed,
    )

    # Model instantiation via Factory
    model_kwargs = models_cfg.get(args.model, {})
    model_kwargs = {k: v for k, v in model_kwargs.items() if k not in ["name", "input_shape"]}
    model = ModelFactory.create(args.model, **model_kwargs).to(device)

    summary_str = get_model_summary_str(model, args.model)
    logger.info(summary_str)
    counts = count_parameters(model)

    # Loss and optimizer
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg["training"].get("min_lr", 1e-5)),
    )

    # Output run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # MLflow Setup
    mlflow_tracking_uri = train_cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    experiment_name = train_cfg.get("mlflow", {}).get(
        "experiment_name", "nxpcup_autonomous_driving"
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    max_batches = 3 if args.dry_run else None

    with mlflow.start_run(run_name=f"{args.model}_{timestamp}"):
        mlflow.log_params(
            {
                "model_architecture": args.model,
                "total_parameters": counts["total"],
                "trainable_parameters": counts["trainable"],
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss_type": "HuberLoss",
                "huber_delta": huber_delta,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "device": device_str,
                "dry_run": args.dry_run,
            }
        )

        best_val_mae = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae_deg": []}

        logger.info("Starting training loop (%d epochs)...", epochs)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=max_batches,
            )
            val_loss, val_metrics, _, _ = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=max_batches,
            )
            scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae_deg"].append(val_metrics["mae_deg"])

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae_deg": val_metrics["mae_deg"],
                    "val_rmse_deg": val_metrics["rmse_deg"],
                    "val_r2": val_metrics["r2"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            logger.info(
                "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val MAE: %.2f° | R2: %.3f",
                epoch,
                epochs,
                elapsed,
                train_loss,
                val_loss,
                val_metrics["mae_deg"],
                val_metrics["r2"],
            )

            # Checkpoint best model
            if val_metrics["mae_deg"] < best_val_mae:
                best_val_mae = val_metrics["mae_deg"]
                best_epoch = epoch
                epochs_no_improve = 0

                checkpoint_path = run_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": args.model,
                        "state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                        "optimizer_state": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
                    },
                    checkpoint_path,
                )
                logger.info(
                    "  -> Saved new best model (MAE: %.2f°) to %s", best_val_mae, checkpoint_path
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and not args.dry_run:
                    logger.info(
                        "Early stopping triggered at epoch %d (patience: %d)", epoch, patience
                    )
                    break

        # Final Evaluation on Test Set using Best Checkpoint
        logger.info("Loading best model from epoch %d for final test evaluation...", best_epoch)
        best_ckpt_path = run_dir / "best_model.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            if isinstance(best_ckpt, dict) and "state_dict" in best_ckpt:
                model.load_state_dict(best_ckpt["state_dict"])

        test_loss, test_metrics, test_preds, test_gts = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )

        logger.info(
            "=== Final Test Results (%s) ===\n"
            "  Test Loss     : %.4f\n"
            "  Test MAE (deg): %.2f°\n"
            "  Test RMSE(deg): %.2f°\n"
            "  Test Max Error: %.2f°\n"
            "  Test R² Score : %.3f",
            args.model,
            test_loss,
            test_metrics["mae_deg"],
            test_metrics["rmse_deg"],
            test_metrics["max_err_deg"],
            test_metrics["r2"],
        )

        # Log final test metrics
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_mae_deg": test_metrics["mae_deg"],
                "test_rmse_deg": test_metrics["rmse_deg"],
                "test_max_err_deg": test_metrics["max_err_deg"],
                "test_r2": test_metrics["r2"],
                "best_epoch": best_epoch,
            }
        )

        # Generate & save visual evaluation artifacts
        curves_path = run_dir / "learning_curves.png"
        plot_learning_curves(history, curves_path)

        scatter_path, residual_path = plot_evaluation_artifacts(
            predictions=test_preds,
            targets=test_gts,
            output_dir=run_dir,
            model_name=args.model,
        )

        # Save summary JSON
        summary = {
            "model": args.model,
            "best_epoch": best_epoch,
            "best_val_mae_deg": best_val_mae,
            "test_metrics": test_metrics,
            "parameters": counts,
            "timestamp": timestamp,
        }
        summary_path = run_dir / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(run_dir / "best_model.pt"), artifact_path="model")
        mlflow.log_artifact(str(curves_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(scatter_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(residual_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(summary_path), artifact_path="metrics")

        logger.info(
            "Training and evaluation successfully completed. Artifacts stored in %s", run_dir
        )


if __name__ == "__main__":
    main()
