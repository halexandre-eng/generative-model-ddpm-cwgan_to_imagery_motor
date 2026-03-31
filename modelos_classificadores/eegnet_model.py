"""
Script de treinamento do EEGNet (classificação binária) com um esquema de treinamento em duas etapas.

Etapa 1:
  - Treina no conjunto TRAIN e valida no conjunto VAL (derivado do TRAIN).
  - Salva o melhor checkpoint com base na acurácia de validação.

Etapa 2:
  - Re-treina usando TRAIN+VAL e avalia no conjunto TEST a cada época.
  - Salva o melhor checkpoint com base na menor loss no teste.
  - Interrompe antecipadamente quando a loss no teste ficar menor do que a loss do teste na Etapa 1 (no melhor ponto de validação).

Formato esperado do CSV:
  - Deve conter uma coluna de rótulo (padrão: "label").
  - Deve conter colunas de agrupamento entre: "patient", "session", "epoch".
    Os trials são construídos agrupando as linhas por essas colunas.
  - Pode conter uma coluna opcional de tempo (padrão: "time") para ordenar as amostras dentro de cada trial.
  - As colunas de canais são inferidas como colunas numéricas que não fazem parte das colunas de metadados.

Saídas:
  - stage1_best.pt e final.pt salvos em --save-dir
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import TrialsInfo, build_trials_from_csv, seed_everything


class EEGTrialsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = int(math.ceil(n * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


class EEGNet(nn.Module):
    """
    Input: (B, 1, C, T)
    Output: log-probabilities (LogSoftmax) for NLLLoss
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_time: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_time)
            out = self._forward_features(dummy)
            n_flat = int(out.shape[1])

        self.classifier = nn.Sequential(
            nn.Linear(n_flat, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(x)
        return self.classifier(feats)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(X)
        loss = criterion(out, y)

        total_loss += float(loss.item()) * int(y.size(0))
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    denom = max(total, 1)
    return total_loss / denom, correct / denom


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(y.size(0))
        total += int(y.size(0))

    return total_loss / max(total, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEGNet on trial-based EEG CSVs (two-stage training).")

    p.add_argument("--train-csv", type=str, required=True, help="Path to training CSV.")
    p.add_argument("--test-csv", type=str, required=True, help="Path to test CSV.")
    p.add_argument("--save-dir", type=str, default="./runs/eegnet_2c", help="Directory to save checkpoints and logs.")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior (slower).")

    p.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")

    p.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction from training trials.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory for DataLoader.")
    p.add_argument("--persistent-workers", action="store_true", help="Enable persistent_workers for DataLoader.")

    p.add_argument("--max-epochs-stage1", type=int, default=1500, help="Max epochs for Stage 1.")
    p.add_argument("--patience", type=int, default=200, help="Early stopping patience for Stage 1.")
    p.add_argument("--max-epochs-stage2", type=int, default=600, help="Max epochs for Stage 2.")
    p.add_argument("--log-every", type=int, default=10, help="Print metrics every N epochs.")

    p.add_argument("--F1", type=int, default=8, help="EEGNet F1.")
    p.add_argument("--D", type=int, default=2, help="EEGNet D.")
    p.add_argument("--F2", type=int, default=16, help="EEGNet F2.")
    p.add_argument("--kernel-length", type=int, default=64, help="EEGNet temporal kernel length.")
    p.add_argument("--dropout", type=float, default=0.25, help="EEGNet dropout.")

    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_stage1 = os.path.join(args.save_dir, "eegnet_stage1_best.pt")
    ckpt_final = os.path.join(args.save_dir, "eegnet_final.pt")
    run_info_path = os.path.join(args.save_dir, "run_info.json")

    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=args.deterministic)

    print(f"Device: {device}")
    print("Loading training CSV...")
    X_tr, y_tr, info_tr = build_trials_from_csv(args.train_csv)
    print("Train info:", info_tr.to_dict())

    print("Loading test CSV...")
    X_te, y_te, info_te = build_trials_from_csv(
        args.test_csv,
        train_channel_cols=info_tr.channel_cols,
        class_to_idx=info_tr.class_to_idx,
    )
    print("Test info:", info_te.to_dict())

    # Ensure same channel/time dims (cut time if needed)
    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError(f"Different number of channels: train={X_tr.shape[1]} vs test={X_te.shape[1]}")

    if X_tr.shape[2] != X_te.shape[2]:
        T = min(X_tr.shape[2], X_te.shape[2])
        X_tr = X_tr[:, :, :T]
        X_te = X_te[:, :, :T]
        print(f"Adjusted n_time to {T} by cutting to the smallest length.")

    n_classes = int(len(np.unique(y_tr)))
    if n_classes != 2:
        print(f"Warning: n_classes in training is {n_classes}. This script is typically used for 2-class problems.")

    # Split train into train/val
    tr_idx, va_idx = make_split_indices(len(y_tr), args.val_split, args.seed)

    ds_train = EEGTrialsDataset(X_tr[tr_idx], y_tr[tr_idx])
    ds_val = EEGTrialsDataset(X_tr[va_idx], y_tr[va_idx])
    ds_test = EEGTrialsDataset(X_te, y_te)

    dl_common = dict(
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **dl_common)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **dl_common)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **dl_common)

    model = EEGNet(
        n_channels=int(X_tr.shape[1]),
        n_classes=n_classes,
        n_time=int(X_tr.shape[2]),
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=min(args.kernel_length, int(X_tr.shape[2])),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Persist run config
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "train_info": info_tr.to_dict(),
                "test_info": info_te.to_dict(),
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================
    # Stage 1
    # =========================
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs_stage1 + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, dl_val, criterion, device)
        te_loss, te_acc = evaluate(model, dl_test, criterion, device)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "test_loss_at_best": te_loss,
                    "test_acc_at_best": te_acc,
                    "train_info": info_tr.to_dict(),
                    "test_info": info_te.to_dict(),
                    "seed": args.seed,
                    "args": vars(args),
                },
                ckpt_stage1,
            )
        else:
            epochs_no_improve += 1

        if epoch == 1 or (epoch % args.log_every == 0):
            print(
                f"[Stage1][{epoch:4d}] "
                f"tr_loss={tr_loss:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}"
            )

        if epochs_no_improve >= args.patience:
            print(f"Early stopping Stage 1 at epoch={epoch} (best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f})")
            break

    ckpt1 = torch.load(ckpt_stage1, map_location=device)
    model.load_state_dict(ckpt1["model_state"])
    stage1_test_loss = float(ckpt1["test_loss_at_best"])
    stage1_test_acc = float(ckpt1["test_acc_at_best"])

    print("Stage 1 best checkpoint")
    print(f"  best_epoch: {ckpt1['epoch']}")
    print(f"  best_val_acc: {float(ckpt1['best_val_acc']):.6f}")
    print(f"  test_loss_at_best: {stage1_test_loss:.6f}")
    print(f"  test_acc_at_best: {stage1_test_acc:.6f}")

    # =========================
    # Stage 2
    # =========================
    ds_trainval = EEGTrialsDataset(X_tr, y_tr)
    dl_trainval = DataLoader(ds_trainval, batch_size=args.batch_size, shuffle=True, **dl_common)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_loss_stage2 = float("inf")
    best_test_acc_stage2 = 0.0
    best_epoch_stage2 = -1

    for epoch in range(1, args.max_epochs_stage2 + 1):
        tr_loss = train_one_epoch(model, dl_trainval, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, dl_test, criterion, device)

        if te_loss < best_test_loss_stage2:
            best_test_loss_stage2 = te_loss
            best_test_acc_stage2 = te_acc
            best_epoch_stage2 = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                    "stage1_test_loss_target": stage1_test_loss,
                    "train_info": info_tr.to_dict(),
                    "test_info": info_te.to_dict(),
                    "seed": args.seed,
                    "args": vars(args),
                },
                ckpt_final,
            )

        if epoch == 1 or (epoch % args.log_every == 0):
            print(
                f"[Stage2][{epoch:4d}] "
                f"tr_loss={tr_loss:.4f} | test_loss={te_loss:.4f} test_acc={te_acc:.4f} | "
                f"target_loss<{stage1_test_loss:.4f}"
            )

        if te_loss < stage1_test_loss:
            print(
                f"Stopping Stage 2 at epoch={epoch} because test_loss={te_loss:.4f} "
                f"is lower than stage1_test_loss={stage1_test_loss:.4f}"
            )
            break

    ckptf = torch.load(ckpt_final, map_location=device)
    print("Final checkpoint")
    print(f"  best_epoch_stage2: {ckptf['epoch']}")
    print(f"  best_test_loss_stage2: {float(ckptf['test_loss']):.6f}")
    print(f"  best_test_acc_stage2: {float(ckptf['test_acc']):.6f}")

    print("Saved files")
    print(f"  {ckpt_stage1}")
    print(f"  {ckpt_final}")
    print(f"  {run_info_path}")


if __name__ == "__main__":
    main()
