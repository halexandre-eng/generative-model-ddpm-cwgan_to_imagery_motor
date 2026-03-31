"""
DGAFF-like Channel Selection (Opção A) usando EXATAMENTE um classificador EEGNet já treinado.

Nota metodológica:
- O EEGNet foi treinado com todos os canais.
- Para manter o classificador exatamente igual, usamos "channel masking":
  canais não selecionados são zerados, mantendo o input (N, 1, C, T).
- Assim, o GA encontra o subconjunto de canais mais informativos para o modelo treinado.

Saídas:
- JSON com resultados e baseline
- TXT com lista de canais selecionados
- CSV com histórico por geração

Dependências:
  pip install deap scikit-learn numpy pandas torch

Uso:
  python dgaff_eegnet_channel_selection.py \
    --ckpt path/to/eegnet_final.pt \
    --csv  path/to/WBCIC_2C_test.csv \
    --out  outputs/ \
    --dataset-name baseline_0 \
    --k 15 --mu 12 --lambda 42 --gen 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from deap import base, creator, tools
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import build_label_mapping_from_values, load_trials_with_metadata, seed_everything


# =========================
# Reprodutibilidade
# =========================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Helpers de dataset
# =========================
META_COLS_CANDIDATES = {
    "dataset_type", "patient", "session", "epoch", "time",
    "label", "label_name"
}


def detect_group_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    if "patient" in df.columns:
        cols.append("patient")
    if "session" in df.columns:
        cols.append("session")
    if "epoch" in df.columns:
        cols.append("epoch")
    if not cols:
        raise ValueError("Não encontrei colunas de agrupamento (patient/session/epoch).")
    return cols


def detect_label_col(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    for c in df.columns:
        if c.lower() in ("y", "target", "class"):
            return c
    raise ValueError("Não encontrei coluna de label (ex.: 'label').")


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    if "time" in df.columns:
        return "time"
    for c in df.columns:
        if c.lower() in ("t", "sample", "idx"):
            return c
    return None


def build_label_mapping_from_values(y_values: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    uniq = sorted(list(set(int(v) for v in y_values)))
    if uniq == [1, 2]:
        class_to_idx = {1: 0, 2: 1}
    else:
        class_to_idx = {c: i for i, c in enumerate(uniq)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_trials_in_train_channel_order(
    csv_path: Path,
    train_channel_cols: Sequence[str],
    class_to_idx: Optional[Dict[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any], Dict[str, Any]]:
    df = pd.read_csv(csv_path)

    group_cols = detect_group_cols(df)
    label_col = detect_label_col(df)
    time_col = detect_time_col(df)

    sort_cols = list(group_cols)
    if time_col:
        sort_cols.append(time_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    missing = [c for c in train_channel_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} não contém canais esperados do treino. "
            f"Missing (primeiros 10): {missing[:10]}"
        )

    X_list: List[np.ndarray] = []
    y_list_raw: List[int] = []
    meta_list: List[Any] = []

    for key, g in df.groupby(group_cols, sort=False):
        y = g[label_col].iloc[0]
        if g[label_col].nunique() > 1:
            y = g[label_col].mode().iloc[0]

        x = g[list(train_channel_cols)].to_numpy(dtype=np.float32)  # (T, C)

        X_list.append(x)
        y_list_raw.append(int(y))
        meta_list.append(key)

    lengths = [x.shape[0] for x in X_list]
    min_len = int(np.min(lengths))
    max_len = int(np.max(lengths))
    if min_len != max_len:
        X_list = [x[:min_len, :] for x in X_list]

    X = np.stack([x.T for x in X_list], axis=0)  # (N, C, T)
    y_raw = np.array(y_list_raw, dtype=np.int64)

    if class_to_idx is None:
        class_to_idx2, idx_to_class = build_label_mapping_from_values(y_raw)
        class_to_idx = class_to_idx2
    else:
        idx_to_class = {i: c for c, i in class_to_idx.items()}

    y = np.array([class_to_idx[int(v)] for v in y_raw], dtype=np.int64)

    info = {
        "n_trials": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_time": int(X.shape[2]),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "group_cols": group_cols,
        "label_col": label_col,
        "time_col": time_col,
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
    }
    return X, y, y_raw, meta_list, info


class EEGTrialsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,C,T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =========================
# EEGNet
# =========================
class EEGNet(nn.Module):
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


# =========================
# Métricas
# =========================
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = cm.shape[0]
    precision = np.zeros(n, dtype=np.float64)
    recall = np.zeros(n, dtype=np.float64)
    f1 = np.zeros(n, dtype=np.float64)
    support = cm.sum(axis=1).astype(np.int64)

    for k in range(n):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1k = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precision[k] = prec
        recall[k] = rec
        f1[k] = f1k

    return precision, recall, f1, support


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    _, recall, _, _ = precision_recall_f1_from_cm(cm)
    return float(np.mean(recall))


@torch.no_grad()
def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        loss = criterion(out, y)

        total_loss += float(loss.item()) * int(y.size(0))
        total += int(y.size(0))

        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())

        probs = torch.exp(out)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_prob) if all_prob else np.array([], dtype=np.float32)

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    return float(avg_loss), float(acc), y_true, y_pred, y_prob


def mask_to_key(mask_bool: np.ndarray) -> Tuple[int, ...]:
    return tuple(np.flatnonzero(mask_bool).tolist())


def repair_to_k(individual: List[int], k: int, rng: np.random.Generator) -> List[int]:
    idx_ones = [i for i, b in enumerate(individual) if b == 1]
    idx_zeros = [i for i, b in enumerate(individual) if b == 0]

    if len(idx_ones) > k:
        drop = rng.choice(idx_ones, size=(len(idx_ones) - k), replace=False)
        for i in drop:
            individual[int(i)] = 0
    elif len(idx_ones) < k:
        add = rng.choice(idx_zeros, size=(k - len(idx_ones)), replace=False)
        for i in add:
            individual[int(i)] = 1
    return individual


def train_surrogate(X_bin: np.ndarray, y_fit: np.ndarray, seed: int) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    model.fit(X_bin, y_fit)
    return model


def apply_channel_mask(X: np.ndarray, subset_mask: np.ndarray) -> np.ndarray:
    X2 = X.copy()
    inv = ~subset_mask
    X2[:, inv, :] = 0.0
    return X2


def compute_fitness_from_eval(
    loss: float,
    acc: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    fitness_metric: str,
) -> float:
    if fitness_metric == "accuracy":
        return float(acc)
    if fitness_metric == "balanced_accuracy":
        cm = confusion_matrix_np(y_true, y_pred, n_classes=n_classes)
        return float(balanced_accuracy_from_cm(cm))
    raise ValueError("fitness_metric deve ser 'accuracy' ou 'balanced_accuracy'.")


@dataclass
class GAConfig:
    seed: int = 42
    mu: int = 12
    lambda_: int = 42
    cx_pb: float = 0.85
    mut_pb: float = 0.08
    n_gen: int = 10
    true_eval_budget: int = 12
    k_channels: int = 15
    surrogate_min_samples: int = 30
    surrogate_retrain_every: int = 3
    fitness_metric: str = "accuracy"


def dgaff_select_channels_eegnet(
    model: nn.Module,
    criterion: nn.Module,
    X_full: np.ndarray,
    y: np.ndarray,
    channel_names: Sequence[str],
    n_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cfg: GAConfig,
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    n_channels = len(channel_names)

    # DEAP creators (evita erro em execuções repetidas)
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual():
        ind = [0] * n_channels
        ones = rng.choice(np.arange(n_channels), size=cfg.k_channels, replace=False)
        for i in ones:
            ind[int(i)] = 1
        return creator.Individual(ind)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / max(n_channels, 1))

    true_cache: Dict[Tuple[int, ...], float] = {}
    surrogate_X: List[np.ndarray] = []
    surrogate_y: List[float] = []
    surrogate_model: Optional[MLPRegressor] = None

    def true_eval(individual: List[int]) -> float:
        mask = np.array(individual, dtype=bool)
        key = mask_to_key(mask)
        if key in true_cache:
            return true_cache[key]

        X_masked = apply_channel_mask(X_full, mask)
        ds = EEGTrialsDataset(X_masked, y)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        loss, acc, y_true, y_pred, _ = evaluate_with_predictions(model, dl, criterion, device=device)
        fit = compute_fitness_from_eval(
            loss, acc, y_true, y_pred, n_classes=n_classes, fitness_metric=cfg.fitness_metric
        )
        true_cache[key] = fit
        return fit

    def predict_eval(individual: List[int]) -> float:
        if surrogate_model is None:
            return 0.5
        xb = np.asarray(individual, dtype=np.float32).reshape(1, -1)
        return float(surrogate_model.predict(xb)[0])

    pop = toolbox.population(n=cfg.mu)

    # Seed inicial com avaliações verdadeiras
    for ind in pop:
        repair_to_k(ind, cfg.k_channels, rng)
        fit = true_eval(ind)
        ind.fitness.values = (fit,)
        surrogate_X.append(np.asarray(ind, dtype=np.float32))
        surrogate_y.append(fit)

    best_ind = tools.selBest(pop, 1)[0]

    history_rows: List[Dict[str, Any]] = []

    for gen in range(1, cfg.n_gen + 1):
        if (len(surrogate_y) >= cfg.surrogate_min_samples) and (gen % cfg.surrogate_retrain_every == 0):
            surrogate_model = train_surrogate(np.vstack(surrogate_X), np.asarray(surrogate_y), seed=cfg.seed)

        offspring: List[Any] = []
        while len(offspring) < cfg.lambda_:
            p1, p2 = toolbox.select(pop, 2)
            c1, c2 = creator.Individual(p1[:]), creator.Individual(p2[:])

            if rng.random() < cfg.cx_pb:
                toolbox.mate(c1, c2)

            if rng.random() < cfg.mut_pb:
                toolbox.mutate(c1)
            if rng.random() < cfg.mut_pb:
                toolbox.mutate(c2)

            repair_to_k(c1, cfg.k_channels, rng)
            repair_to_k(c2, cfg.k_channels, rng)

            offspring.extend([c1, c2])

        offspring = offspring[: cfg.lambda_]

        scored: List[Tuple[Any, float, bool]] = []
        for ind in offspring:
            mask = np.array(ind, dtype=bool)
            key = mask_to_key(mask)
            if key in true_cache:
                scored.append((ind, true_cache[key], True))
            else:
                scored.append((ind, predict_eval(ind), False))

        scored.sort(key=lambda x: x[1], reverse=True)
        to_true = [t for t in scored if not t[2]][: cfg.true_eval_budget]

        for ind, _, _ in to_true:
            fit = true_eval(ind)
            ind.fitness.values = (fit,)
            surrogate_X.append(np.asarray(ind, dtype=np.float32))
            surrogate_y.append(fit)

        for ind, pred, _is_true in scored:
            if not ind.fitness.valid:
                ind.fitness.values = (pred,)

        combined = pop + [s[0] for s in scored]
        pop = tools.selBest(combined, cfg.mu)

        current_best = tools.selBest(pop, 1)[0]
        if float(current_best.fitness.values[0]) > float(best_ind.fitness.values[0]):
            best_ind = creator.Individual(current_best[:])
            best_ind.fitness.values = current_best.fitness.values

        best_true = true_eval(best_ind)

        row = {
            "gen": gen,
            "best_fitness_current": float(best_ind.fitness.values[0]),
            "best_true_fitness": float(best_true),
            "k": int(sum(best_ind)),
            "cache_size": int(len(true_cache)),
            "surrogate_on": bool(surrogate_model is not None),
        }
        history_rows.append(row)

        print(
            f"[Gen {gen:02d}/{cfg.n_gen}] "
            f"best_true={best_true:.4f} | "
            f"k={row['k']} | cache={row['cache_size']} | "
            f"surrogate={'ON' if row['surrogate_on'] else 'OFF'}"
        )

    best_mask = np.array(best_ind, dtype=bool)
    best_channels = [ch for ch, keep in zip(channel_names, best_mask) if keep]
    best_true = true_eval(best_ind)

    return {
        "best_true_fitness": float(best_true),
        "best_channels": best_channels,
        "best_mask": best_mask.astype(int).tolist(),
        "history": history_rows,
        "cache_size": int(len(true_cache)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DGAFF-like channel selection for a pretrained EEGNet (masking approach).")

    p.add_argument("--ckpt", type=str, required=True, help="Path para checkpoint .pt do EEGNet (com info_train).")
    p.add_argument("--csv", type=str, required=True, help="Path para CSV de teste/avaliação (precisa ter patient/session/epoch + canais).")
    p.add_argument("--out", type=str, required=True, help="Diretório de saída (JSON/TXT/CSV).")
    p.add_argument("--dataset-name", type=str, default="baseline", help="Nome do dataset (apenas para nomear outputs).")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device. 'auto' escolhe cuda se disponível.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)

    # GA config
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mu", type=int, default=12)
    p.add_argument("--lambda", dest="lambda_", type=int, default=42)
    p.add_argument("--cx", dest="cx_pb", type=float, default=0.85)
    p.add_argument("--mut", dest="mut_pb", type=float, default=0.08)
    p.add_argument("--gen", dest="n_gen", type=int, default=10)
    p.add_argument("--true-eval-budget", type=int, default=12)
    p.add_argument("--k", dest="k_channels", type=int, default=15)

    p.add_argument("--fitness-metric", type=str, default="accuracy", choices=["accuracy", "balanced_accuracy"])
    p.add_argument("--surrogate-min-samples", type=int, default=30)
    p.add_argument("--surrogate-retrain-every", type=int, default=3)

    # label mapping default p/ WBCIC 2C
    p.add_argument("--class-map", type=str, default="1:0,2:1",
                   help="Mapeamento de classes do CSV para índices do modelo. Ex: '1:0,2:1'")

    return p.parse_args()


def parse_class_map(s: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    s = s.strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        a, b = part.split(":")
        out[int(a)] = int(b)
    return out


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print("Device:", device)

    seed_everything(int(args.seed))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    info_train = ckpt.get("info_train", None)
    if info_train is None:
        raise ValueError("Checkpoint não tem 'info_train'. Use checkpoints gerados pelo treino que incluem info_train.")

    train_channel_cols = info_train["channel_cols"]
    n_channels_train = int(info_train["n_channels"])
    n_time_train = int(info_train["n_time"])

    # Modelo
    class_to_idx = parse_class_map(args.class_map) or None
    # Se o map for fornecido, inferimos n_classes pelo maior idx + 1
    if class_to_idx is not None:
        n_classes = max(class_to_idx.values()) + 1
    else:
        n_classes = 2  # fallback

    model = EEGNet(
        n_channels=n_channels_train,
        n_classes=n_classes,
        n_time=n_time_train,
        F1=8, D=2, F2=16,
        kernel_length=min(64, n_time_train),
        dropout=0.25
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.NLLLoss()

    # Carrega dados alinhados ao treino
    X, y, y_raw, meta_list, info_te = load_trials_with_metadata(
        csv_path,
        train_channel_cols=train_channel_cols,
        class_to_idx=class_to_idx,
    )

    if X.shape[1] != n_channels_train:
        raise ValueError(f"Canais diferentes: test={X.shape[1]} vs train={n_channels_train}")

    if X.shape[2] != n_time_train:
        T = min(X.shape[2], n_time_train)
        X = X[:, :, :T]
        print(f"[WARN] Ajustei n_time para {T} (corte pelo menor).")

    print("\n==============================")
    print("BASELINE CHECK (full channels)")
    print("==============================")
    ds_full = EEGTrialsDataset(X, y)
    dl_full = DataLoader(
        ds_full,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    loss_full, acc_full, y_true, y_pred, _ = evaluate_with_predictions(model, dl_full, criterion, device=device)
    cm_full = confusion_matrix_np(y_true, y_pred, n_classes=n_classes)
    bal_full = balanced_accuracy_from_cm(cm_full)

    print(f"Full loss={loss_full:.6f} acc={acc_full:.6f} bal_acc={bal_full:.6f}")
    print(f"Fitness metric='{args.fitness_metric}'")

    cfg = GAConfig(
        seed=int(args.seed),
        mu=int(args.mu),
        lambda_=int(args.lambda_),
        cx_pb=float(args.cx_pb),
        mut_pb=float(args.mut_pb),
        n_gen=int(args.n_gen),
        true_eval_budget=int(args.true_eval_budget),
        k_channels=int(args.k_channels),
        surrogate_min_samples=int(args.surrogate_min_samples),
        surrogate_retrain_every=int(args.surrogate_retrain_every),
        fitness_metric=str(args.fitness_metric),
    )

    print("\n==============================")
    print("DGAFF-LIKE CHANNEL SELECTION")
    print("==============================")
    print(f"Dataset: {args.dataset_name}")
    print(f"CSV    : {csv_path}")
    print(f"K      : {cfg.k_channels}")
    print(f"GA     : MU={cfg.mu}, LAMBDA={cfg.lambda_}, CX={cfg.cx_pb}, MUT={cfg.mut_pb}, GEN={cfg.n_gen}")
    print(f"Budget : TRUE_EVAL_BUDGET={cfg.true_eval_budget}")

    res = dgaff_select_channels_eegnet(
        model=model,
        criterion=criterion,
        X_full=X,
        y=y,
        channel_names=train_channel_cols,
        n_classes=n_classes,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        cfg=cfg,
    )

    best_channels = res["best_channels"]
    best_true = float(res["best_true_fitness"])

    print("\n======================")
    print("RESULTADO FINAL (DGAFF)")
    print("======================")
    print(f"Best true fitness ({cfg.fitness_metric}): {best_true:.6f}")
    print(f"Nº canais: {len(best_channels)}")
    print("Canais selecionados:", best_channels)

    # Avalia subset vs full
    best_mask = np.array(res["best_mask"], dtype=bool)
    X_best = apply_channel_mask(X, best_mask)
    ds_best = EEGTrialsDataset(X_best, y)
    dl_best = DataLoader(
        ds_best,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )
    loss_best, acc_best, y_true_b, y_pred_b, _ = evaluate_with_predictions(model, dl_best, criterion, device=device)
    cm_best = confusion_matrix_np(y_true_b, y_pred_b, n_classes=n_classes)
    bal_best = balanced_accuracy_from_cm(cm_best)

    print("\n==============================")
    print("COMPARAÇÃO (full vs subset-mask)")
    print("==============================")
    print(f"FULL  : loss={loss_full:.6f} acc={acc_full:.6f} bal_acc={bal_full:.6f}")
    print(f"SUBSET: loss={loss_best:.6f} acc={acc_best:.6f} bal_acc={bal_best:.6f}")

    # Salva outputs
    tag = f"{args.dataset_name}_K{cfg.k_channels}"
    out_json = out_dir / f"dgaff_best_channels_{tag}.json"
    out_txt = out_dir / f"dgaff_best_channels_{tag}.txt"
    out_hist = out_dir / f"dgaff_history_{tag}.csv"

    payload = {
        "dataset": str(args.dataset_name),
        "csv_path": str(csv_path),
        "ckpt_path": str(ckpt_path),
        "fitness_metric": cfg.fitness_metric,
        "K_channels": cfg.k_channels,
        "ga_params": {
            "MU": cfg.mu,
            "LAMBDA": cfg.lambda_,
            "CX_PB": cfg.cx_pb,
            "MUT_PB": cfg.mut_pb,
            "N_GEN": cfg.n_gen,
            "TRUE_EVAL_BUDGET": cfg.true_eval_budget,
            "SURROGATE_MIN_SAMPLES": cfg.surrogate_min_samples,
            "SURROGATE_RETRAIN_EVERY": cfg.surrogate_retrain_every,
            "SEED": cfg.seed,
        },
        "baseline_full": {
            "loss": float(loss_full),
            "accuracy": float(acc_full),
            "balanced_accuracy": float(bal_full),
        },
        "best_subset_masking": {
            "best_true_fitness": float(best_true),
            "loss": float(loss_best),
            "accuracy": float(acc_best),
            "balanced_accuracy": float(bal_best),
            "channels": list(best_channels),
            "mask": list(res["best_mask"]),
        },
        "cache_size": int(res["cache_size"]),
        "data_info": info_te,
        "train_info_from_ckpt": {
            "n_channels": n_channels_train,
            "n_time": n_time_train,
            "n_classes": n_classes,
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(best_channels) + "\n", encoding="utf-8")
    pd.DataFrame(res["history"]).to_csv(out_hist, index=False)

    print("\nSaved:")
    print(" ", out_json)
    print(" ", out_txt)
    print(" ", out_hist)
    print("Outputs dir:")
    print(" ", out_dir)


if __name__ == "__main__":
    main()
