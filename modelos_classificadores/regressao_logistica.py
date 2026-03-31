"""
Classificador de Regressao Logistica (SGDClassifier com loss log_loss) para EEG em formato de trials.

Etapa 1:
  - Constroi os trials a partir do CSV de treino, divide em TRAIN/VAL e faz busca em grade (ALPHA_GRID).
  - Seleciona o melhor alpha com base na acuracia de validacao (val_acc).
  - Salva o melhor pipeline (StandardScaler + SGDClassifier) em .joblib e um checkpoint .pt com metadados.

Etapa 2:
  - Re-treina o pipeline com o melhor alpha usando TRAIN+VAL (todo o conjunto de treino).
  - Avalia no CSV de teste (TEST) e salva o modelo final em .joblib e um checkpoint .pt com metricas.

Formato esperado do CSV:
  - Deve conter uma coluna de rotulo (padrao: "label").
  - Deve conter colunas de agrupamento entre: "patient", "session", "epoch" (usadas para formar os trials).
  - Pode conter uma coluna opcional de tempo (padrao: "time") para ordenar as amostras dentro de cada trial.
  - As colunas de canais sao inferidas como colunas numericas que nao fazem parte dos metadados.

Saidas:
  - Modelos sklearn salvos em .joblib (stage1 e final).
  - Checkpoints com metricas e metadados em .pt (stage1 e final).

Observacao:
O classificador opera sobre um vetor por trial. Cada trial (C x T) e achatado em (C*T).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import TrialsInfo, build_trials_from_csv, seed_everything


def make_split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(math.ceil(n * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def flatten_trials(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)


def eval_clf(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict:
    y_pred = pipe.predict(X)
    cm = confusion_matrix(y, y_pred).astype(int)
    return {
        "acc": float(accuracy_score(y, y_pred)),
        "precision_w": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall_w": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_w": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "cm": cm.tolist(),
    }


def save_ckpt(obj: Dict, path: str) -> None:
    torch.save(obj, path, pickle_protocol=4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino de Regressao Logistica (SGDClassifier) para EEG em trials.")

    p.add_argument("--train-csv", type=str, required=True, help="Caminho para o CSV de treino.")
    p.add_argument("--test-csv", type=str, required=True, help="Caminho para o CSV de teste.")
    p.add_argument("--save-dir", type=str, default="./runs/logreg_2c", help="Diretorio para salvar modelos e checkpoints.")

    p.add_argument("--seed", type=int, default=42, help="Seed.")
    p.add_argument("--val-split", type=float, default=0.1, help="Fracao do treino usada como validacao.")

    p.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        help="Lista de alphas para busca em grade no Stage 1.",
    )

    p.add_argument(
        "--class-weight-balanced",
        action="store_true",
        help="Ativa class_weight='balanced' no SGDClassifier.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_stage1 = os.path.join(args.save_dir, "logreg_stage1_best.pt")
    ckpt_final = os.path.join(args.save_dir, "logreg_final.pt")

    model_stage1 = os.path.join(args.save_dir, "logreg_stage1_best.joblib")
    model_final = os.path.join(args.save_dir, "logreg_final.joblib")

    run_info_path = os.path.join(args.save_dir, "run_info.json")

    class_weight = "balanced" if args.class_weight_balanced else None

    print("Lendo treino...")
    X_tr, y_tr, info_tr = build_trials_from_csv(args.train_csv)
    print("Info treino:", info_tr.to_dict())

    train_channel_cols = info_tr.channel_cols

    print("Lendo teste...")
    X_te, y_te, info_te = build_trials_from_csv(args.test_csv, train_channel_cols=train_channel_cols)
    print("Info teste:", info_te.to_dict())

    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError(f"Canais diferentes: train={X_tr.shape[1]} vs test={X_te.shape[1]}")

    if X_tr.shape[2] != X_te.shape[2]:
        T = min(X_tr.shape[2], X_te.shape[2])
        X_tr = X_tr[:, :, :T]
        X_te = X_te[:, :, :T]
        print(f"Ajustei n_time para {T} (corte pelo menor).")

    n_classes = int(len(np.unique(y_tr)))
    print(f"n_classes: {n_classes}")

    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "train_info": info_tr.to_dict(),
                "test_info": info_te.to_dict(),
                "train_channel_cols": train_channel_cols,
                "class_weight": class_weight,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    tr_idx, va_idx = make_split_indices(len(y_tr), args.val_split, args.seed)

    X_train = flatten_trials(X_tr[tr_idx])
    y_train = y_tr[tr_idx]
    X_val = flatten_trials(X_tr[va_idx])
    y_val = y_tr[va_idx]

    X_test = flatten_trials(X_te)
    y_test = y_te

    print("Shapes")
    print(f"  X_train: {X_train.shape} {X_train.dtype}")
    print(f"  X_val  : {X_val.shape} {X_val.dtype}")
    print(f"  X_test : {X_test.shape} {X_test.dtype}")

    # =========================
    # Stage 1: search em alpha por val_acc
    # =========================
    best_val_acc = -1.0
    best_alpha: Optional[float] = None
    best_val_metrics: Optional[Dict] = None
    best_test_metrics_at_best: Optional[Dict] = None

    for alpha in args.alpha_grid:
        print(f"[Stage1] Treinando alpha={alpha} ...")

        pipe = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=float(alpha),
                        max_iter=2000,
                        tol=1e-3,
                        class_weight=class_weight,
                        random_state=int(args.seed),
                        n_jobs=-1,
                        verbose=0,
                    ),
                ),
            ]
        )

        pipe.fit(X_train, y_train)

        val_m = eval_clf(pipe, X_val, y_val)
        test_m = eval_clf(pipe, X_test, y_test)

        print(f"[Stage1] alpha={alpha:.1e} | val_acc={val_m['acc']:.4f} | test_acc={test_m['acc']:.4f}")

        if val_m["acc"] > best_val_acc:
            best_val_acc = float(val_m["acc"])
            best_alpha = float(alpha)
            best_val_metrics = val_m
            best_test_metrics_at_best = test_m

            joblib.dump(pipe, model_stage1)

            save_ckpt(
                {
                    "stage": "stage1",
                    "best_alpha": best_alpha,
                    "best_val_acc": best_val_acc,
                    "val_metrics": best_val_metrics,
                    "test_metrics_at_best": best_test_metrics_at_best,
                    "model_path_joblib": model_stage1,
                    "train_info": info_tr.to_dict(),
                    "test_info": info_te.to_dict(),
                    "train_channel_cols": train_channel_cols,
                    "seed": int(args.seed),
                    "val_split": float(args.val_split),
                    "alpha_grid": [float(x) for x in args.alpha_grid],
                    "class_weight": class_weight,
                },
                ckpt_stage1,
            )

    if best_alpha is None:
        raise RuntimeError("Nao foi possivel selecionar best_alpha. Verifique os dados e o grid de alpha.")

    ckpt1 = torch.load(ckpt_stage1, map_location="cpu", weights_only=False)

    print("Stage1 best checkpoint")
    print(f"  best_alpha: {ckpt1['best_alpha']}")
    print(f"  best_val_acc: {ckpt1['best_val_acc']:.6f}")
    print(f"  test_acc_at_best: {ckpt1['test_metrics_at_best']['acc']:.6f}")
    print(f"  model_joblib: {ckpt1['model_path_joblib']}")

    # =========================
    # Stage 2: train on train+val with best_alpha
    # =========================
    X_trainval = flatten_trials(X_tr)
    y_trainval = y_tr

    print("[Stage2] Treinando no train+val com best_alpha...")
    pipe_final = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=float(ckpt1["best_alpha"]),
                    max_iter=2000,
                    tol=1e-3,
                    class_weight=class_weight,
                    random_state=int(args.seed),
                    n_jobs=-1,
                    verbose=0,
                ),
            ),
        ]
    )
    pipe_final.fit(X_trainval, y_trainval)

    final_test_m = eval_clf(pipe_final, X_test, y_test)

    joblib.dump(pipe_final, model_final)

    save_ckpt(
        {
            "stage": "final",
            "best_alpha": float(ckpt1["best_alpha"]),
            "test_metrics": final_test_m,
            "model_path_joblib": model_final,
            "train_info": info_tr.to_dict(),
            "test_info": info_te.to_dict(),
            "train_channel_cols": train_channel_cols,
            "seed": int(args.seed),
            "class_weight": class_weight,
        },
        ckpt_final,
    )

    print("Final checkpoint")
    print(f"  best_alpha: {float(ckpt1['best_alpha'])}")
    print(f"  test_acc: {final_test_m['acc']:.6f}")
    print(f"  precision_w: {final_test_m['precision_w']:.6f}")
    print(f"  recall_w: {final_test_m['recall_w']:.6f}")
    print(f"  f1_w: {final_test_m['f1_w']:.6f}")
    print("  confusion_matrix:")
    for row in final_test_m["cm"]:
        print(f"   {row}")

    print("Saved files")
    print(f"  {ckpt_stage1}")
    print(f"  {model_stage1}")
    print(f"  {ckpt_final}")
    print(f"  {model_final}")
    print(f"  {run_info_path}")


if __name__ == "__main__":
    main()
