"""Treinamento de SVM com CSP para EEG em formato de trials."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from mne.decoding import CSP
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import build_trials_from_csv, seed_everything


def make_split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(math.ceil(n * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def eval_metrics(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict:
    y_pred = pipe.predict(X)
    return {
        "acc": float(accuracy_score(y, y_pred)),
        "precision_w": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall_w": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_w": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "cm": confusion_matrix(y, y_pred).astype(int).tolist(),
    }


def save_ckpt(obj: Dict, path: str) -> None:
    torch.save(obj, path, pickle_protocol=4)


def build_pipeline(n_components: int, c_value: float, kernel: str, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("csp", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("svm", SVC(C=float(c_value), kernel=kernel, random_state=seed)),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino de SVM com CSP para EEG em formato de trials.")

    parser.add_argument("--train-csv", type=str, required=True, help="Caminho para o CSV de treino.")
    parser.add_argument("--test-csv", type=str, required=True, help="Caminho para o CSV de teste.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./runs/svm_csp_2c",
        help="Diretorio para salvar modelos e checkpoints.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fracao do treino usada como validacao.")

    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="Lista de valores de C para busca em grade no Stage 1.",
    )
    parser.add_argument(
        "--kernel-grid",
        type=str,
        nargs="+",
        default=["linear"],
        help="Lista de kernels para busca em grade no Stage 1.",
    )
    parser.add_argument(
        "--csp-components-grid",
        type=int,
        nargs="+",
        default=[4, 6],
        help="Lista de numeros de componentes CSP para busca em grade no Stage 1.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_stage1 = os.path.join(args.save_dir, "svm_csp_stage1_best.pt")
    ckpt_final = os.path.join(args.save_dir, "svm_csp_final.pt")

    model_stage1 = os.path.join(args.save_dir, "svm_csp_stage1_best.joblib")
    model_final = os.path.join(args.save_dir, "svm_csp_final.joblib")

    run_info_path = os.path.join(args.save_dir, "run_info.json")

    print("Lendo treino...")
    X_tr, y_tr, info_tr = build_trials_from_csv(args.train_csv)
    print("Info treino:", info_tr.to_dict())

    train_channel_cols = info_tr.channel_cols
    class_to_idx = info_tr.class_to_idx

    print("Lendo teste...")
    X_te, y_te, info_te = build_trials_from_csv(
        args.test_csv,
        train_channel_cols=train_channel_cols,
        class_to_idx=class_to_idx,
    )
    print("Info teste:", info_te.to_dict())

    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError(f"Canais diferentes: train={X_tr.shape[1]} vs test={X_te.shape[1]}")

    if X_tr.shape[2] != X_te.shape[2]:
        T = min(X_tr.shape[2], X_te.shape[2])
        X_tr = X_tr[:, :, :T]
        X_te = X_te[:, :, :T]
        print(f"Ajustei n_time para {T} (corte pelo menor).")

    with open(run_info_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "train_info": info_tr.to_dict(),
                "test_info": info_te.to_dict(),
                "train_channel_cols": train_channel_cols,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    tr_idx, va_idx = make_split_indices(len(y_tr), args.val_split, args.seed)

    X_train = X_tr[tr_idx]
    y_train = y_tr[tr_idx]
    X_val = X_tr[va_idx]
    y_val = y_tr[va_idx]
    X_test = X_te
    y_test = y_te

    valid_csp_components = [value for value in args.csp_components_grid if 1 <= value <= X_train.shape[1]]
    if not valid_csp_components:
        raise ValueError(
            f"Nenhum valor em --csp-components-grid e valido para {X_train.shape[1]} canais de treino."
        )

    best_val_acc = -1.0
    best_params: Optional[Dict[str, object]] = None
    best_val_metrics: Optional[Dict] = None
    best_test_metrics_at_best: Optional[Dict] = None

    for n_components in valid_csp_components:
        for kernel in args.kernel_grid:
            for c_value in args.c_grid:
                pipe = build_pipeline(
                    n_components=int(n_components),
                    c_value=float(c_value),
                    kernel=kernel,
                    seed=int(args.seed),
                )
                pipe.fit(X_train, y_train)

                val_metrics = eval_metrics(pipe, X_val, y_val)
                test_metrics = eval_metrics(pipe, X_test, y_test)

                print(
                    f"[Stage1] CSP={int(n_components):>2} | kernel={kernel:<7} | "
                    f"C={float(c_value):>6.2f} | val_acc={val_metrics['acc']:.4f} | "
                    f"test_acc={test_metrics['acc']:.4f}"
                )

                if val_metrics["acc"] > best_val_acc:
                    best_val_acc = float(val_metrics["acc"])
                    best_params = {
                        "n_components": int(n_components),
                        "C": float(c_value),
                        "kernel": kernel,
                    }
                    best_val_metrics = val_metrics
                    best_test_metrics_at_best = test_metrics

                    joblib.dump(pipe, model_stage1)
                    save_ckpt(
                        {
                            "stage": "stage1",
                            "best_params": best_params,
                            "best_val_acc": best_val_acc,
                            "val_metrics": best_val_metrics,
                            "test_metrics_at_best": best_test_metrics_at_best,
                            "model_path_joblib": model_stage1,
                            "train_info": info_tr.to_dict(),
                            "test_info": info_te.to_dict(),
                            "train_channel_cols": train_channel_cols,
                            "seed": int(args.seed),
                            "val_split": float(args.val_split),
                            "c_grid": [float(value) for value in args.c_grid],
                            "kernel_grid": list(args.kernel_grid),
                            "csp_components_grid": [int(value) for value in valid_csp_components],
                        },
                        ckpt_stage1,
                    )

    if best_params is None:
        raise RuntimeError("Nao foi possivel selecionar hiperparametros para o SVM com CSP.")

    ckpt1 = torch.load(ckpt_stage1, map_location="cpu", weights_only=False)

    print("Stage1 best checkpoint")
    print(f"  best_params: {ckpt1['best_params']}")
    print(f"  best_val_acc: {ckpt1['best_val_acc']:.6f}")
    print(f"  test_acc_at_best: {ckpt1['test_metrics_at_best']['acc']:.6f}")
    print(f"  model_joblib: {ckpt1['model_path_joblib']}")

    X_trainval = X_tr
    y_trainval = y_tr

    pipe_final = build_pipeline(
        n_components=int(ckpt1["best_params"]["n_components"]),
        c_value=float(ckpt1["best_params"]["C"]),
        kernel=str(ckpt1["best_params"]["kernel"]),
        seed=int(args.seed),
    )
    pipe_final.fit(X_trainval, y_trainval)

    final_test_metrics = eval_metrics(pipe_final, X_test, y_test)
    joblib.dump(pipe_final, model_final)

    save_ckpt(
        {
            "stage": "final",
            "best_params": ckpt1["best_params"],
            "test_metrics": final_test_metrics,
            "model_path_joblib": model_final,
            "train_info": info_tr.to_dict(),
            "test_info": info_te.to_dict(),
            "train_channel_cols": train_channel_cols,
            "seed": int(args.seed),
        },
        ckpt_final,
    )

    print("Final checkpoint")
    print(f"  best_params: {ckpt1['best_params']}")
    print(f"  test_acc: {final_test_metrics['acc']:.6f}")
    print(f"  precision_w: {final_test_metrics['precision_w']:.6f}")
    print(f"  recall_w: {final_test_metrics['recall_w']:.6f}")
    print(f"  f1_w: {final_test_metrics['f1_w']:.6f}")
    print("  confusion_matrix:")
    for row in final_test_metrics["cm"]:
        print(f"   {row}")

    print("Saved files")
    print(f"  {ckpt_stage1}")
    print(f"  {model_stage1}")
    print(f"  {ckpt_final}")
    print(f"  {model_final}")
    print(f"  {run_info_path}")


if __name__ == "__main__":
    main()
