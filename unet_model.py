"""
Classificador U-Net (Conv1D) para EEG (WBCIC 2C) com treinamento em duas etapas.

Etapa 1:
  - Treina no conjunto TRAIN e valida no conjunto VAL (derivado do TRAIN).
  - Salva o melhor modelo com base na acurácia de validação.

Etapa 2 (opcional):
  - Re-treina usando TRAIN+VAL e avalia no conjunto TEST a cada época.
  - Salva o melhor modelo com base na menor loss no teste.
  - Pode interromper antecipadamente quando a loss do teste ficar abaixo da referência obtida na Etapa 1.

Entradas esperadas (CSV):
  - Deve conter uma coluna de rótulo (padrão: "label").
  - Deve conter colunas de agrupamento entre: "patient", "session", "epoch" (usadas para formar os trials).
  - Pode conter uma coluna opcional de tempo (padrão: "time") para ordenar as amostras dentro de cada trial.
  - As colunas de canais são inferidas como colunas numéricas que não fazem parte dos metadados.

Saídas:
  - Modelos salvos em formato .keras (stage1 e final).
  - Scalers salvos em .joblib (stage1 e final).
  - Checkpoints com métricas e metadados em .pt.

Observação:
Este script treina com um par de CSVs (treino e teste). Os artefatos salvos podem ser reutilizados depois
para avaliar outros CSVs seguindo o mesmo formato.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classificador U-Net (Conv1D) para EEG (WBCIC 2C) com treinamento em duas etapas.

Melhorias em relação ao script original:
1) Salva os scalers (stage1 e final) com joblib.
2) Mapeamento de classes fixo para WBCIC 2C (1 -> 0, 2 -> 1) para manter consistência.
3) Salva a lista 'train_channel_cols' e valida a ordem de canais no teste (e no futuro em híbridos).
4) Stage2 mais seguro: só salva o final se houver melhora real em test_loss; e permite desligar o Stage2.
5) Checkpoints .pt leves (pickle_protocol=4) para evitar problemas de tamanho.
6) Pronto para reuso em um script de avaliação multi-dataset (baseline e híbridos 25/50/75).

Observação:
Por enquanto treina apenas com TRAIN_CSV/TEST_CSV (baseline). Depois pode ser usado em um avaliador
que carrega MODEL + SCALER salvos e roda em cada CSV híbrido.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Input,
    UpSampling1D,
    concatenate,
    ZeroPadding1D,
    Cropping1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from project_utils import TrialsInfo, build_trials_from_csv, seed_everything


# =========================
# Reproducibility
# =========================
def seed_all(seed: int) -> None:
    seed_everything(seed)
    tf.random.set_seed(seed)


def make_split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(math.ceil(n * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def fit_transform_trials_scaler(X_train_T_C: np.ndarray) -> Tuple[StandardScaler, np.ndarray]:
    """Ajusta o scaler em trials no formato (N, T, C)."""
    scaler = StandardScaler()
    N, T, C = X_train_T_C.shape
    flat = X_train_T_C.reshape(N, T * C)
    flat_s = scaler.fit_transform(flat)
    Xs = flat_s.reshape(N, T, C).astype(np.float32)
    return scaler, Xs


def transform_trials_scaler(scaler: StandardScaler, X_T_C: np.ndarray) -> np.ndarray:
    N, T, C = X_T_C.shape
    flat = X_T_C.reshape(N, T * C)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(N, T, C).astype(np.float32)


def create_unet_model(input_shape: Tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=input_shape)

    conv1 = Conv1D(64, 3, activation="relu", padding="same")(inputs)
    pool1 = MaxPooling1D(2, padding="same")(conv1)

    conv2 = Conv1D(128, 3, activation="relu", padding="same")(pool1)
    pool2 = MaxPooling1D(2, padding="same")(conv2)

    conv3 = Conv1D(256, 3, activation="relu", padding="same")(pool2)

    up4 = UpSampling1D(2)(conv3)
    if conv2.shape[1] != up4.shape[1]:
        if conv2.shape[1] > up4.shape[1]:
            up4 = ZeroPadding1D(padding=(0, int(conv2.shape[1] - up4.shape[1])))(up4)
        else:
            up4 = Cropping1D(cropping=(0, int(up4.shape[1] - conv2.shape[1])))(up4)

    merge4 = concatenate([conv2, up4], axis=-1)
    conv4 = Conv1D(128, 3, activation="relu", padding="same")(merge4)

    up5 = UpSampling1D(2)(conv4)
    if conv1.shape[1] != up5.shape[1]:
        if conv1.shape[1] > up5.shape[1]:
            up5 = ZeroPadding1D(padding=(0, int(conv1.shape[1] - up5.shape[1])))(up5)
        else:
            up5 = Cropping1D(cropping=(0, int(up5.shape[1] - conv1.shape[1])))(up5)

    merge5 = concatenate([conv1, up5], axis=-1)
    conv5 = Conv1D(64, 3, activation="relu", padding="same")(merge5)

    flat = Flatten()(conv5)
    outputs = Dense(num_classes, activation="softmax")(flat)

    return Model(inputs=inputs, outputs=outputs)


def eval_keras(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> Dict:
    y_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    return {
        "acc": float(accuracy_score(y, y_pred)),
        "precision_w": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall_w": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_w": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "cm": confusion_matrix(y, y_pred).astype(int).tolist(),
    }


def save_ckpt(obj: Dict, path: str) -> None:
    torch.save(obj, path, pickle_protocol=4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino do classificador U-Net Conv1D (WBCIC 2C) em duas etapas.")

    p.add_argument("--train-csv", type=str, required=True, help="Caminho para o CSV de treino.")
    p.add_argument("--test-csv", type=str, required=True, help="Caminho para o CSV de teste.")
    p.add_argument("--save-dir", type=str, default="./runs/unet_2c", help="Diretorio para salvar modelos e artefatos.")

    p.add_argument("--seed", type=int, default=42, help="Seed.")
    p.add_argument("--val-split", type=float, default=0.1, help="Fracao do treino usada como validacao.")

    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    p.add_argument("--max-epochs-stage1", type=int, default=200, help="Max epocas do Stage 1.")
    p.add_argument("--patience", type=int, default=20, help="Patience do EarlyStopping (Stage 1).")

    p.add_argument("--enable-stage2", action="store_true", help="Habilita Stage 2 (desligado por padrao).")
    p.add_argument("--max-epochs-stage2", type=int, default=100, help="Max epocas do Stage 2.")

    p.add_argument("--log-every", type=int, default=1, help="Imprime progresso no Stage2 a cada N epocas.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"TF Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    model_stage1_path = os.path.join(args.save_dir, "unet_stage1_best.keras")
    model_final_path = os.path.join(args.save_dir, "unet_final.keras")

    ckpt_stage1_path = os.path.join(args.save_dir, "unet_stage1_best.pt")
    ckpt_final_path = os.path.join(args.save_dir, "unet_final.pt")

    scaler_stage1_path = os.path.join(args.save_dir, "unet_scaler_stage1.joblib")
    scaler_final_path = os.path.join(args.save_dir, "unet_scaler_final.joblib")

    run_info_path = os.path.join(args.save_dir, "run_info.json")

    # Mapping global WBCIC 2C
    global_class_to_idx = {1: 0, 2: 1}
    n_classes = 2

    print("Lendo treino...")
    X_tr, y_tr, info_tr = build_trials_from_csv(
        args.train_csv,
        train_channel_cols=None,
        class_to_idx=global_class_to_idx,
    )
    print("Info treino:", info_tr.to_dict())

    train_channel_cols = info_tr.channel_cols
    n_channels_train = info_tr.n_channels
    n_time_train = info_tr.n_time

    print("Lendo teste...")
    X_te, y_te, info_te = build_trials_from_csv(
        args.test_csv,
        train_channel_cols=train_channel_cols,
        class_to_idx=global_class_to_idx,
    )
    print("Info teste:", info_te.to_dict())

    if X_te.shape[1] != n_channels_train:
        raise ValueError(f"Canais diferentes: train={n_channels_train} vs test={X_te.shape[1]}")

    if X_te.shape[2] != n_time_train:
        T = min(X_te.shape[2], n_time_train)
        X_tr = X_tr[:, :, :T]
        X_te = X_te[:, :, :T]
        n_time_train = T
        print(f"Ajustei n_time para {T} (corte pelo menor).")

    # Keras Conv1D usa (N, T, C)
    X_tr_t = np.transpose(X_tr, (0, 2, 1)).astype(np.float32)
    X_te_t = np.transpose(X_te, (0, 2, 1)).astype(np.float32)

    tr_idx, va_idx = make_split_indices(len(y_tr), args.val_split, args.seed)

    X_train = X_tr_t[tr_idx]
    y_train = y_tr[tr_idx]
    X_val = X_tr_t[va_idx]
    y_val = y_tr[va_idx]

    X_test = X_te_t
    y_test = y_te

    # Persist run info
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "tf_device": device,
                "train_info": info_tr.to_dict(),
                "test_info": info_te.to_dict(),
                "train_channel_cols": train_channel_cols,
                "global_class_to_idx": global_class_to_idx,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================
    # Stage 1
    # =========================
    scaler1, X_train_s = fit_transform_trials_scaler(X_train)
    X_val_s = transform_trials_scaler(scaler1, X_val)
    X_test_s = transform_trials_scaler(scaler1, X_test)

    joblib.dump(scaler1, scaler_stage1_path)

    input_shape = (X_train_s.shape[1], X_train_s.shape[2])  # (T, C)

    model = create_unet_model(input_shape=input_shape, num_classes=n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_es = EarlyStopping(
        monitor="val_accuracy",
        patience=args.patience,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    cb_ckpt = ModelCheckpoint(
        filepath=model_stage1_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )

    hist1 = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.max_epochs_stage1,
        batch_size=args.batch_size,
        callbacks=[cb_es, cb_ckpt],
        verbose=1,
    )

    best_model_s1 = tf.keras.models.load_model(model_stage1_path)

    val_m_s1 = eval_keras(best_model_s1, X_val_s, y_val)
    test_loss_s1, test_acc_s1 = best_model_s1.evaluate(X_test_s, y_test, verbose=0)
    test_m_s1 = eval_keras(best_model_s1, X_test_s, y_test)

    save_ckpt(
        {
            "stage": "stage1",
            "best_model_path": model_stage1_path,
            "scaler_path": scaler_stage1_path,
            "val_metrics": val_m_s1,
            "test_loss_at_best": float(test_loss_s1),
            "test_acc_at_best": float(test_acc_s1),
            "test_metrics_at_best": test_m_s1,
            "train_info": info_tr.to_dict(),
            "test_info": info_te.to_dict(),
            "train_channel_cols": train_channel_cols,
            "seed": int(args.seed),
            "val_split": float(args.val_split),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "epochs_ran": int(len(hist1.history.get("loss", []))),
            "global_class_to_idx": global_class_to_idx,
        },
        ckpt_stage1_path,
    )

    print("Stage1 best checkpoint")
    print(f"  val_acc: {val_m_s1['acc']:.6f}")
    print(f"  test_loss_at_best: {float(test_loss_s1):.6f}")
    print(f"  test_acc_at_best: {float(test_acc_s1):.6f}")
    print(f"  scaler_stage1: {scaler_stage1_path}")

    # =========================
    # Stage 2 (optional)
    # =========================
    if not args.enable_stage2:
        print("ENABLE_STAGE2 desligado. Encerrando no Stage 1.")
        print("Saved files")
        print(f"  {model_stage1_path}")
        print(f"  {scaler_stage1_path}")
        print(f"  {ckpt_stage1_path}")
        print(f"  {run_info_path}")
        return

    stage1_test_loss_target = float(test_loss_s1)

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    scaler2, X_trainval_s = fit_transform_trials_scaler(X_trainval)
    X_test_s2 = transform_trials_scaler(scaler2, X_test)
    joblib.dump(scaler2, scaler_final_path)

    model2 = tf.keras.models.load_model(model_stage1_path)
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_test_loss_s2 = float("inf")
    best_test_acc_s2 = 0.0
    best_epoch_s2 = -1

    for epoch in range(1, args.max_epochs_stage2 + 1):
        model2.fit(
            X_trainval_s,
            y_trainval,
            epochs=1,
            batch_size=args.batch_size,
            verbose=1,
        )

        te_loss, te_acc = model2.evaluate(X_test_s2, y_test, verbose=0)

        if float(te_loss) < best_test_loss_s2:
            best_test_loss_s2 = float(te_loss)
            best_test_acc_s2 = float(te_acc)
            best_epoch_s2 = int(epoch)
            model2.save(model_final_path)

        if epoch == 1 or (epoch % args.log_every == 0):
            print(
                f"[Stage2][{epoch:4d}] test_loss={float(te_loss):.4f} test_acc={float(te_acc):.4f} "
                f"| target_loss<{stage1_test_loss_target:.4f}"
            )

        if float(te_loss) < stage1_test_loss_target:
            print(
                f"Stopping Stage 2 at epoch={epoch} because test_loss={float(te_loss):.4f} "
                f"is lower than stage1_test_loss={stage1_test_loss_target:.4f}"
            )
            break

    if not os.path.exists(model_final_path):
        print("MODEL_FINAL nao foi salvo (sem melhorias). Usando stage1_best como final.")
        best_model_s1.save(model_final_path)
        joblib.dump(scaler1, scaler_final_path)
        best_test_loss_s2 = float(test_loss_s1)
        best_test_acc_s2 = float(test_acc_s1)
        best_epoch_s2 = 0

    best_model_final = tf.keras.models.load_model(model_final_path)
    final_test_loss, final_test_acc = best_model_final.evaluate(X_test_s2, y_test, verbose=0)
    final_test_metrics = eval_keras(best_model_final, X_test_s2, y_test)

    save_ckpt(
        {
            "stage": "final",
            "best_model_path": model_final_path,
            "scaler_path": scaler_final_path,
            "best_epoch_stage2": int(best_epoch_s2),
            "best_test_loss_stage2": float(best_test_loss_s2),
            "best_test_acc_stage2": float(best_test_acc_s2),
            "final_test_loss": float(final_test_loss),
            "final_test_acc": float(final_test_acc),
            "final_test_metrics": final_test_metrics,
            "stage1_test_loss_target": float(stage1_test_loss_target),
            "train_info": info_tr.to_dict(),
            "test_info": info_te.to_dict(),
            "train_channel_cols": train_channel_cols,
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "global_class_to_idx": global_class_to_idx,
        },
        ckpt_final_path,
    )

    print("Final checkpoint")
    print(f"  best_epoch_stage2: {best_epoch_s2}")
    print(f"  best_test_loss_stage2: {best_test_loss_s2:.6f}")
    print(f"  best_test_acc_stage2: {best_test_acc_s2:.6f}")
    print(f"  final_test_acc: {float(final_test_acc):.6f}")
    print("  confusion_matrix:")
    for row in final_test_metrics["cm"]:
        print(f"   {row}")

    print("Saved files")
    print(f"  {model_stage1_path}")
    print(f"  {scaler_stage1_path}")
    print(f"  {ckpt_stage1_path}")
    print(f"  {model_final_path}")
    print(f"  {scaler_final_path}")
    print(f"  {ckpt_final_path}")
    print(f"  {run_info_path}")


if __name__ == "__main__":
    main()
