from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


META_COLS_CANDIDATES = {
    "dataset_type",
    "patient",
    "session",
    "epoch",
    "time",
    "label",
    "label_name",
}


def _try_load_yaml(path: str) -> Optional[dict]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        return None


def load_diretrizes(path: str) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("Voce precisa informar --diretrizes com um arquivo YAML ou JSON.")

    yaml_data = _try_load_yaml(path)
    if yaml_data is not None:
        diretrizes = yaml_data.get("diretrizes") if isinstance(yaml_data, dict) else None
        if not diretrizes:
            raise ValueError("Arquivo YAML invalido. Era esperada a chave 'diretrizes'.")
        return diretrizes

    with open(path, "r", encoding="utf-8") as handle:
        json_data = json.load(handle)

    if isinstance(json_data, dict) and "diretrizes" in json_data:
        return json_data["diretrizes"]
    if isinstance(json_data, list):
        return json_data

    raise ValueError("Arquivo JSON invalido. Use {'diretrizes': [...]} ou uma lista direta.")


def seed_everything(seed: int, deterministic: bool = True, allow_tf32: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def detect_group_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    if "patient" in df.columns:
        cols.append("patient")
    if "session" in df.columns:
        cols.append("session")
    if "epoch" in df.columns:
        cols.append("epoch")

    if not cols:
        raise ValueError("Nao encontrei colunas de agrupamento. Use pelo menos patient, session ou epoch.")
    return cols


def detect_label_col(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"

    for column in df.columns:
        if column.lower() in ("y", "target", "class"):
            return column

    raise ValueError("Nao encontrei coluna de label. Use 'label', 'y', 'target' ou 'class'.")


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    if "time" in df.columns:
        return "time"

    for column in df.columns:
        if column.lower() in ("t", "sample", "idx"):
            return column

    return None


def detect_channel_cols(
    df: pd.DataFrame,
    label_col: str,
    time_col: Optional[str],
) -> List[str]:
    meta_cols = set(META_COLS_CANDIDATES)
    meta_cols.add(label_col)
    if time_col:
        meta_cols.add(time_col)

    channel_cols = [
        column
        for column in df.columns
        if column not in meta_cols and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not channel_cols:
        raise ValueError("Nao encontrei colunas numericas de canais apos remover os metadados.")
    return channel_cols


@dataclass(frozen=True)
class TrialsInfo:
    n_trials: int
    n_channels: int
    n_time: int
    min_len: int
    max_len: int
    group_cols: List[str]
    label_col: str
    time_col: Optional[str]
    channel_cols: List[str]
    classes_original: Optional[List[int]] = None
    class_to_idx: Optional[Dict[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "n_trials": self.n_trials,
            "n_channels": self.n_channels,
            "n_time": self.n_time,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "group_cols": self.group_cols,
            "label_col": self.label_col,
            "time_col": self.time_col,
            "channel_cols": self.channel_cols,
        }
        if self.classes_original is not None:
            data["classes_original"] = self.classes_original
        if self.class_to_idx is not None:
            data["class_to_idx"] = self.class_to_idx
        return data


def build_label_mapping_from_values(y_values: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    unique_values = sorted({int(value) for value in y_values})
    if unique_values == [1, 2]:
        class_to_idx = {1: 0, 2: 1}
    else:
        class_to_idx = {value: idx for idx, value in enumerate(unique_values)}
    idx_to_class = {idx: value for value, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def _prepare_trials(
    csv_path: str | Path,
    train_channel_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], str, Optional[str], List[str]]:
    df = pd.read_csv(csv_path)
    group_cols = detect_group_cols(df)
    label_col = detect_label_col(df)
    time_col = detect_time_col(df)

    if train_channel_cols is None:
        channel_cols = detect_channel_cols(df, label_col=label_col, time_col=time_col)
    else:
        missing = [column for column in train_channel_cols if column not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {csv_path} nao contem todos os canais esperados. Exemplo de faltantes: {missing[:10]}"
            )
        channel_cols = list(train_channel_cols)

    sort_cols = list(group_cols)
    if time_col:
        sort_cols.append(time_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, group_cols, label_col, time_col, channel_cols


def _collect_trials(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    label_col: str,
    channel_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Any], int, int]:
    X_list: List[np.ndarray] = []
    y_list_raw: List[int] = []
    group_keys: List[Any] = []

    for group_key, group_df in df.groupby(list(group_cols), sort=False):
        label = group_df[label_col].iloc[0]
        if group_df[label_col].nunique() > 1:
            label = group_df[label_col].mode().iloc[0]

        X_list.append(group_df[list(channel_cols)].to_numpy(dtype=np.float32))
        y_list_raw.append(int(label))
        group_keys.append(group_key)

    lengths = [array.shape[0] for array in X_list]
    min_len = int(np.min(lengths))
    max_len = int(np.max(lengths))

    if min_len != max_len:
        X_list = [array[:min_len, :] for array in X_list]

    X = np.stack([array.T for array in X_list], axis=0).astype(np.float32)
    y_raw = np.array(y_list_raw, dtype=np.int64)
    return X, y_raw, group_keys, min_len, max_len


def build_trials_from_csv(
    csv_path: str | Path,
    train_channel_cols: Optional[Sequence[str]] = None,
    class_to_idx: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, TrialsInfo]:
    df, group_cols, label_col, time_col, channel_cols = _prepare_trials(
        csv_path,
        train_channel_cols=train_channel_cols,
    )
    X, y_raw, _, min_len, max_len = _collect_trials(df, group_cols, label_col, channel_cols)

    if class_to_idx is None:
        class_to_idx = {int(value): idx for idx, value in enumerate(np.unique(y_raw))}

    y = np.array([class_to_idx[int(value)] for value in y_raw], dtype=np.int64)
    info = TrialsInfo(
        n_trials=int(X.shape[0]),
        n_channels=int(X.shape[1]),
        n_time=int(X.shape[2]),
        min_len=min_len,
        max_len=max_len,
        group_cols=list(group_cols),
        label_col=label_col,
        time_col=time_col,
        channel_cols=list(channel_cols),
        classes_original=[int(value) for value in np.unique(y_raw)],
        class_to_idx={int(key): int(value) for key, value in class_to_idx.items()},
    )
    return X, y, info


def load_trials_with_metadata(
    csv_path: str | Path,
    train_channel_cols: Sequence[str],
    class_to_idx: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any], Dict[str, Any]]:
    df, group_cols, label_col, time_col, channel_cols = _prepare_trials(
        csv_path,
        train_channel_cols=train_channel_cols,
    )
    X, y_raw, group_keys, min_len, max_len = _collect_trials(df, group_cols, label_col, channel_cols)

    if class_to_idx is None:
        class_to_idx, idx_to_class = build_label_mapping_from_values(y_raw)
    else:
        idx_to_class = {value: key for key, value in class_to_idx.items()}

    y = np.array([class_to_idx[int(value)] for value in y_raw], dtype=np.int64)
    info = {
        "n_trials": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_time": int(X.shape[2]),
        "min_len": min_len,
        "max_len": max_len,
        "group_cols": list(group_cols),
        "label_col": label_col,
        "time_col": time_col,
        "channel_cols": list(channel_cols),
        "idx_to_class": idx_to_class,
        "class_to_idx": {int(key): int(value) for key, value in class_to_idx.items()},
    }
    return X, y, y_raw, group_keys, info
