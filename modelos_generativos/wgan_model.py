
"""
WGAN-GP (condicional) com comparação justa vs DDPM:

- Treina APENAS no TRAIN
- Avalia SOMENTE no TEST
- Mesmas métricas (MSE, Corr, PSD cosine, mu/beta)
- Reprodutível (seed + torch.Generator)
- CLI para rodar localmente e compartilhar no GitHub

Outputs (padrão igual ao DDPM/Colab):
- models/modelo_wgan_wbcic_<CANAL>.pth
- results/avaliacao_resultados_wgan_wbcic_TEST.csv
- results/real_vs_synthetic_wgan_wbcic_TEST_todos_canais.csv
"""

import os
import json
import random
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.signal import welch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_utils import (
    load_diretrizes as shared_load_diretrizes,
    seed_everything as shared_seed_everything,
)


# ============================================================
# Utils: YAML opcional (sem depender de pyyaml obrigatoriamente)
# ============================================================
def _try_load_yaml(path: str) -> Optional[dict]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def load_diretrizes(path: str) -> List[Dict]:
    """
    Aceita YAML ou JSON.
    Formato esperado (YAML):
      diretrizes:
        - canal_alvo: "C3"
          canais_entrada: ["C1","C5","CP3"]
    JSON:
      {"diretrizes":[...]} ou uma lista direta [...]
    """
    if not path:
        raise ValueError("Você precisa passar --diretrizes <arquivo.yaml|json>")

    data = _try_load_yaml(path)
    if data is not None:
        diretrizes = data.get("diretrizes") if isinstance(data, dict) else None
        if not diretrizes:
            raise ValueError("YAML inválido. Esperado chave 'diretrizes'.")
        return diretrizes

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "diretrizes" in data:
        return data["diretrizes"]
    if isinstance(data, list):
        return data

    raise ValueError("JSON inválido. Use {'diretrizes':[...]} ou uma lista direta.")


# ============================================================
# Determinismo
# ============================================================
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ============================================================
# Métricas
# ============================================================
def compute_basic_metrics(real_flat, synth_flat):
    mse = mean_squared_error(real_flat, synth_flat)
    corr, _ = pearsonr(real_flat, synth_flat)
    return mse, corr


def compute_psd_metrics(real_flat, synth_flat, fs=250.0):
    f_r, Pxx = welch(real_flat, fs=fs, nperseg=1024)
    f_s, Pyy = welch(synth_flat, fs=fs, nperseg=1024)

    min_len = min(len(Pxx), len(Pyy))
    Pxx = Pxx[:min_len]
    Pyy = Pyy[:min_len]
    freqs = f_r[:min_len]

    num = np.dot(Pxx, Pyy)
    den = (np.linalg.norm(Pxx) * np.linalg.norm(Pyy) + 1e-12)
    psd_cos_sim = num / den

    def band_power(freqs_, psd_, f_low, f_high):
        mask = (freqs_ >= f_low) & (freqs_ <= f_high)
        if not np.any(mask):
            return 0.0
        return np.trapezoid(psd_[mask], freqs_[mask])

    mu_real = band_power(freqs, Pxx, 8.0, 12.0)
    mu_synth = band_power(freqs, Pyy, 8.0, 12.0)
    beta_real = band_power(freqs, Pxx, 13.0, 30.0)
    beta_synth = band_power(freqs, Pyy, 13.0, 30.0)

    def rel_error(true, est):
        if true == 0:
            return np.nan
        return (est - true) / true

    return {
        "psd_cosine_similarity": psd_cos_sim,
        "mu_power_real": mu_real,
        "mu_power_synth": mu_synth,
        "mu_power_rel_error": rel_error(mu_real, mu_synth),
        "beta_power_real": beta_real,
        "beta_power_synth": beta_synth,
        "beta_power_rel_error": rel_error(beta_real, beta_synth),
    }


# ============================================================
# WGAN-GP – Modelos
# ============================================================
class Generator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, output_dim: int, n_features: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + cond_dim, n_features * 4),
            nn.ReLU(),
            nn.Linear(n_features * 4, n_features * 8),
            nn.ReLU(),
            nn.Linear(n_features * 8, output_dim),
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, n_features: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, n_features * 8),
            nn.ReLU(),
            nn.Linear(n_features * 8, n_features * 4),
            nn.ReLU(),
            nn.Linear(n_features * 4, 1),
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)


def gradient_penalty(discriminator, real_data, fake_data, cond, lambda_gp: float,
                     device: torch.device, torch_gen: torch.Generator):
    batch_size = real_data.size(0)
    eps = torch.rand(batch_size, 1, device=device, generator=torch_gen).expand_as(real_data)
    interpolated = eps * real_data + (1 - eps) * fake_data
    interpolated = interpolated.detach().requires_grad_(True)

    prob_interpolated = discriminator(interpolated, cond)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * lambda_gp
    return gp


def train_wgan_gp(
    generator,
    discriminator,
    dataloader,
    num_epochs: int,
    z_dim: int,
    lambda_gp: float,
    n_critic: int,
    lr: float,
    betas: Tuple[float, float],
    device: torch.device,
    torch_gen: torch.Generator,
):
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for batch_inputs, batch_targets in tqdm(
            dataloader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            B = batch_inputs.size(0)
            cond = batch_inputs
            real_data = batch_targets

            for _ in range(n_critic):
                z = torch.randn(B, z_dim, device=device, generator=torch_gen)
                fake_data = generator(z, cond)

                real_validity = discriminator(real_data, cond)
                fake_validity = discriminator(fake_data.detach(), cond)

                gp = gradient_penalty(
                    discriminator=discriminator,
                    real_data=real_data,
                    fake_data=fake_data.detach(),
                    cond=cond,
                    lambda_gp=lambda_gp,
                    device=device,
                    torch_gen=torch_gen,
                )

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp

                optimizer_D.zero_grad(set_to_none=True)
                d_loss.backward()
                optimizer_D.step()

                epoch_d_loss += float(d_loss.item())

            z = torch.randn(B, z_dim, device=device, generator=torch_gen)
            fake_data = generator(z, cond)
            g_loss = -torch.mean(discriminator(fake_data, cond))

            optimizer_G.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += float(g_loss.item())

        avg_d_loss = epoch_d_loss / max(1, len(dataloader))
        avg_g_loss = epoch_g_loss / max(1, len(dataloader))
        print(f"[Epoch {epoch+1}/{num_epochs}] D Loss: {avg_d_loss:.6f}, G Loss: {avg_g_loss:.6f}")


@torch.no_grad()
def generate_synthetic_data(
    generator,
    conditioning_inputs: torch.Tensor,
    z_dim: int,
    batch_size: int,
    device: torch.device,
    torch_gen: torch.Generator,
):
    generator.eval()
    N = conditioning_inputs.size(0)
    outs = []
    for i in range(0, N, batch_size):
        cond = conditioning_inputs[i:i + batch_size].to(device)
        B = cond.size(0)
        z = torch.randn(B, z_dim, device=device, generator=torch_gen)
        syn = generator(z, cond)
        outs.append(syn.detach().cpu())
    return torch.cat(outs, dim=0).numpy()


# ============================================================
# Dataset / segmentação
# ============================================================
META_COLS_DEFAULT = ["dataset_type", "patient", "session", "epoch", "time", "label", "label_name"]


class EEGDatasetFlat(torch.utils.data.Dataset):
    def __init__(self, inputs_flat, targets_flat):
        self.inputs = torch.tensor(inputs_flat, dtype=torch.float32)
        self.targets = torch.tensor(targets_flat, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def _available_meta_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    return [c for c in meta_cols if c in df.columns]


def build_segments_from_wbcic_flat(
    df: pd.DataFrame,
    canais_entrada: List[str],
    canal_alvo: str,
    seq_length: int,
    max_segments: int,
    patients_to_use: Optional[List[int]],
    meta_cols_all: List[str],
):
    seg_inputs, seg_targets, seg_meta = [], [], []
    meta_cols_use = _available_meta_cols(df, meta_cols_all)

    df = df.sort_values(["patient", "epoch", "time"])
    grouped = df.groupby(["patient", "epoch"])

    for (patient, epoch), g in grouped:
        if patients_to_use is not None and patient not in patients_to_use:
            continue

        X = g[canais_entrada].values
        y = g[[canal_alvo]].values
        meta = g[meta_cols_use].values if meta_cols_use else None

        T_total = X.shape[0]
        if T_total < seq_length:
            continue

        n_seg = T_total // seq_length
        if n_seg <= 0:
            continue

        X = X[: n_seg * seq_length].reshape(n_seg, seq_length, len(canais_entrada))
        y = y[: n_seg * seq_length].reshape(n_seg, seq_length, 1)

        seg_inputs.append(X)
        seg_targets.append(y)

        if meta is not None:
            meta = meta[: n_seg * seq_length].reshape(n_seg, seq_length, meta.shape[1])
            seg_meta.append(meta)

        total_segs = sum(arr.shape[0] for arr in seg_inputs)
        if total_segs >= max_segments:
            break

    if not seg_inputs:
        raise ValueError("Nenhum segmento foi gerado. Verifique SEQ_LENGTH, filtros e colunas.")

    inputs_segments = np.concatenate(seg_inputs, axis=0)    # (N, T, C_in)
    targets_segments = np.concatenate(seg_targets, axis=0)  # (N, T, 1)

    N, T, C_in = inputs_segments.shape
    inputs_flat = inputs_segments.reshape(N, T * C_in)
    targets_flat = targets_segments.reshape(N, T)

    if seg_meta:
        meta_segments = np.concatenate(seg_meta, axis=0)
        meta_flat = meta_segments.reshape(-1, meta_segments.shape[-1])  # (N*T, M)
    else:
        meta_flat = None
        meta_cols_use = []

    return inputs_flat, targets_flat, targets_segments, meta_flat, meta_cols_use


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("WGAN-GP EEG - single file")

    ap.add_argument("--train_csv", "--train-csv", dest="train_csv", required=True, help="Caminho do CSV de treino")
    ap.add_argument("--test_csv", "--test-csv", dest="test_csv", required=True, help="Caminho do CSV de teste")
    ap.add_argument("--diretrizes", required=True, help="Arquivo YAML/JSON com diretrizes")
    ap.add_argument("--out_dir", "--out-dir", dest="out_dir", default="outputs", help="Pasta base de saída (models/results)")

    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--max_segments", type=int, default=12000)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)

    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--n_features", type=int, default=128)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--lambda_gp", type=float, default=10.0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--gen_batch_size", type=int, default=2048,
                    help="Batch para geração no TEST (memória/velocidade)")

    ap.add_argument("--patients", nargs="*", type=int, default=None,
                    help="Opcional: lista de pacientes. Ex: --patients 1 3 4")

    return ap.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    results_dir = out_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    shared_seed_everything(args.seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(args.seed)

    diretrizes = shared_load_diretrizes(args.diretrizes)

    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    avaliacao_resultados = []
    todas_series = {}

    meta_flat_global = None
    meta_cols_use_global = None

    for diretriz in diretrizes:
        canal_alvo = diretriz["canal_alvo"]
        canais_entrada = diretriz["canais_entrada"]

        for col in [canal_alvo] + list(canais_entrada):
            if col not in df_train.columns or col not in df_test.columns:
                raise ValueError(f"Coluna '{col}' não encontrada em TRAIN/TEST.")

        print("\n========================================")
        print(f"WGAN | canal alvo: {canal_alvo}")
        print(f"Entradas: {canais_entrada}")
        print("========================================")

        # TRAIN
        train_inputs_flat, train_targets_flat, _, _, _ = build_segments_from_wbcic_flat(
            df=df_train,
            canais_entrada=canais_entrada,
            canal_alvo=canal_alvo,
            seq_length=args.seq_length,
            max_segments=args.max_segments,
            patients_to_use=args.patients,
            meta_cols_all=META_COLS_DEFAULT,
        )

        cond_dim = train_inputs_flat.shape[1]
        out_dim = train_targets_flat.shape[1]

        train_ds = EEGDatasetFlat(train_inputs_flat, train_targets_flat)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        # MODELOS
        G = Generator(z_dim=args.z_dim, cond_dim=cond_dim, output_dim=out_dim, n_features=args.n_features).to(device)
        D = Discriminator(input_dim=out_dim, cond_dim=cond_dim, n_features=args.n_features).to(device)

        # TREINO
        train_wgan_gp(
            generator=G,
            discriminator=D,
            dataloader=train_loader,
            num_epochs=args.epochs,
            z_dim=args.z_dim,
            lambda_gp=args.lambda_gp,
            n_critic=args.n_critic,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            device=device,
            torch_gen=torch_gen,
        )

        # SALVAR (nome igual ao seu padrão)
        model_path = models_dir / f"modelo_wgan_wbcic_{canal_alvo}.pth"
        torch.save(G.state_dict(), model_path)
        print(f"Gerador salvo: {model_path}")

        # TEST
        test_inputs_flat, _, test_targets_segments, meta_flat, meta_cols_use = build_segments_from_wbcic_flat(
            df=df_test,
            canais_entrada=canais_entrada,
            canal_alvo=canal_alvo,
            seq_length=args.seq_length,
            max_segments=args.max_segments,
            patients_to_use=args.patients,
            meta_cols_all=META_COLS_DEFAULT,
        )

        if meta_flat is not None and meta_cols_use:
            if meta_flat_global is None:
                meta_flat_global = meta_flat
                meta_cols_use_global = meta_cols_use

        # GERAÇÃO NO TEST
        conditioning_test = torch.tensor(test_inputs_flat, dtype=torch.float32, device=device)
        synth_test_flat = generate_synthetic_data(
            generator=G,
            conditioning_inputs=conditioning_test,
            z_dim=args.z_dim,
            batch_size=args.gen_batch_size,
            device=device,
            torch_gen=torch_gen,
        )

        real_1d = test_targets_segments.reshape(-1)
        synth_1d = synth_test_flat.reshape(-1)

        mse, corr = compute_basic_metrics(real_1d, synth_1d)
        psd_stats = compute_psd_metrics(real_1d, synth_1d, fs=args.fs)

        print(f"MSE ({canal_alvo}) = {mse:.6f}")
        print(f"Corr ({canal_alvo}) = {corr:.6f}")
        print(f"PSD cosine similarity = {psd_stats['psd_cosine_similarity']:.6f}")
        print(f"Mu rel error = {psd_stats['mu_power_rel_error']:.6f}")
        print(f"Beta rel error = {psd_stats['beta_power_rel_error']:.6f}")

        avaliacao_resultados.append({
            "Canal_Alvo": canal_alvo,
            "Canais_Entrada": ", ".join(canais_entrada),
            "MSE_TEST": mse,
            "Correlacao_TEST": corr,
            "PSD_Cosine_Similarity_TEST": psd_stats["psd_cosine_similarity"],
            "Mu_Power_Real_TEST": psd_stats["mu_power_real"],
            "Mu_Power_Synth_TEST": psd_stats["mu_power_synth"],
            "Mu_Power_Rel_Error_TEST": psd_stats["mu_power_rel_error"],
            "Beta_Power_Real_TEST": psd_stats["beta_power_real"],
            "Beta_Power_Synth_TEST": psd_stats["beta_power_synth"],
            "Beta_Power_Rel_Error_TEST": psd_stats["beta_power_rel_error"],
            "Model_Path": str(model_path),
        })

        todas_series[f"{canal_alvo}_real"] = real_1d
        todas_series[f"{canal_alvo}_synthetic"] = synth_1d

    # ========================================================
    # SALVAR RESULTADOS (nome igual ao seu padrão)
    # ========================================================
    eval_df = pd.DataFrame(avaliacao_resultados)
    out_eval = results_dir / "avaliacao_resultados_wgan_wbcic_TEST.csv"
    eval_df.to_csv(out_eval, index=False)
    print(f"\nAvaliacao salva em: {out_eval}")

    # real_vs_synthetic (nome igual ao seu padrão)
    min_len = min(len(v) for v in todas_series.values())
    signals_df = pd.DataFrame({k: v[:min_len] for k, v in todas_series.items()})

    if meta_flat_global is not None and meta_cols_use_global:
        meta_df = pd.DataFrame(meta_flat_global, columns=meta_cols_use_global)
        min_len = min(min_len, len(meta_df))
        df_pairs_all = pd.concat(
            [
                meta_df.iloc[:min_len].reset_index(drop=True),
                signals_df.iloc[:min_len].reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        df_pairs_all = signals_df.iloc[:min_len].reset_index(drop=True)

    out_pairs = results_dir / "real_vs_synthetic_wgan_wbcic_TEST_todos_canais.csv"
    df_pairs_all.to_csv(out_pairs, index=False)
    print(f"real_vs_synthetic (TEST) salvo em: {out_pairs}")

    # run_config (extra útil)
    with open(results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"run_config.json salvo em: {results_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
