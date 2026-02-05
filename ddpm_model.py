#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDPM (treino) + avaliação no TEST (justo vs WGAN)
- Treina no TRAIN_CSV
- Avalia (métricas + real_vs_synthetic) no TEST_CSV
- Determinístico (seeds + cudnn deterministic + geradores fixos)
- Sampling ancestral DDPM (estocástico): adiciona ruído quando t > 0
  (Se quiser determinístico de verdade, use --deterministic_sampling)
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.signal import welch
from scipy.stats import pearsonr


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
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int, base_seed: int):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ============================================================
# Difusão – Beta schedule
# ============================================================
def linear_beta_schedule(timesteps: int, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def make_diffusion_constants(timesteps: int, device: torch.device):
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": posterior_variance,
    }


# ============================================================
# Embedding de tempo
# ============================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ============================================================
# Bloco conv 1D
# ============================================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# U-NET 1D
# ============================================================
class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        self.enc1 = ConvBlock1D(in_channels + time_emb_dim, base_channels)
        self.enc2 = ConvBlock1D(base_channels, base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.enc3 = ConvBlock1D(base_channels * 2, base_channels * 2)
        self.enc4 = ConvBlock1D(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.mid1 = ConvBlock1D(base_channels * 4, base_channels * 4)
        self.mid2 = ConvBlock1D(base_channels * 4, base_channels * 4)

        self.up1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock1D(base_channels * 4, base_channels * 2)
        self.dec2 = ConvBlock1D(base_channels * 2, base_channels * 2)

        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock1D(base_channels * 2, base_channels)
        self.dec4 = ConvBlock1D(base_channels, base_channels)

        self.out_conv = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        B, _, T = x.shape
        t_emb = self.time_mlp(t).unsqueeze(-1).repeat(1, 1, T)
        x = torch.cat([x, t_emb], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.down1(e2)

        e3 = self.enc3(d1)
        e4 = self.enc4(e3)
        d2 = self.down2(e4)

        m = self.mid1(d2)
        m = self.mid2(m)

        u1 = self.up1(m)
        u1 = torch.cat([u1, e4], dim=1)
        u1 = self.dec1(u1)
        u1 = self.dec2(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec3(u2)
        u2 = self.dec4(u2)

        return self.out_conv(u2)


# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * p.detach()
            self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.copy_(self.shadow[name])


# ============================================================
# Dataset / segmentação
# ============================================================
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def build_segments_from_wbcic(
    df: pd.DataFrame,
    canais_entrada: List[str],
    canal_alvo: str,
    meta_cols: List[str],
    seq_length: int,
    max_segments: int,
    patients_to_use: Optional[List[int]] = None,
):
    seg_inputs, seg_targets, seg_meta = [], [], []

    df = df.sort_values(["patient", "epoch", "time"])
    grouped = df.groupby(["patient", "epoch"])

    for (patient, epoch), g in grouped:
        if patients_to_use is not None and patient not in patients_to_use:
            continue

        X = g[canais_entrada].values
        y = g[[canal_alvo]].values
        meta = g[meta_cols].values

        T_total = X.shape[0]
        if T_total < seq_length:
            continue

        n_seg = T_total // seq_length
        if n_seg == 0:
            continue

        X = X[: n_seg * seq_length].reshape(n_seg, seq_length, len(canais_entrada))
        y = y[: n_seg * seq_length].reshape(n_seg, seq_length, 1)
        meta = meta[: n_seg * seq_length].reshape(n_seg, seq_length, len(meta_cols))

        seg_inputs.append(X)
        seg_targets.append(y)
        seg_meta.append(meta)

        total_segs = sum(arr.shape[0] for arr in seg_inputs)
        if total_segs >= max_segments:
            break

    if not seg_inputs:
        raise ValueError("Nenhum segmento foi gerado. Verifique SEQ_LENGTH, filtros e colunas.")

    inputs_segments = np.concatenate(seg_inputs, axis=0)    # (N, T, C_in)
    targets_segments = np.concatenate(seg_targets, axis=0)  # (N, T, 1)
    meta_segments = np.concatenate(seg_meta, axis=0)        # (N, T, M)

    inputs_segments = np.transpose(inputs_segments, (0, 2, 1))    # (N, C_in, T)
    targets_segments = np.transpose(targets_segments, (0, 2, 1))  # (N, 1, T)

    return inputs_segments, targets_segments, meta_segments


# ============================================================
# Treino DDPM
# ============================================================
def train_ddpm(
    model,
    ema_obj,
    optimizer,
    dataloader,
    consts,
    epochs: int,
    timesteps_train: int,
    grad_clip: float,
    use_amp: bool,
    device: torch.device,
    torch_gen: torch.Generator,
):
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast(device_type="cuda", enabled=use_amp)
    else:
        scaler = None
        autocast_ctx = lambda: torch.autocast(device_type="cpu", enabled=False)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            B = batch_inputs.size(0)
            t = torch.randint(0, timesteps_train, (B,), device=device, generator=torch_gen).long()

            noise = torch.randn(
                batch_targets.shape,
                device=batch_targets.device,
                dtype=batch_targets.dtype,
                generator=torch_gen
            )

            sqrt_ac = consts["sqrt_alphas_cumprod"][t].view(B, 1, 1)
            sqrt_om = consts["sqrt_one_minus_alphas_cumprod"][t].view(B, 1, 1)
            x_t = sqrt_ac * batch_targets + sqrt_om * noise

            model_input = torch.cat([batch_inputs, x_t], dim=1)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                noise_pred = model(model_input, t)
                loss = nn.MSELoss()(noise_pred, noise)

            if device.type == "cuda" and use_amp:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if ema_obj is not None:
                ema_obj.update(model)

            epoch_loss += float(loss.item())

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss/len(dataloader):.6f}")


# ============================================================
# Sampling DDPM
# ============================================================
@torch.no_grad()
def ddpm_sample(
    model,
    cond_inputs,             # (N, C_in, T) torch tensor
    consts,
    timesteps_train: int,
    sampling_steps: int,
    device: torch.device,
    torch_gen: torch.Generator,
    deterministic_sampling: bool = False,
):
    model.eval()
    N, _, T = cond_inputs.shape

    if sampling_steps >= timesteps_train:
        steps = list(range(timesteps_train - 1, -1, -1))
    else:
        idx = np.linspace(0, timesteps_train - 1, sampling_steps, dtype=np.int64)
        steps = list(idx[::-1])

    x = torch.randn((N, 1, T), device=device, generator=torch_gen)

    betas = consts["betas"]
    sqrt_recip_alphas = consts["sqrt_recip_alphas"]
    sqrt_one_minus_alphas_cumprod = consts["sqrt_one_minus_alphas_cumprod"]
    posterior_variance = consts["posterior_variance"]

    for t in steps:
        t = int(t)
        t_batch = torch.full((N,), t, device=device, dtype=torch.long)

        model_in = torch.cat([cond_inputs, x], dim=1)
        eps = model(model_in, t_batch)

        beta_t = betas[t]
        sqrt_om = sqrt_one_minus_alphas_cumprod[t]

        x = sqrt_recip_alphas[t] * (x - (beta_t / (sqrt_om + 1e-12)) * eps)

        if t > 0 and not deterministic_sampling:
            var = posterior_variance[t]
            noise = torch.randn(x.shape, device=x.device, generator=torch_gen)
            x = x + torch.sqrt(var) * noise

    return x


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
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("DDPM EEG - single file")
    ap.add_argument("--train_csv", required=True, help="Caminho do CSV de treino")
    ap.add_argument("--test_csv", required=True, help="Caminho do CSV de teste")
    ap.add_argument("--diretrizes", required=True, help="Arquivo YAML/JSON com diretrizes")
    ap.add_argument("--out_dir", default="outputs", help="Pasta de saída (models/results)")

    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--train_max_segments", type=int, default=20000)
    ap.add_argument("--test_max_segments", type=int, default=20000)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--timesteps_train", type=int, default=1000)
    ap.add_argument("--sampling_steps", type=int, default=1000)

    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--deterministic_sampling", action="store_true",
                    help="Se ligado, não adiciona ruído no reverse process (z=0)")

    ap.add_argument("--patients", nargs="*", type=int, default=None)

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

    seed_everything(args.seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(args.seed)

    diretrizes = load_diretrizes(args.diretrizes)

    META_COLS = ['dataset_type', 'patient', 'session', 'epoch', 'time', 'label', 'label_name']

    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    missing_meta_train = [c for c in META_COLS if c not in df_train.columns]
    missing_meta_test  = [c for c in META_COLS if c not in df_test.columns]
    if missing_meta_train:
        raise ValueError(f"Faltam colunas META_COLS no TRAIN: {missing_meta_train}")
    if missing_meta_test:
        raise ValueError(f"Faltam colunas META_COLS no TEST: {missing_meta_test}")

    consts = make_diffusion_constants(args.timesteps_train, device=device)

    avaliacao_resultados = []
    todas_series = {}
    meta_flat_global = None

    for diretriz in diretrizes:
        canal_alvo = diretriz["canal_alvo"]
        canais_entrada = diretriz["canais_entrada"]

        for col in [canal_alvo] + list(canais_entrada):
            if col not in df_train.columns or col not in df_test.columns:
                raise ValueError(f"Coluna '{col}' não encontrada em TRAIN/TEST.")

        print("\n========================================")
        print(f"DDPM | canal alvo: {canal_alvo}")
        print(f"Entradas: {canais_entrada}")
        print("========================================")

        tr_inputs, tr_targets, _ = build_segments_from_wbcic(
            df_train,
            canais_entrada=canais_entrada,
            canal_alvo=canal_alvo,
            meta_cols=META_COLS,
            seq_length=args.seq_length,
            max_segments=args.train_max_segments,
            patients_to_use=args.patients,
        )
        print("Train segmentos:", tr_inputs.shape[0])

        train_ds = EEGDataset(tr_inputs, tr_targets)

        dl_gen = torch.Generator()
        dl_gen.manual_seed(args.seed)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=lambda wid: seed_worker(wid, args.seed),
            generator=dl_gen,
            pin_memory=(device.type == "cuda"),
        )

        in_channels = len(canais_entrada) + 1
        model = UNet1D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=128,
            time_emb_dim=256,
        ).to(device)

        ema_obj = EMA(model, decay=args.ema_decay) if args.use_ema else None
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_ddpm(
            model=model,
            ema_obj=ema_obj,
            optimizer=optimizer,
            dataloader=train_loader,
            consts=consts,
            epochs=args.epochs,
            timesteps_train=args.timesteps_train,
            grad_clip=args.grad_clip,
            use_amp=args.use_amp,
            device=device,
            torch_gen=torch_gen,
        )

        out_model = models_dir / f"ddpm_wbcic_{canal_alvo}.pth"
        torch.save(model.state_dict(), out_model)
        print(f"Modelo salvo: {out_model}")

        out_ema = ""
        ema_model = None
        if args.use_ema and ema_obj is not None:
            ema_model = UNet1D(
                in_channels=in_channels,
                out_channels=1,
                base_channels=128,
                time_emb_dim=256,
            ).to(device)
            ema_model.load_state_dict(model.state_dict())
            ema_obj.copy_to(ema_model)

            out_ema_path = models_dir / f"ddpm_wbcic_{canal_alvo}_EMA.pth"
            torch.save(ema_model.state_dict(), out_ema_path)
            out_ema = str(out_ema_path)
            print(f"Modelo EMA salvo: {out_ema_path}")

        te_inputs, te_targets, te_meta = build_segments_from_wbcic(
            df_test,
            canais_entrada=canais_entrada,
            canal_alvo=canal_alvo,
            meta_cols=META_COLS,
            seq_length=args.seq_length,
            max_segments=args.test_max_segments,
            patients_to_use=args.patients,
        )
        print("Test segmentos:", te_inputs.shape[0])

        meta_flat = te_meta.reshape(-1, te_meta.shape[-1])
        if meta_flat_global is None:
            meta_flat_global = meta_flat

        cond = torch.tensor(te_inputs, dtype=torch.float32, device=device)
        model_to_eval = ema_model if ema_model is not None else model

        synth = ddpm_sample(
            model=model_to_eval,
            cond_inputs=cond,
            consts=consts,
            timesteps_train=args.timesteps_train,
            sampling_steps=args.sampling_steps,
            device=device,
            torch_gen=torch_gen,
            deterministic_sampling=args.deterministic_sampling,
        ).detach().cpu().numpy()

        real = te_targets
        synth_flat = synth.reshape(-1)
        real_flat  = real.reshape(-1)

        mse, corr = compute_basic_metrics(real_flat, synth_flat)
        psd_stats = compute_psd_metrics(real_flat, synth_flat, fs=args.fs)

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
            "Model_Path": str(out_model),
            "Model_EMA_Path": out_ema,
        })

        todas_series[f"{canal_alvo}_real"] = real_flat
        todas_series[f"{canal_alvo}_synthetic"] = synth_flat

    eval_df = pd.DataFrame(avaliacao_resultados)
    out_eval = results_dir / "avaliacao_resultados_ddpm_TEST.csv"
    eval_df.to_csv(out_eval, index=False)
    print(f"\nAvaliacao salva em: {out_eval}")

    if meta_flat_global is None:
        raise RuntimeError("meta_flat_global está vazio. Verifique segmentação e META_COLS.")

    meta_df = pd.DataFrame(meta_flat_global, columns=META_COLS)

    min_len = min(min(len(v) for v in todas_series.values()), len(meta_df))
    meta_df_trim = meta_df.iloc[:min_len].reset_index(drop=True)
    signals_df = pd.DataFrame({k: v[:min_len] for k, v in todas_series.items()})

    out_pairs_all = results_dir / "real_vs_synthetic_ddpm_TEST_todos_canais.csv"
    pd.concat([meta_df_trim, signals_df], axis=1).to_csv(out_pairs_all, index=False)
    print(f"real_vs_synthetic (TEST) salvo em: {out_pairs_all}")

    run_cfg = vars(args)
    with open(results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    print(f"run_config.json salvo em: {results_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
