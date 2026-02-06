# EEG Synthetic Generation + Classification + DGAFF-like Channel Selection (WBCIC 2C)

Este repositório reúne **(1) modelos generativos** para criação de EEG sintético, **(2) modelos de classificação** para avaliar performance *downstream* e **(3) seleção de canais estilo DGAFF** usando *masking* em um EEGNet já treinado.

A proposta central é permitir **comparação justa** entre abordagens (mesmas convenções de entrada/saída, métricas padronizadas, reprodutibilidade por seed) e, quando aplicável, **evitar data leakage** (treinar apenas no TRAIN e avaliar apenas no TEST para geração sintética).

---

## Visão geral

### 1) Modelos Generativos
- **DDPM (Diffusion Models)** para geração condicional de canais-alvo de EEG.
- **WGAN-GP Condicional** (por canal-alvo) com canais de entrada definidos em diretrizes (YAML/JSON).
- Avaliação padronizada no TEST, com métricas compatíveis para comparação direta entre DDPM e WGAN:
  - **MSE**
  - **Correlação (Pearson)**
  - **PSD cosine similarity**
  - **Erro relativo de potência** nas bandas **µ (8–12 Hz)** e **β (13–30 Hz)**

### 2) Modelos de Classificação
Classificadores implementados para EEG em formato de *trials* (épocas), com esquema **em duas etapas** (Stage 1: seleção em VAL; Stage 2: treino final em TRAIN+VAL e avaliação no TEST):

- **EEGNet (PyTorch)** — entrada `(N, 1, C, T)`, treino em 2 etapas e checkpoints `.pt`.
- **k-NN (scikit-learn)** — trials achatados `(C*T)`, busca de `k`, pipeline com `StandardScaler`.
- **Regressão Logística (SGDClassifier / log_loss)** — trials achatados `(C*T)`, busca de `alpha`, pipeline com `StandardScaler(with_mean=False)`.
- **U-Net / Conv1D (TensorFlow/Keras)** — entrada `(N, T, C)`, normalização por trial com scaler salvo em `.joblib`, Stage 2 opcional.

### 3) Seleção de Canais (DGAFF-like) com GA + Masking
- Seleção de canais **sem alterar a arquitetura** do EEGNet.
- Em vez de remover canais (o que mudaria `n_channels` e pode quebrar o modelo), aplica **masking**: canais não selecionados são **zerados**, preservando o input `(N, 1, C, T)`.
- GA do tipo **(μ+λ)** com:
  - reparo para manter **K canais exatos**
  - **cache** de avaliações reais
  - **surrogate model** (MLPRegressor) para reduzir custo
  - orçamento de avaliações reais por geração (`true_eval_budget`)

---

## Requisitos

Recomendado **Python 3.9+**.

Dependências típicas (variam por script):

- **Básicas**: `numpy`, `pandas`
- **Classificação (sklearn)**: `scikit-learn`, `joblib`
- **EEGNet / DDPM / WGAN (PyTorch)**: `torch`, `tqdm`, `scipy` (para PSD), `pyyaml` (opcional)
- **U-Net / Conv1D**: `tensorflow`

---

## Instalação

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
````

Instale um conjunto mínimo (sklearn + torch):

```bash
pip install -U pip
pip install numpy pandas scikit-learn joblib torch tqdm scipy
```

Se usar diretrizes em YAML:

```bash
pip install pyyaml
```

Se usar o classificador U-Net (Keras):

```bash
pip install tensorflow
```

---

## Formato esperado do CSV (padrão do repositório)

Os scripts esperam EEG em formato “long” (linhas por amostra) e reconstróem *trials* agrupando as linhas.

### Colunas obrigatórias

* **Label**: por padrão `label`

  * Alternativas aceitas em alguns scripts: `y`, `target`, `class`
* **Ao menos 1 coluna de agrupamento** para formar trials:

  * `patient` e/ou `session` e/ou `epoch`

### Colunas opcionais

* `time` (ou equivalentes em alguns detectores): usada para **ordenar amostras** dentro do trial

### Colunas de canais

* Inferidas como **colunas numéricas** que **não** estão no conjunto de metadados
* O trial é montado como matriz `(C, T)` e:

  * EEGNet usa `(N, 1, C, T)`
  * sklearn usa `(N, C*T)`
  * U-Net Conv1D usa `(N, T, C)`

### Compatibilidade entre TRAIN e TEST

* Os scripts verificam consistência do número/ordem de canais.
* Se `T` divergir, normalmente cortam pelo menor `T` para manter compatibilidade.

---

## Organização sugerida

Você pode organizar como quiser, mas um exemplo:

```
.
├── diretrizes.yaml
├── ddpm_model.py
├── wgan_model.py
├── eegnet_model.py
├── knn_model.py
├── regressao_logistica.py
├── unet_modelc.py
├── dgaff.py

```

> Observação: nomes de scripts podem variar. Ajuste conforme os arquivos reais do seu repo.

---

# 1) Modelos Generativos

## 1.1 Diretrizes de canais (YAML/JSON)

Você define **quais canais gerar** (canal-alvo) e **quais canais condicionam** (canais de entrada).

### YAML (recomendado) — `diretrizes.yaml`

```yaml
diretrizes:
  - canal_alvo: "C3"
    canais_entrada: ["C1", "C5", "CP3"]
  - canal_alvo: "C4"
    canais_entrada: ["C2", "C6", "CP4"]
```

---

## 1.2 DDPM (Diffusion Models) — geração condicional (por canal-alvo)

> Este README assume que você tem um script no mesmo padrão do WGAN, com entradas `TRAIN_CSV/TEST_CSV`, `diretrizes` e saída padronizada.
> Ajuste o comando conforme o nome do seu arquivo.

Exemplo:

```bash
python ddpm_eeg_single.py \
  --train_csv data/WBCIC_2C_train_norm.csv \
  --test_csv  data/WBCIC_2C_test_norm.csv \
  --diretrizes diretrizes.yaml \
  --out_dir outputs_ddpm
```

Saídas típicas (exemplo):

```
outputs_ddpm/
├── models/
└── results/
    ├── avaliacao_resultados_ddpm_wbcic_TEST.csv
    ├── real_vs_synthetic_ddpm_wbcic_TEST_todos_canais.csv
    └── run_config.json
```

---

## 1.3 WGAN-GP Condicional — `wgan_model.py`

Treina **somente no TRAIN** e avalia **somente no TEST**, gerando um modelo por **canal-alvo** das diretrizes.

Exemplo básico:

```bash
python wgan_eeg_single.py \
  --train_csv data/WBCIC_2C_train_norm.csv \
  --test_csv  data/WBCIC_2C_test_norm.csv \
  --diretrizes diretrizes.yaml \
  --out_dir outputs_wgan
```

Parâmetros úteis (exemplos):

* `--epochs 40`
* `--batch_size 64`
* `--z_dim 128`
* `--n_critic 5`
* `--lambda_gp 10.0`
* `--max_segments 12000`
* `--patients 1 3 4`
* `--gen_batch_size 2048`

Saídas:

```
outputs_wgan/
├── models/
│   ├── modelo_wgan_wbcic_C3.pth
│   ├── modelo_wgan_wbcic_C4.pth
│   └── ...
└── results/
    ├── avaliacao_resultados_wgan_wbcic_TEST.csv
    ├── real_vs_synthetic_wgan_wbcic_TEST_todos_canais.csv
    └── run_config.json
```

---

# 2) Modelos de Classificação

Todos os classificadores seguem o mesmo conceito:

* reconstrução de *trials* a partir do CSV
* Stage 1: **TRAIN/VAL** (split interno) para selecionar hiperparâmetro ou melhor época
* Stage 2: treino final em **TRAIN+VAL** e avaliação em **TEST**

## 2.1 EEGNet (PyTorch) — `eegnet_model.py`

**Entrada:** `(N, 1, C, T)`.

Etapas:

* **Stage 1**: melhor checkpoint por `val_acc` (com early stopping por `patience`)
* **Stage 2**: re-treina em TRAIN+VAL, salva melhor por menor `test_loss`

  * pode parar quando `test_loss` < `test_loss` do melhor ponto do Stage 1

Exemplo:

```bash
python eegnet_model.py \
  --train-csv data/WBCIC_2C_train_norm.csv \
  --test-csv  data/WBCIC_2C_test_norm.csv \
  --save-dir  runs/eegnet_2c \
  --val-split 0.1 \
  --batch-size 16 \
  --lr 1e-3 \
  --max-epochs-stage1 1500 \
  --patience 200 \
  --max-epochs-stage2 600 \
  --device auto
```

Saídas em `--save-dir`:

* `eegnet_stage1_best.pt`
* `eegnet_final.pt`
* `run_info.json`

---

## 2.2 k-NN (scikit-learn) — `train_knn_2c.py`

**Entrada:** trials achatados `(N, C*T)`.

Etapas:

* **Stage 1**: busca em grade para `k` (melhor por `val_acc`)
* **Stage 2**: treino final em TRAIN+VAL e avaliação no TEST

Exemplo:

```bash
python knn_model.py \
  --train-csv data/WBCIC_2C_train_norm.csv \
  --test-csv  data/WBCIC_2C_test_norm.csv \
  --save-dir  runs/knn_2c \
  --val-split 0.1 \
  --k-grid 1 3 5 7 9 11 15 21 31
```

Saídas:

* `knn_stage1_best.joblib`, `knn_final.joblib`
* `knn_stage1_best.pt`, `knn_final.pt`
* `run_info.json`

---

## 2.3 Regressão Logística (SGDClassifier) — `regressao_logistica.py`

**Entrada:** trials achatados `(N, C*T)`.

Etapas:

* **Stage 1**: busca em grade para `alpha` (melhor por `val_acc`)
* **Stage 2**: treino final em TRAIN+VAL e avaliação no TEST

Exemplo:

```bash
python regressao_logistica.py \
  --train-csv data/WBCIC_2C_train_norm.csv \
  --test-csv  data/WBCIC_2C_test_norm.csv \
  --save-dir  runs/logreg_2c \
  --val-split 0.1 \
  --alpha-grid 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 \
  --class-weight-balanced
```

Saídas:

* `logreg_stage1_best.joblib`, `logreg_final.joblib`
* `logreg_stage1_best.pt`, `logreg_final.pt`
* `run_info.json`

---

## 2.4 U-Net / Conv1D (Keras) — `train_unet_2c.py`

**Entrada:** `(N, T, C)` (Conv1D).

Características:

* Normalização por trial com `StandardScaler` aplicado no vetor `(T*C)`:

  * salva scalers stage1 e final via `joblib`
* Mapeamento fixo de classes (WBCIC 2C): `{1: 0, 2: 1}`
* **Stage 2 é opcional** (`--enable-stage2`)

Exemplo (apenas Stage 1):

```bash
python unet_model.py \
  --train-csv data/WBCIC_2C_train_norm.csv \
  --test-csv  data/WBCIC_2C_test_norm.csv \
  --save-dir  runs/unet_2c \
  --val-split 0.1 \
  --batch-size 32 \
  --lr 1e-3
```

Exemplo (com Stage 2):

```bash
python unet_model.py \
  --train-csv data/WBCIC_2C_train_norm.csv \
  --test-csv  data/WBCIC_2C_test_norm.csv \
  --save-dir  runs/unet_2c \
  --enable-stage2 \
  --max-epochs-stage2 100
```

Saídas:

* `unet_stage1_best.keras`, `unet_final.keras`
* `unet_scaler_stage1.joblib`, `unet_scaler_final.joblib`
* `unet_stage1_best.pt`, `unet_final.pt`
* `run_info.json`

---

# 3) GA DGAFF-like para seleção de canais (Masking) com EEGNet treinado

## Motivação (por que masking?)

Se o EEGNet foi treinado com **todos os canais**, remover canais “de verdade” muda `n_channels` e pode quebrar o modelo (ex.: depthwise conv depende do número de canais).

Com **masking**:

* o modelo permanece **idêntico** (mesmos pesos/arquitetura)
* você mede **quais canais são mais informativos** para aquele classificador
* opcionalmente: você pode **treinar do zero** um EEGNet só com os canais escolhidos (experimento adicional)

## O que o script faz

1. Carrega o checkpoint do EEGNet (`.pt`) contendo `model_state` e `info_train`
2. Reconstrói trials do CSV para `(N, C, T)` e adapta para `(N, 1, C, T)`
3. Avalia baseline com todos os canais
4. Roda o GA **(μ+λ)** selecionando **K canais** com:

   * reparo para manter K exatos
   * cache de avaliações reais
   * surrogate model (MLPRegressor)
   * orçamento de avaliações reais por geração (`true_eval_budget`)
5. Salva resultados (JSON/TXT/CSV) no diretório de saída

## Estrutura esperada do checkpoint (EEGNet)

O checkpoint deve conter pelo menos:

* `model_state` (state_dict do EEGNet)
* `info_train` com:

  * `channel_cols` (lista de canais usados no treino)
  * `n_channels`
  * `n_time`

Exemplo:

```python
ckpt = {
  "model_state": ...,
  "info_train": {
    "channel_cols": ["C3", "Cz", ...],
    "n_channels": 58,
    "n_time": 1000
  }
}
```

## Execução (exemplo)

> Ajuste o nome do script/args conforme seu arquivo real.

```bash
python dgaff.py \
  --ckpt runs/eegnet_2c/eegnet_final.pt \
  --csv  data/WBCIC_2C_test_norm.csv \
  --k 15 \
  --mu 30 \
  --lambda 60 \
  --generations 40 \
  --true_eval_budget 30 \
  --out_dir runs/ga_masking
```

Saídas típicas:

* `best_mask.json` / `best_channels.txt`
* logs/curvas do GA
* resultados em CSV para auditoria

---

## Reprodutibilidade

* Todos os scripts aceitam `--seed` (ou equivalente) e fixam seeds de `random`, `numpy` e frameworks.
* Em GPU, podem ocorrer pequenas variações dependendo de driver/CUDA/PyTorch/TensorFlow.
* Para maior determinismo em PyTorch, use flags/arg de comportamento determinístico (quando disponível).
