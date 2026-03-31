# Projeto de Geracao e Classificacao de EEG

Este repositorio concentra os scripts usados na dissertacao para:

- geracao sintetica de sinais EEG com `DDPM` e `WGAN-GP`
- classificacao com `EEGNet`, `k-NN`, `Regressao Logistica`, `SVM + CSP` e `U-Net/Conv1D`
- selecao de canais com abordagem `DGAFF-like`

O foco desta versao e facilitar a execucao local sem alterar a estrutura principal dos experimentos.

## Estrutura

- `executar.py`: ponto de entrada unico para rodar qualquer experimento
- `project_utils.py`: utilitarios compartilhados de leitura, seed e validacao
- `diretrizes.yaml`: definicao de canais alvo e canais de entrada
- `dgaff.py`: selecao de canais com masking

### `modelos_generativos/`

- `ddpm_model.py`: geracao com DDPM
- `wgan_model.py`: geracao com WGAN-GP

### `modelos_classificadores/`

- `eegnet_model.py`: classificacao com EEGNet
- `knn_model.py`: classificacao com k-NN
- `regressao_logistica.py`: classificacao com SGD log-loss
- `svm_csp_model.py`: classificacao com SVM e Common Spatial Patterns
- `unet_model.py`: classificacao com U-Net/Conv1D

## Ambiente

Crie um ambiente virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale as dependencias principais:

```powershell
pip install -U pip
pip install -r requirements.txt
```

Se for usar o modelo U-Net:

```powershell
pip install -r requirements-unet.txt
```

Se for usar o modelo SVM + CSP:

```powershell
pip install -r requirements-svm-csp.txt
```

## Formato esperado dos CSVs

Os scripts esperam um CSV em formato longo, com:

- uma coluna de label, como `label`, `y`, `target` ou `class`
- pelo menos uma coluna de agrupamento entre `patient`, `session` e `epoch`
- opcionalmente uma coluna temporal como `time`
- colunas numericas dos canais EEG

Os trials sao reconstruidos automaticamente a partir dessas colunas.

## Execucao rapida

O jeito mais simples de usar o projeto agora e pelo arquivo `executar.py`.

### EEGNet

```powershell
python executar.py eegnet --train-csv data\treino.csv --test-csv data\teste.csv --save-dir runs\eegnet
```

### k-NN

```powershell
python executar.py knn --train-csv data\treino.csv --test-csv data\teste.csv --save-dir runs\knn
```

### Regressao logistica

```powershell
python executar.py logreg --train-csv data\treino.csv --test-csv data\teste.csv --save-dir runs\logreg
```

### SVM + CSP

```powershell
python executar.py svm-csp --train-csv data\treino.csv --test-csv data\teste.csv --save-dir runs\svm_csp
```

### U-Net

```powershell
python executar.py unet --train-csv data\treino.csv --test-csv data\teste.csv --save-dir runs\unet --enable-stage2
```

### WGAN

```powershell
python executar.py wgan --train-csv data\treino.csv --test-csv data\teste.csv --diretrizes diretrizes.yaml --out-dir outputs\wgan
```

### DDPM

```powershell
python executar.py ddpm --train-csv data\treino.csv --test-csv data\teste.csv --diretrizes diretrizes.yaml --out-dir outputs\ddpm
```

### DGAFF-like

```powershell
python executar.py dgaff --ckpt runs\eegnet\eegnet_final.pt --csv data\teste.csv --out runs\ga_masking
```

## Execucao direta

Os scripts tambem podem ser executados pelos novos caminhos:

```powershell
python modelos_classificadores\eegnet_model.py --help
python modelos_classificadores\knn_model.py --help
python modelos_classificadores\regressao_logistica.py --help
python modelos_classificadores\svm_csp_model.py --help
python modelos_classificadores\unet_model.py --help
python modelos_generativos\wgan_model.py --help
python modelos_generativos\ddpm_model.py --help
python dgaff.py --help
```

Os scripts `modelos_generativos/ddpm_model.py` e `modelos_generativos/wgan_model.py` aceitam tanto argumentos com underscore quanto com hifen para os parametros principais. Exemplo:

```powershell
python modelos_generativos\wgan_model.py --train-csv data\treino.csv --test-csv data\teste.csv --out-dir outputs\wgan --diretrizes diretrizes.yaml
```

## Saidas

Cada script salva seus resultados no diretorio configurado por argumento:

- checkpoints dos modelos
- metricas em `.pt`, `.json`, `.csv` ou `.joblib`
- artefatos auxiliares como scalers e historicos

## Observacoes

- O `modelos_classificadores/eegnet_model.py` alinha a ordem dos canais do teste com a usada no treino.
- A logica comum de leitura de CSV e seeds foi centralizada em `project_utils.py`.
