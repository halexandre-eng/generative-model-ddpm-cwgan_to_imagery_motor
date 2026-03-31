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
- `ddpm_model.py`: geracao com DDPM
- `wgan_model.py`: geracao com WGAN-GP
- `eegnet_model.py`: classificacao com EEGNet
- `knn_model.py`: wrapper amigavel para o script de k-NN
- `regressao_logistica.py`: classificacao com SGD log-loss
- `svm_csp_model.py`: classificacao com SVM e Common Spatial Patterns
- `unet_model.py`: classificacao com U-Net/Conv1D
- `dgaff.py`: selecao de canais com masking

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

Os scripts originais continuam funcionando. Exemplos:

```powershell
python eegnet_model.py --help
python knn_model.py --help
python regressao_logistica.py --help
python svm_csp_model.py --help
python unet_model.py --help
python wgan_model.py --help
python ddpm_model.py --help
python dgaff.py --help
```

Os scripts `ddpm_model.py` e `wgan_model.py` aceitam tanto argumentos com underscore quanto com hifen para os parametros principais, por exemplo:

```powershell
python wgan_model.py --train-csv data\treino.csv --test-csv data\teste.csv --out-dir outputs\wgan --diretrizes diretrizes.yaml
```

## Saidas

Cada script salva seus resultados no diretorio configurado por argumento:

- checkpoints dos modelos
- metricas em `.pt`, `.json`, `.csv` ou `.joblib`
- artefatos auxiliares como scalers e historicos

## Observacoes

- O `eegnet_model.py` agora força a ordem dos canais do teste a seguir a ordem usada no treino.
- O arquivo `knn_model.py` foi adicionado para facilitar a chamada direta por nome esperado.
- A logica comum de leitura de CSV e seeds foi centralizada em `project_utils.py`.
