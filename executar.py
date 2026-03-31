from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "ddpm": "ddpm_model.py",
    "wgan": "wgan_model.py",
    "eegnet": "eegnet_model.py",
    "knn": "knn_model",
    "svm-csp": "svm_csp_model.py",
    "logreg": "regressao_logistica.py",
    "unet": "unet_model.py",
    "dgaff": "dgaff.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ponto de entrada unico para executar os scripts do projeto.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python executar.py eegnet --train-csv treino.csv --test-csv teste.csv\n"
            "  python executar.py wgan --train-csv treino.csv --test-csv teste.csv --diretrizes diretrizes.yaml\n"
            "  python executar.py dgaff --ckpt runs/eegnet_2c/eegnet_final.pt --csv teste.csv --out runs/ga"
        ),
    )
    parser.add_argument("comando", choices=sorted(COMMANDS), help="Script que voce quer executar.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Argumentos repassados ao script escolhido.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = ROOT / COMMANDS[args.comando]
    cmd = [sys.executable, str(target), *args.args]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
