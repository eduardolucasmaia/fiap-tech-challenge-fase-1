from __future__ import annotations

"""Chama os scripts da raiz do trabalho em ordem: de ingestão e EDA ao treino, métricas no teste e XAI.
Útil para reproduzir tudo de uma vez antes de preencher o relatório ou a entrega."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def run_step(script_name: str) -> None:
    """Roda o arquivo com o mesmo Python deste processo; se o subprocesso falhar, o pipeline para aqui."""
    script_path = SRC_DIR / script_name
    print(f"\n>>> {script_name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise SystemExit(f"Parou no passo: {script_name} (código {result.returncode})")


def main() -> None:
    for step in [
        "data.py",
        "eda.py",
        "preprocess.py",
        "train.py",
        "evaluate.py",
        "explain.py",
    ]:
        run_step(step)
    print("\nPipeline executado com sucesso.")


if __name__ == "__main__":
    main()
