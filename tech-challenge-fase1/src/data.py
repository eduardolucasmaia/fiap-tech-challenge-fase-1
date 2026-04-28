from __future__ import annotations

"""Carga e exportação do Breast Cancer Wisconsin (Diagnostic).

Usamos o carregador do scikit-learn, trocamos o alvo numérico por rótulos texto,
gravamos o CSV e um JSON com referência e nota breve de uso/ética na pasta `data/`.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def build_dataset() -> pd.DataFrame:
    """Monta o DataFrame: features do sklearn + coluna `diagnosis` (malignant / benign).

    O `load_breast_cancer` devolve `target` com 0 ou 1. Mapeamos para texto e removemos
    `target` para não carregar a mesma informação duas vezes no pipeline.
    """
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.drop(columns=["target"])
    return df


def save_dataset(df: pd.DataFrame) -> Path:
    """Grava o CSV em UTF-8, sem coluna de índice (só features + diagnosis)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "breast_cancer_wisconsin.csv"
    df.to_csv(output_path, index=False)
    return output_path


def save_metadata(df: pd.DataFrame) -> Path:
    """Grava `dataset_metadata.json` ao lado do CSV: fonte, tamanho e alvo (para o relatório).

    `features` é shape[1] - 1 porque uma coluna é o alvo `diagnosis`, não contador
    de preditora no sentido de modelagem.
    """
    metadata_path = DATA_DIR / "dataset_metadata.json"
    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "scikit-learn loader for UCI Breast Cancer Wisconsin dataset",
        "original_reference": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "records": int(df.shape[0]),
        "features": int(df.shape[1] - 1),
        "target": "diagnosis",
        "target_labels": ["malignant", "benign"],
        "ethics_and_lgpd_note": (
            "Dataset é anônimo e público. O modelo é para apoio à triagem clínica e "
            "não substitui decisão médica."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    """Gera o CSV, o JSON e imprime os caminhos; `value_counts` no diagnosis para inspecionar o balanceamento."""
    df = build_dataset()
    dataset_path = save_dataset(df)
    metadata_path = save_metadata(df)
    print(f"Dataset salvo em: {dataset_path}")
    print(f"Metadados em: {metadata_path}")
    print(df["diagnosis"].value_counts().to_string())


if __name__ == "__main__":
    main()
