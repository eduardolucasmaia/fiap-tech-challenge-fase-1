from __future__ import annotations

"""Pré-processamento: separa `X` e `y`, codifica o alvo e faz hold-out 80/20 estratificado
(mantém a proporção de benign/malignant em treino e teste). Grava tudo em `data_split.joblib` para
`train` / `evaluate` / `explain` reutilizarem o mesmo corte e o mesmo encoding."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLIT_PATH = ARTIFACTS_DIR / "data_split.joblib"


def load_data() -> pd.DataFrame:
    """Lê o CSV produzido por `data.py` (precisa ter coluna `diagnosis`)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei o dataset em {DATA_PATH}. Rode src/data.py antes."
        )
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame) -> dict:
    """LabelEncoder deixa 0/1; `stratify` evita desbalancear treino e teste por acaso (sorte de split)."""
    y = df["diagnosis"].copy()
    X = df.drop(columns=["diagnosis"]).copy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "target_mapping": {label: int(code) for code, label in enumerate(encoder.classes_)},
        "class_names": list(encoder.classes_),
    }
    return payload


def save_split(payload: dict) -> None:
    """Grava o joblib e um JSON enxuto com tamanho dos conjuntos e NaNs (duplicata no dado bruto: ver EDA)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, SPLIT_PATH)

    data_quality = {
        "train_shape": [int(payload["X_train"].shape[0]), int(payload["X_train"].shape[1])],
        "test_shape": [int(payload["X_test"].shape[0]), int(payload["X_test"].shape[1])],
        "train_missing_values": int(payload["X_train"].isna().sum().sum()),
        "test_missing_values": int(payload["X_test"].isna().sum().sum()),
        "duplicate_rows_raw_data": 0,
    }
    (ARTIFACTS_DIR / "data_quality_report.json").write_text(
        json.dumps(data_quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    df = load_data()
    payload = split_data(df)
    save_split(payload)
    print(f"Split salvo em: {SPLIT_PATH}")
    print("Mapeamento rótulo → código:", json.dumps(payload["target_mapping"], ensure_ascii=False))


if __name__ == "__main__":
    main()
