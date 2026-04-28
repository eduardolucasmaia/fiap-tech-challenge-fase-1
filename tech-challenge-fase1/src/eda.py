from __future__ import annotations

"""Análise exploratória: checagem rápida de qualidade, classe e correlação entre features.

Os resultados vão para `artifacts/metrics/eda_profile.json` e gráficos em
`artifacts/figures/`. O CSV tem de existir (rode `data.py` antes, se ainda não gerou).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não foi encontrado o dataset em {DATA_PATH}. Execute src/data.py antes."
        )

    df = pd.read_csv(DATA_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_distribution": df["diagnosis"].value_counts().to_dict(),
    }
    (METRICS_DIR / "eda_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="diagnosis", palette="Set2", hue="diagnosis", legend=False)
    plt.title("Distribuição da classe (diagnosis)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Heatmap sem o alvo: correlação só entre preditoras numéricas
    corr = df.drop(columns=["diagnosis"]).corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlação entre features (numéricas)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("Resumo (também em artifacts/metrics/eda_profile.json):")
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
