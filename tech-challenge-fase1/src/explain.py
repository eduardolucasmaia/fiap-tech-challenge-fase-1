from __future__ import annotations

"""Explicabilidade global: importância nativa do modelo, permutação (alinhada ao recall do treino) e
SHAP quando a lib estiver disponível. Saídas em `artifacts/xai/` e gráficos em `artifacts/figures/`."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
XAI_DIR = ARTIFACTS_DIR / "xai"
SPLIT_PATH = ARTIFACTS_DIR / "data_split.joblib"


def load_inputs() -> tuple[dict[str, Any], Any]:
    """Lê o mesmo split e `best_model` que o `evaluate.py` usa."""
    split = joblib.load(SPLIT_PATH)
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    return split, model


def model_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Importância “de fábrica”: `feature_importances_` em modelos de árvore, ou |`coef_`| no linear. Senão, DataFrame vazio."""
    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_based_importance(
    model, X_test: pd.DataFrame, y_test, feature_names: list[str]
) -> pd.DataFrame:
    """Mede quão cai o recall se cada coluna for embaralhada (média sobre 10 repetições)."""
    p = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="recall",
    )
    return (
        pd.DataFrame({"feature": feature_names, "importance": p.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def generate_shap(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: list[str]
) -> str:
    """Tenta beeswarm + CSV; se faltar `shap` ou der erro, devolve string curta (vai pro JSON do trabalho)."""
    try:
        import shap

        XAI_DIR.mkdir(parents=True, exist_ok=True)
        sample_train = X_train.sample(min(200, len(X_train)), random_state=42)
        sample_test = X_test.sample(min(120, len(X_test)), random_state=42)

        explainer = shap.Explainer(model.predict, sample_train)
        shap_values = explainer(sample_test)
        shap.plots.beeswarm(shap_values, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        ranking = np.abs(shap_values.values).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": ranking})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_df.to_csv(XAI_DIR / "shap_summary.csv", index=False)
        return "SHAP: ok (beeswarm + CSV em artifacts/)."
    except Exception as exc:  # noqa: BLE001
        return f"SHAP não rodou: {exc!s}"


def main() -> None:
    split, model = load_inputs()
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_test = split["y_test"]
    feature_names = split["feature_names"]

    XAI_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    native_importance = model_feature_importance(model, feature_names)
    perm_importance = permutation_based_importance(model, X_test, y_test, feature_names)

    native_importance.to_csv(XAI_DIR / "feature_importance_native.csv", index=False)
    perm_importance.to_csv(XAI_DIR / "feature_importance_permutation.csv", index=False)

    plt.figure(figsize=(8, 6))
    top = perm_importance.head(15).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title("Importância por permutação (queda de recall)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_permutation.png", dpi=150)
    plt.close()

    shap_status = generate_shap(model, X_train, X_test, feature_names)

    xai_summary = {
        "native_feature_importance_rows": int(native_importance.shape[0]),
        "permutation_feature_importance_rows": int(perm_importance.shape[0]),
        "shap_status": shap_status,
    }
    (METRICS_DIR / "xai_summary.json").write_text(
        json.dumps(xai_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Resumo XAI (também em metrics/xai_summary.json):")
    print(json.dumps(xai_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
