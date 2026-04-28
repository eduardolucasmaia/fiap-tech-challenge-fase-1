from __future__ import annotations

"""Avalia o modelo no conjunto de teste: métricas, relatório, matriz de confusão e ROC.

Lê o split e o `best_model` gerados no treino; quais modelos foram testados e o porquê do
`GridSearch` ficam em `train.py` (e no extra de CNN, se o trabalho tiver usado)."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = ARTIFACTS_DIR / "models"
SPLIT_PATH = ARTIFACTS_DIR / "data_split.joblib"


def load_inputs() -> tuple[dict[str, Any], Any]:
    """Lê o split salvo (`data_split.joblib`) e o pipeline vencedor (`best_model.joblib`)."""
    split = joblib.load(SPLIT_PATH)
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    return split, model


def evaluate(split: dict, model) -> dict:
    """Predições no `X_test`, métricas, relatório e gráficos em `artifacts/`; retorna o dict de métricas."""
    X_test = split["X_test"]
    y_test = split["y_test"]
    class_names = split["class_names"]

    y_pred = model.predict(X_test)
    # Coluna 1 = P(y=1); com o LabelEncoder usado no preprocess, 1 costuma ser malignant
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )

    matrix = confusion_matrix(y_test, y_pred)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    (METRICS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (METRICS_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        METRICS_DIR / "confusion_matrix.csv", index=True
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusão (teste)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Taxa de falsos positivos")
        plt.ylabel("Taxa de verdadeiros positivos")
        plt.title("Curva ROC (teste)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
        plt.close()

    return metrics


def main() -> None:
    split, model = load_inputs()
    metrics = evaluate(split, model)
    print("Avaliação no teste concluída. Métricas:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
