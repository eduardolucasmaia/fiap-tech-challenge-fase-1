from __future__ import annotations

"""Classificação tabular no Wisconsin: comparamos quatro famílias (regressão logística, k-NN, floresta
aleatória, SVM). O `StandardScaler` entra no pipeline onde a escala muda a geometria do problema
(logístico, vizinhos, SVM); a floresta fica sem escalar, porque partição por limiar não depende da
mesma forma de magnitude.

O `GridSearchCV` maximiza o recall (prioridade em triagem: não perder muitos casos de maligno).
O melhor pipeline completo fica em `best_model.joblib` para avaliação e explicabilidade abaixo."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
SPLIT_PATH = ARTIFACTS_DIR / "data_split.joblib"


def load_split() -> dict[str, Any]:
    """Lê o `data_split.joblib` produzido pelo `preprocess.py` (X/y já divididos e codificados)."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei o split em {SPLIT_PATH}. Rode src/preprocess.py antes."
        )
    return joblib.load(SPLIT_PATH)


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """Cada chave: um `Pipeline` + dicionário de grid (`model__` = passo final).

    LR: linha de base com `predict_proba` e coeficientes; `class_weight` opcional. KNN: não
    paramétrico, exige `StandardScaler`; o grid varia k, peso e métrica de distância. Random
    forest: captura não linearidade e interação, expõe `feature_importances_`, sem normalizar
    (árvore corta no valor bruto). SVM: RBF e linear, margem, `probability=True` para bater
    com ROC/AUC no `evaluate.py`.
    """
    models: dict[str, tuple[Pipeline, dict]] = {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, random_state=42)),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": [5, 9, 15],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("model", RandomForestClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "svm": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True, random_state=42)),
                ]
            ),
            {
                "model__C": [0.5, 1.0, 2.0],
                "model__kernel": ["rbf", "linear"],
            },
        ),
    }
    return models


def run_cv_and_tuning(X_train: pd.DataFrame, y_train) -> tuple[pd.DataFrame, dict]:
    """Por modelo: `cross_validate` dá um panorama; em seguida o grid refina pelo recall. O vencedor é
    quem tiver maior recall no `GridSearchCV` (não o baseline isolado, embora o baseline vá pro CSV)."""
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []
    best_by_recall = {"name": None, "score": -1.0, "estimator": None}

    for name, (pipeline, param_grid) in models.items():
        baseline = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=make_scorer(recall_score),
            n_jobs=-1,
            cv=cv,
            refit=True,
        )
        grid.fit(X_train, y_train)

        row = {
            "model": name,
            "cv_accuracy_mean": float(baseline["test_accuracy"].mean()),
            "cv_recall_mean": float(baseline["test_recall"].mean()),
            "cv_f1_mean": float(baseline["test_f1"].mean()),
            "cv_roc_auc_mean": float(baseline["test_roc_auc"].mean()),
            "best_recall_from_grid": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        }
        rows.append(row)

        if grid.best_score_ > best_by_recall["score"]:
            best_by_recall = {
                "name": name,
                "score": float(grid.best_score_),
                "estimator": grid.best_estimator_,
            }

    results_df = pd.DataFrame(rows).sort_values(by="best_recall_from_grid", ascending=False)
    return results_df, best_by_recall


def save_artifacts(results_df: pd.DataFrame, best_model: dict) -> None:
    """CSV com CV por linha, joblib do pipeline escolhido, JSON mínimo com o nome e o recall de seleção."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(METRICS_DIR / "cv_results.csv", index=False)
    joblib.dump(best_model["estimator"], MODELS_DIR / "best_model.joblib")

    selection = {
        "selected_model": best_model["name"],
        "selection_metric": "recall",
        "selection_score": best_model["score"],
    }
    (METRICS_DIR / "model_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    payload = load_split()
    X_train = payload["X_train"]
    y_train = payload["y_train"]
    results_df, best_model = run_cv_and_tuning(X_train, y_train)
    save_artifacts(results_df, best_model)
    print("Treino e ajuste concluídos. Tabela (ordenada por recall do grid):")
    print(results_df.to_string(index=False))
    print(
        f"Modelo escolhido: {best_model['name']} | recall (grid) = {best_model['score']:.4f}"
    )


if __name__ == "__main__":
    main()
