# Relatório técnico — Tech Challenge - Fase 1

## 1. Problema e objetivo

- O enunciado trata de apoio à triagem em saúde da mulher.
- O trabalho implementa classificação de risco câncer de mama a partir do Wisconsin: `malignant` / `benign`.
- A métrica que orienta a escolha do modelo no treino é o **recall** (foco em reduzir falsos negativos nesse primeiro corte).
- O output da pipeline é apoio à triagem, não diagnóstico isolado nem substituto de decisão clínica.

## 2. O que foi validado no workspace (e onde)

- **Discussões da EDA:** implementadas em `src/eda.py` e registradas em `artifacts/metrics/eda_profile.json` + figuras.
- **Estratégias de pré-processamento:** implementadas em `src/preprocess.py` (split estratificado, encoding, persistência do split) e refletidas em `artifacts/data_split.joblib`.
- **Modelos usados e por quê:** implementados em `src/train.py` (quatro famílias, pipelines e grids por modelo; seleção por recall).
- **Resultados e interpretação:** implementados em `src/evaluate.py` e `src/explain.py`, com saída em `artifacts/metrics/`, `artifacts/figures/` e `artifacts/xai/`.
- **Orquestração reprodutível:** `src/run_pipeline.py` executa a sequência `data` → `eda` → `preprocess` → `train` → `evaluate` → `explain`.

## 3. Dados utilizados

- **Base:** Breast Cancer Wisconsin (Diagnostic), UCI, trazida via scikit-learn.
- **CSV no repo:** `data/breast_cancer_wisconsin.csv`
- **Metadados (fonte, dimensão, alvo, nota de ética/uso):** `data/dataset_metadata.json`

## 4. Análise exploratória (EDA)

- Verificação de faltantes e de duplicatas
- Distribuição do alvo (balanceamento aproximado)
- Estatística descritiva
- Matriz de correlação (redundância e possíveis multicolinearidades grosseiras)

**Principais achados (artefatos atuais)**

- 569 linhas e 31 colunas (30 features + `diagnosis`)
- 0 valores ausentes e 0 duplicatas
- Distribuição de classe: 357 `benign` e 212 `malignant`
- Leitura prática: base é utilizável sem imputação nesse recorte, mas com leve desbalanceamento entre classes

**Saídas principais**

- `artifacts/metrics/eda_profile.json`
- `artifacts/figures/class_distribution.png`
- `artifacts/figures/correlation_heatmap.png`

## 5. Pré-processamento

- Separação de `X` (explicativas) e alvo
- `LabelEncoder` no alvo
- `train_test_split` 80/20, `stratify`, `random_state` fixo; persistido em `artifacts/data_split.joblib`
- Treino/ajuste só com treino, sem o teste na hora de escolher modelo / hiperparâmetros (reduz fuga de informação)
- `StandardScaler` no `Pipeline` nas famílias em que a escala altera a geometria (ex.: regressão logística, k-NN, SVM); *random forest* no fluxo fica sem essa etapa, conforme o código

## 6. Modelagem (modelos usados e por quê)

- **Regressão logística:** baseline forte para tabular, interpretável e com probabilidade (`predict_proba`).
- **k-NN:** compara uma abordagem por vizinhança/distância, útil para fronteira não linear.
- **Random forest:** captura interação e não linearidade sem precisar escalar.
- **SVM:** modelo de margem, com alternativa linear e RBF no grid.
- **Estratégia de comparação:** `cross_validate` + `GridSearchCV` por família, todos com seleção por `recall`.

**Ranking por recall do grid (artefatos atuais)**

- `logistic_regression`: 0.9588 (vencedor)
- `svm`: 0.9471
- `random_forest`: 0.9412
- `knn`: 0.9353

**Artefatos relevantes**

- `artifacts/metrics/cv_results.csv` — resumo de CV e grid
- `artifacts/metrics/model_selection.json` — modelo escolhido e score de seleção
- `artifacts/models/best_model.joblib` — pipeline final salvo

## 7. Avaliação (conjunto de teste)

- Métricas: accuracy, precision, recall, F1
- Com `predict_proba`: também ROC-AUC e curva ROC
- Relatório por classe e matriz de confusão (tab + figura)

**Resultados do teste (artefatos atuais)**

- Accuracy: 0.9737
- Precision: 0.9756
- Recall: 0.9524
- F1-score: 0.9639
- ROC-AUC: 0.9954

**Interpretação direta**

- O modelo final mantém recall alto no hold-out (coerente com o critério da seleção no treino).
- Matriz de confusão: 71 benignos corretos, 1 falso positivo, 40 malignos corretos e 2 falsos negativos.
- Para triagem, os 2 falsos negativos continuam sendo o risco principal a monitorar em evolução do trabalho.

**Ficheiros / figuras**

- `artifacts/metrics/test_metrics.json`
- `artifacts/metrics/classification_report.json`
- `artifacts/metrics/confusion_matrix.csv`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/figures/roc_curve.png`

## 8. Explicabilidade

- Importância “nativa”: `coef_` (linear) ou `feature_importances_` (florestas), no limite do que o estimador oferece
- *Permutation importance* (no teste, *scoring* = recall, alinhado à escolha do `train.py`)
- Tentativa de **SHAP**; se faltar a lib ou o passo gráfico falhar, o estado fica em `xai_summary.json` (script não aborta a pipeline inteira)

**Estado atual dos artefatos de XAI**

- `native_feature_importance_rows`: 30
- `permutation_feature_importance_rows`: 30
- `shap_status`: SHAP gerado com sucesso (`beeswarm` + CSV)

**Onde fica o resultado**

- `artifacts/xai/feature_importance_native.csv`
- `artifacts/xai/feature_importance_permutation.csv`
- `artifacts/figures/feature_importance_permutation.png`
- `artifacts/figures/shap_beeswarm.png` e `artifacts/xai/shap_summary.csv` se o SHAP correr
- `artifacts/metrics/xai_summary.json` — resumo, incl. mensagem se SHAP não tiver ido

## 9. Discussão crítica

- Papel: **triagem** e priorização, não autonomia diagnóstica
- Risco a mitigar: falsos negativos → reforça o critério de `recall` no treino
- Uso real: exigiria outra coorte, validação externa e acompanhamento, à parte deste *challenge*
- Limites do estudo: dataset clássico, possível *gap* de domínio e *drift* ao longo do tempo
- Continua a valer a leitura de ética no repositório e o notebook (se for parte da entrega) como comprovação

## 10. Reprodutibilidade

Na raiz do clone, entre no código do desafio e suba a pipeline; os artefatos sobem em `artifacts/`.

```bash
cd tech-challenge-fase1
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r requirements.txt
python src/run_pipeline.py
```

- Os caminhos de `data/` e `artifacts/` partem de `tech-challenge-fase1` como diretório de trabalho.
