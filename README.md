# Tech Challenge — Fase 1 (saúde da mulher)

Projeto para apoio à triagem clínica em saúde da mulher, com classificação de risco de câncer de mama (`malignant` vs `benign`) com aprendizado supervisionado.

Antes de rodar os `python src/...`, entre na pasta do código: `cd tech-challenge-fase1` (a partir da raiz deste repositório).

## 1) Recorte adotado

- Problema: classificação de exames para sinalizar risco de malignidade.
- Uso: apoio à triagem e priorização, sem substituir decisão médica.
- Métrica principal: `recall`, para reduzir falsos negativos.

## 2) Fonte de dados

- Dataset público Breast Cancer Wisconsin (Diagnostic) (UCI), carregado via `sklearn.datasets.load_breast_cancer`.
- Arquivo local gerado: `data/breast_cancer_wisconsin.csv`.
- Metadados (rastreio / ética): `data/dataset_metadata.json`.

## 3) Estrutura do projeto

- `src/data.py` — ingestão e materialização do dataset.
- `src/eda.py` — análise exploratória (distribuição, correlação, qualidade básica).
- `src/preprocess.py` — split estratificado treino/teste (sem re-treino em cima do teste).
- `src/train.py` — vários modelos, validação cruzada e ajuste de hiperparâmetros.
- `src/evaluate.py` — métricas no teste, matriz de confusão e curva ROC.
- `src/explain.py` — explicabilidade (importância, permutação, SHAP quando rolar).
- `src/run_pipeline.py` — sobe tudo em sequência.

## 4) Requisitos

```bash
cd tech-challenge-fase1
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r requirements.txt
```

## 5) Como executar

### Pipeline de ponta a ponta (recomendado)

```bash
cd tech-challenge-fase1
python src/run_pipeline.py
```

### Passo a passo (ordem)

```bash
cd tech-challenge-fase1
python src/data.py
python src/eda.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/explain.py
```

## 6) Artefatos gerados

- `artifacts/metrics/cv_results.csv`
- `artifacts/metrics/test_metrics.json`
- `artifacts/metrics/classification_report.json`
- `artifacts/metrics/confusion_matrix.csv`
- `artifacts/metrics/xai_summary.json`
- `artifacts/xai/feature_importance_native.csv`
- `artifacts/xai/feature_importance_permutation.csv`
- `artifacts/xai/shap_summary.csv` (quando o SHAP for gerado)
- `artifacts/figures/*.png` (EDA, métricas, XAI)
- `artifacts/models/best_model.joblib`

## 7) Considerações éticas

- Dados públicos e anônimos.
- O modelo apoia triagem; não é diagnóstico sozinho.
- Vale medir desempenho e viés por classe antes de qualquer uso real.

---

## Segue uma visão passo a passo

Abaixo, tudo o que fica abaixo de `tech-challenge-fase1/` (depois de `cd tech-challenge-fase1`).

### 1. O que o trabalho cobre (visão geral)

- **Problema:** classificar a partir de **features numéricas** (dados tabulares) para **priorizar triagem**, **sem substituir** a decisão médica.
- **Métrica de escolha do modelo (treino):** **recall** no `GridSearch` (foco em não perder muitos casos da classe de interesse, típica em triagem).
- **Stack usada no fluxo tabular:** Python, `pandas`, `scikit-learn`, visualização (`matplotlib` / `seaborn`), explicabilidade (SHAP quando instalado, permutation importance). O `requirements.txt` ainda traz outras peças (Jupyter, TensorFlow) para o notebook e para experimentos extras, se forem exigidos no curso.

### 2. Passo 0 — Ambiente e dependências

- `requirements.txt` lista, entre outras: NumPy, Pandas, scikit-learn, matplotlib, seaborn, SHAP, joblib, Jupyter, TensorFlow.
- Crie e ative o venv, depois `pip install -r requirements.txt` (veja a seção **4) Requisitos** no início deste documento; os comandos assumem `cd tech-challenge-fase1`).

### 3. Passo 1 — Dados (`src/data.py`)

1. Carrega o dataset **Breast Cancer Wisconsin (Diagnostic)** com `sklearn.datasets.load_breast_cancer`.
2. Monta um `DataFrame` e cria a coluna **`diagnosis`**: mapeia o `target` do sklearn (0/1) para as strings `malignant` / `benign`.
3. Grava **`data/breast_cancer_wisconsin.csv`**.
4. Grava **`data/dataset_metadata.json`**, com identificação da fonte, dimensão, alvo, rótulos e nota de uso / ética.

**Sem** esse passo, os scripts seguintes que leem o CSV falham.

### 4. Passo 2 — EDA (`src/eda.py`)

1. Lê o CSV gerado no passo 1.
2. Calcula um **perfil** (linhas, colunas, ausentes, duplicatas, distribuição de classes) e salva em **`artifacts/metrics/eda_profile.json`**.
3. Gera figuras em **`artifacts/figures/`**: contagem da classe e **mapa de calor** da correlação entre features numéricas.

Serve para enxergar o balanceamento das classes, a correlação entre atributos e a qualidade básica do quadro de dados.

### 5. Passo 3 — Pré-processamento e split (`src/preprocess.py`)

1. Lê `data/breast_cancer_wisconsin.csv`.
2. Separa **X** (features) e **y** (`diagnosis` como texto).
3. Aplica **`LabelEncoder`** em `y` (rótulo → inteiros) e faz **`train_test_split` estratificado** (20% teste, `random_state=42`, `stratify` no y codificado).
4. Persiste o dicionário completo em **`artifacts/data_split.joblib`**: treino/teste, `feature_names`, mapeamento de classes, etc.
5. Gera **`artifacts/data_quality_report.json`**: formas dos conjuntos, contagem de faltantes e, no dicionário, campo fixo de duplicatas do “raw” (detalhamento alinhado ao EDA, se você comparar os relatórios).

**Importante:** o split é feito **uma única vez**; treino, avaliação e XAI reutilizam esse joblib, evitando vazar informação de teste para ajuste se você respeitar sempre esse artefato.

### 6. Passo 4 — Treino e escolha do modelo (`src/train.py`)

1. Carrega o split a partir de `data_split.joblib`.
2. Define **quatro famílias** em `Pipeline` do sklearn:
   - **Regressão logística, k-NN e SVM** com **`StandardScaler`** no início;
   - **Random Forest** **sem** scaler (partição em árvore no valor bruto).
3. Para **cada** família:
   - Roda **`cross_validate`** no treino (5 folds, estratificados) e registra `accuracy`, `recall`, `F1` e `ROC AUC` médias no cruzamento;
   - Roda **`GridSearchCV`** otimizando a métrica **`recall`**, coerente com a prioridade de triagem;
   - A escolha do **melhor pipeline** (para salvar) corresponde ao **maior recall no grid** entre as famílias, não o baseline isolado — embora o baseline CV entre no CSV.
4. Artefatos típicos:
   - `artifacts/metrics/cv_results.csv` — resumo do CV e melhor `recall` do grid por família, com melhores hiperparâmetros;
   - `artifacts/models/best_model.joblib` — pipeline ajustado com tudo o que o sklearn precisa;
   - `artifacts/metrics/model_selection.json` — qual modelo e qual score de `recall` (grid) venceram.

### 7. Passo 5 — Avaliação no teste (`src/evaluate.py`)

1. Carrega o `data_split` e o **`best_model.joblib`**.
2. Aplica o modelo no **conjunto de teste**; se o pipeline tiver `predict_proba`, calcula também **ROC AUC** e a **curva ROC**.
3. Gera: `test_metrics.json`, `classification_report.json`, `confusion_matrix.csv` em `artifacts/metrics/`.
4. Gera as figuras **`confusion_matrix.png`** e **`roc_curve.png`** em `artifacts/figures/`.

### 8. Passo 6 — Explicabilidade (`src/explain.py`)

1. Carrega novamente o modelo vencedor e o **mesmo** `data_split` (garante paridade com o teste e com o EDA, **sem re-treino** nesse módulo).
2. **Importância “nativa”** (coeficientes de modelos lineares, ou `feature_importances_` em florestas, conforme o pipeline final) → `artifacts/xai/feature_importance_native.csv` (e DataFrame vazio de linhas, se a família vencedora não tiver o atributo clássico de importância).
3. **Permutation importance** no teste, com `scoring="recall"`, e gráfico de barras (top-15) em `feature_importance_permutation.png` em `figures/`.
4. Tenta importar o **SHAP** e rodar o beeswarm + tabela; se a lib não estiver, ou o backend falhar, a mensagem sobre o ocorrido vai para o campo apropriado de **`xai_summary.json`**.

### 9. Orquestração do pipeline completo (`src/run_pipeline.py`)

- Executa, **nesta ordem e como processos do mesmo Python**:
`data.py` → `eda.py` → `preprocess.py` → `train.py` → `evaluate.py` → `explain.py`.
- Se qualquer subprocesso sair com código de erro, o orquestrador chama `SystemExit` (o pipeline interrompe na hora, sem seguir com etapas inconsistentes com os artefatos).

## Estrutura (referência)

```
├── 📁 tech-challenge-fase1
│   ├── 📁 artifacts
│   │   ├── 📁 figures
│   │   │   ├── 🖼️ class_distribution.png
│   │   │   ├── 🖼️ confusion_matrix.png
│   │   │   ├── 🖼️ correlation_heatmap.png
│   │   │   ├── 🖼️ feature_importance_permutation.png
│   │   │   ├── 🖼️ roc_curve.png
│   │   │   └── 🖼️ shap_beeswarm.png
│   │   ├── 📁 metrics
│   │   │   ├── ⚙️ classification_report.json
│   │   │   ├── 📄 confusion_matrix.csv
│   │   │   ├── 📄 cv_results.csv
│   │   │   ├── ⚙️ eda_profile.json
│   │   │   ├── ⚙️ model_selection.json
│   │   │   ├── ⚙️ test_metrics.json
│   │   │   └── ⚙️ xai_summary.json
│   │   ├── 📁 models
│   │   │   └── 📄 best_model.joblib
│   │   ├── 📁 xai
│   │   │   ├── 📄 feature_importance_native.csv
│   │   │   ├── 📄 feature_importance_permutation.csv
│   │   │   └── 📄 shap_summary.csv
│   │   ├── ⚙️ data_quality_report.json
│   │   └── 📄 data_split.joblib
│   ├── 📁 data
│   │   ├── 📄 breast_cancer_wisconsin.csv
│   │   └── ⚙️ dataset_metadata.json
│   ├── 📁 docs
│   │   └── 📕 IADT - Fase 1 - Tech challenge A.pdf
│   ├── 📁 reports
│   │   └── 📝 relatorio_tecnico.md
│   ├── 📁 src
│   │   ├── 🐍 data.py
│   │   ├── 🐍 eda.py
│   │   ├── 🐍 evaluate.py
│   │   ├── 🐍 explain.py
│   │   ├── 🐍 preprocess.py
│   │   ├── 🐍 run_pipeline.py
│   │   └── 🐍 train.py
│   └── 📄 requirements.txt
├── ⚙️ .gitignore
├── 📝 README.md
└── 📄 trabalho_classificacao_cancer_mama_wisconsin.ipynb
```
