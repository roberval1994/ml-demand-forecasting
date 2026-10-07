<h1 align="center">🚲 Previsão de Demanda — Projeto de Machine Learning Ponta a Ponta</h1>

<p align="center">
  <b>Um pipeline de ML completo e documentado para previsão de demanda horária</b><br>
  Ingestão de dados → EDA → engenharia de atributos → modelagem → tuning → avaliação → explicabilidade
</p>

🌐 **Idioma / Language:** **Português** | [English](README.md)

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/testes-pytest-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Objetivo

Este repositório é um **projeto de Machine Learning ponta a ponta de referência**, feito
para ser lido, reutilizado e estendido. Ele prevê a **demanda horária** (aluguel de
bicicletas) e demonstra cada etapa de um fluxo de ML profissional com código limpo,
modular e testado.

Serve também como meu **template pessoal de estudo e consulta** para projetos futuros.

## 🧭 Por que demanda de bike-sharing?

O [dataset Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
(UCI) é um problema público de **regressão em série temporal**, tabular e com estrutura
rica: sazonalidade (hora, dia da semana, estação), efeitos climáticos e feriados. Ideal
para demonstrar engenharia de atributos e comparação de modelos sem nenhuma restrição de
licenciamento.

## 🗂️ Estrutura do projeto

```
ml-demand-forecasting/
├── data/
│   ├── raw/                     # Dataset original (baixado, não versionado)
│   └── processed/               # Dados limpos e com atributos (gerados)
├── notebooks/                   # (opcional) notebooks exploratórios
├── src/demand_forecasting/
│   ├── config.py                # Configuração central (caminhos, parâmetros)
│   ├── data.py                  # Download e carregamento do dataset
│   ├── features.py              # Engenharia de atributos (calendário, cíclico, lags)
│   ├── model.py                 # Fábrica de modelos e pipeline de treino
│   ├── evaluate.py              # Métricas e gráficos de diagnóstico
│   └── explain.py               # Importância de atributos / SHAP
├── scripts/
│   └── run_pipeline.py          # Execução ponta a ponta em um comando
├── tests/                       # Testes unitários (pytest)
├── requirements.txt
├── LICENSE
├── README.md                    # Inglês
└── README.pt-BR.md              # Português (este arquivo)
```

## 🔬 Etapas do pipeline

| Etapa | Módulo | O que faz |
|---|---|---|
| **1. Dados** | `data.py` | Baixa o dataset da UCI e carrega em um DataFrame organizado |
| **2. EDA** | `notebooks/` | Distribuições, sazonalidade, correlações |
| **3. Atributos** | `features.py` | Atributos de calendário, **codificação cíclica** (sin/cos), lags e janelas móveis |
| **4. Modelo** | `model.py` | Baseline + Random Forest + Gradient Boosting em um `Pipeline` scikit-learn |
| **5. Tuning** | `model.py` | Busca de hiperparâmetros com validação cruzada temporal |
| **6. Avaliação** | `evaluate.py` | MAE, RMSE, R²; gráficos de resíduos e de previsão |
| **7. Explicabilidade** | `explain.py` | Importância por permutação e (opcional) valores SHAP |

## ⚡ Início rápido

```bash
# 1. Clonar
git clone https://github.com/roberval1994/ml-demand-forecasting.git
cd ml-demand-forecasting

# 2. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 3. Rodar todo o pipeline ponta a ponta
python scripts/run_pipeline.py

# 4. (opcional) rodar os testes
pytest -q
```

O pipeline baixa os dados, constrói os atributos, treina e tuna os modelos, avalia em um
conjunto de teste cronológico e escreve as métricas e gráficos em `outputs/`.

## 📈 O que o pipeline reporta

- Um **ranking** comparando baseline vs. modelos baseados em árvores (MAE / RMSE / R²).
- Gráficos de **resíduos** e de **real vs. previsto**.
- Ranking de **importância de atributos** para interpretar os drivers da demanda.

## 🧪 Testes

```bash
pytest -q
```

Os testes unitários cobrem a lógica de engenharia de atributos e o carregador de dados,
protegendo contra regressões — um hábito que vale manter em qualquer base de ML séria.

## 🧰 Tecnologias

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn` · `SHAP` · `pytest`

## 📚 Princípios de design

- **Modular**: a lógica fica em `src/`, os notebooks apenas orquestram e narram.
- **Reprodutível**: sementes fixas, dependências fixadas, execução em um comando.
- **Consciente do tempo**: a validação respeita a ordem temporal (sem vazamento do futuro).
- **Documentado**: cada módulo tem docstring explicando o "porquê", não só o "como".

## 👤 Autor

**Roberval Gonçalves Moreira Filho** — Cientista de Dados | Analista de Pesquisa Operacional

[![LinkedIn](https://img.shields.io/badge/LinkedIn-robervalOr-blue)](https://www.linkedin.com/in/robervalOr)
[![Lattes](https://img.shields.io/badge/Lattes-CNPq-00599C)](http://lattes.cnpq.br/4394523940603239)
[![GitHub](https://img.shields.io/badge/GitHub-roberval1994-black)](https://github.com/roberval1994)

## 📄 Licença

Distribuído sob a Licença MIT. Veja [LICENSE](LICENSE).
