# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

## Project Overview

This capstone project develops an end-to-end smart city traffic intelligence solution using the Metro Interstate Traffic Volume dataset.

The project progresses through three connected stages:

1. **Part 1 – Data Analytics**
   - SQL traffic analysis
   - Descriptive statistics and correlation
   - Probability analysis
   - Power BI dashboarding and interpretation

2. **Part 2 – Python Data Engineering and Analytics**
   - Reproducible data-cleaning pipeline
   - Feature engineering
   - Traffic visualisations
   - Command-line traffic analytics application
   - Structured logging and reproducible outputs

3. **Part 3 – Machine Learning and AI**
   - Supervised classification and regression
   - K-means clustering and association-rule mining
   - Neural-network traffic-demand prediction
   - SHAP explainability
   - MLflow experiment tracking
   - Traffic timing recommendations
   - FastAPI deployment simulation
   - Model monitoring and alerting
   - Responsible and sustainable AI analysis

The final solution demonstrates how raw traffic data can progress from analytics and reproducible data engineering into predictive modelling, explainability, decision support, deployment simulation and monitored AI operations.

---

## Dataset and Accident-Risk Proxy

The project uses the **Metro Interstate Traffic Volume** dataset containing approximately 48,000 hourly observations of traffic volume, weather conditions, holiday information and timestamps.

After cleaning, **48,187 observations** are used for modelling.

No real accident dataset was provided with the capstone. Therefore, Part 3 uses the documented assignment proxy for the classification task:

**High risk = High/Severe congestion AND severe or low-visibility weather.**

This proxy is used only to demonstrate the required classification workflow. It must **not** be interpreted as prediction of actual accidents.

Because weather-derived variables are used both in constructing the proxy and as model predictors, the classification results may be optimistic and should be interpreted only as evidence of successful proxy-label modelling.

---

## Repository Structure

```text
smart-city-traffic-capstone/
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
│
├── part1_data_analytics/
│   ├── sql/
│   ├── powerbi/
│   └── part1_insights_report.md
│
├── part2_python/
│   ├── pipeline.py
│   ├── feature_engineering.py
│   ├── visualizations.py
│   ├── traffic_app.py
│   ├── figures/
│   ├── pipeline.log
│   ├── part2_report.md
│   └── README.md
│
├── part3_machine_learning/
│   ├── prepare_ml_data.py
│   ├── supervised_models.py
│   ├── unsupervised_models.py
│   ├── deep_learning_explainability.py
│   ├── mlflow_tracking.py
│   ├── recommendation_system.py
│   ├── deployment_api.py
│   ├── monitoring.py
│   ├── data/
│   ├── figures/
│   ├── models/
│   ├── results/
│   ├── part3_report.md
│   ├── bias_fairness_report.md
│   └── README.md
│
├── README.md
└── .gitignore
```

---

## Tools and Technologies

- Python 3.12
- Jupyter Notebook
- Pandas
- NumPy
- SQLite
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- SHAP
- MLflow
- FastAPI
- Joblib
- Power BI
- Git
- GitHub

---

## Running Part 2

Run commands from the repository root.

### 1. Data cleaning

```powershell
python part2_python\pipeline.py
```

This loads the raw CSV, validates the expected schema, cleans inconsistent or invalid values and produces the cleaned dataset while recording the processing trail in the Part 2 log.

### 2. Feature engineering

```powershell
python part2_python\feature_engineering.py
```

This creates time-based, cyclical, weather, scaled numerical and congestion-category features.

### 3. Visualisations

```powershell
python part2_python\visualizations.py
```

This generates and saves the required traffic-pattern visualisations and their interpretations.

### 4. Mini traffic analytics application

The command-line application supports multiple traffic queries.

Examples:

```powershell
python part2_python\traffic_app.py compare
```

```powershell
python part2_python\traffic_app.py date --date 2017-01-15
```

```powershell
python part2_python\traffic_app.py high --threshold 5500
```

Use:

```powershell
python part2_python\traffic_app.py --help
```

to view the available commands and arguments.

---

## Running Part 3

Run the Part 3 stages from the repository root.

### 1. Prepare the ML dataset

```powershell
python part3_machine_learning\prepare_ml_data.py
```

### 2. Supervised machine learning

```powershell
python part3_machine_learning\supervised_models.py
```

This trains and evaluates:

- Logistic Regression
- Random Forest Classifier
- Linear Regression
- Random Forest Regressor

### 3. Unsupervised machine learning

```powershell
python part3_machine_learning\unsupervised_models.py
```

This performs:

- K-means evaluation and clustering
- Association-rule mining
- Cluster interpretation
- Model/scaler persistence

Candidate values from **k = 2 to k = 8** are evaluated. Among the tested candidates, **k = 8** achieved the highest silhouette score of **0.4822**.

### 4. Deep learning and explainability

```powershell
python part3_machine_learning\deep_learning_explainability.py
```

This trains a feed-forward neural network for traffic-demand prediction and applies SHAP explainability to a comparable tree-based regression model.

### 5. MLflow experiment tracking

```powershell
python part3_machine_learning\mlflow_tracking.py
```

The workflow tracks four supervised model runs and records parameters, metrics, model versions and experiment metadata.

The local SQLite tracking database and local artifact store are intentionally excluded from GitHub because they contain environment-specific paths and generated artifacts.

Sanitized experiment evidence is committed in:

```text
part3_machine_learning/results/mlflow_runs.csv
```

Model-version comparisons are stored in:

```text
part3_machine_learning/results/model_versions.csv
```

### 6. Traffic recommendation system

Use:

```powershell
python part3_machine_learning\recommendation_system.py --help
```

to view the supported recommendation arguments.

The recommendation system focuses on **when to travel**, rather than alternative route selection, because the supplied dataset represents a single traffic corridor.

### 7. FastAPI deployment self-test

```powershell
python part3_machine_learning\deployment_api.py --test
```

The deployment simulation loads the trained Random Forest Regressor and returns a traffic-volume prediction for a sample request.

### 8. Model monitoring

```powershell
python part3_machine_learning\monitoring.py
```

The monitoring simulation uses a chronological split:

- Earliest 70%: monitoring-model training
- Next 10%: unseen baseline/reference window
- Latest 20%: unseen simulated current/live window

Current monitoring evidence:

- Baseline MAE: **263.16**
- Current MAE: **321.24**
- MAE increase: **22.07%**
- Prediction-error threshold: **328.95**
- Temperature SMD: **1.445**
- Overall status: **ALERT / Requires investigation**

The alert is caused by temperature distribution drift. Prediction-error drift remains below its configured threshold.

An alert is treated as an investigation signal, not automatic proof of model failure.

---

## Key Model Results

### Classification

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9567 | 0.8095 | 0.9814 | 0.8872 | 0.9915 |
| Random Forest Classifier | 0.9799 | 0.9357 | 0.9491 | 0.9424 | 0.9964 |

These metrics describe reproduction of the engineered **proxy** target, not real accident prediction.

### Traffic-volume regression

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 813.47 | 0.7260 |
| Random Forest Regressor | 268.67 | 0.9454 |
| Neural Network | 355.93 | 0.9250 |

The Random Forest Regressor produced the strongest traffic-volume result among the implemented regression models.

---

## Logging

Python's `logging` module is used throughout the Python workflow.

Modules use:

```python
logging.getLogger(__name__)
```

Log messages use meaningful levels:

- **DEBUG** – internal values useful for troubleshooting
- **INFO** – normal milestones such as successful loading, processing or saving
- **WARNING** – recoverable but noteworthy conditions, including monitoring alerts
- **ERROR** – failures that prevent the workflow from continuing as planned

Log formatting includes:

- Timestamp
- Log level
- Module name
- Message

Part 2 logging output is stored in:

```text
part2_python/pipeline.log
```

Part 3 execution logs are written locally to:

```text
part3_machine_learning/part3.log
```

The Part 3 runtime log is intentionally excluded from Git because it is regenerated during execution.

`print()` is used only for direct user-facing command-line summaries rather than internal progress reporting.

---

## Reproducing the Analysis

A typical reproduction sequence is:

```text
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required Python packages.
4. Run the Part 2 cleaning pipeline.
5. Run Part 2 feature engineering and visualisation.
6. Run the Part 3 ML-data preparation workflow.
7. Run supervised, unsupervised and deep-learning scripts.
8. Run MLflow tracking.
9. Run recommendation, deployment and monitoring simulations.
10. Review generated results, figures and reports.
```

Random-state values are fixed where appropriate so that model training and evaluation are reproducible.

Generated outputs are stored in the relevant `figures/`, `models/` and `results/` directories.

---

## Power BI

The Power BI dashboard is stored at:

```text
part1_data_analytics/powerbi/traffic_dashboard.pbix
```

It provides interactive views of traffic patterns, weather relationships and selected time-based trends.

---

## Assumptions and Limitations

Important limitations include:

- Some historical years contain only partial date coverage, so raw annual totals are not directly comparable across all years.
- The dataset represents one westbound I-94 traffic corridor and should not be generalised automatically to an entire city or transport network.
- The classification target is a synthetic proxy, not a verified accident outcome.
- Weather variables conceptually overlap with the proxy definition, which may inflate classification performance.
- Historical traffic relationships do not guarantee future traffic conditions.
- Recommendation outputs are timing recommendations based on historical patterns rather than guaranteed future conditions.
- Chronological monitoring detected substantial temperature-distribution drift, which may reflect seasonal or temporal changes and requires investigation.
- The FastAPI and MLflow components are local deployment/MLOps simulations rather than production infrastructure.
- Real-world use would require broader representative data, independent validation, governance controls, security, continuous monitoring and human oversight.

---

## Responsible and Sustainable AI

A dedicated discussion is provided in:

```text
part3_machine_learning/bias_fairness_report.md
```

The project considers:

- Sampling and coverage bias
- Proxy-label risk
- Uneven performance across time and conditions
- Distribution shift
- Explainability
- Human oversight
- Governance
- Model complexity and computational-resource trade-offs

The models should be treated as decision-support tools rather than autonomous traffic-control systems.

---

## Reports

The main written outputs are:

```text
part1_data_analytics/part1_insights_report.md
part2_python/part2_report.md
part3_machine_learning/part3_report.md
part3_machine_learning/bias_fairness_report.md
```

These reports document the methodology, results, limitations and interpretation for the corresponding project stages.
