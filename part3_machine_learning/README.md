# Part 3 – Machine Learning and AI: Intelligent Mobility Solution

## Overview

This folder contains the Machine Learning and AI component of the Smart City Traffic Capstone.

Part 3 extends the cleaned and engineered traffic data from Parts 1 and 2 into an integrated intelligent mobility solution covering:

- Supervised classification and regression
- Unsupervised learning
- Deep learning
- SHAP explainability
- MLflow experiment tracking
- Traffic travel-time recommendations
- FastAPI deployment simulation
- Model monitoring and alerting
- Responsible and sustainable AI

The dataset is based on the Metro Interstate Traffic Volume dataset.

After Part 2 cleaning, 48,187 observations were used for Part 3 modelling.

---

## Important Note: Accident Dataset

No real accident dataset was provided for this capstone.

In accordance with the capstone instructions, a documented proxy accident-risk label was created.

A record is classified as `high_risk = 1` when:

- congestion is High or Severe based on traffic-volume quartiles, and
- severe or low-visibility weather is present.

This proxy is used only to demonstrate the classification workflow.

It must not be interpreted as prediction of actual road accidents.

---

## Project Structure

```text
part3_machine_learning/
│
├── prepare_ml_data.py
├── supervised_models.py
├── unsupervised_models.py
├── deep_learning_explainability.py
├── mlflow_tracking.py
├── recommendation_system.py
├── deployment_api.py
├── monitoring.py
│
├── README.md
├── part3_report.md
├── bias_fairness_report.md
├── part3.log
├── mlflow.db
│
├── data/
│   └── ml_ready_traffic.csv
│
├── models/
│   ├── logistic_classifier.joblib
│   ├── random_forest_classifier.joblib
│   ├── linear_regression.joblib
│   ├── random_forest_regressor.joblib
│   ├── classification_scaler.joblib
│   ├── regression_scaler.joblib
│   ├── neural_network_scaler.joblib
│   └── traffic_neural_network.keras
│
├── results/
│   ├── supervised_metrics.json
│   ├── kmeans_cluster_profiles.csv
│   ├── association_rules.csv
│   ├── unsupervised_summary.json
│   ├── shap_feature_importance.csv
│   ├── deep_learning_results.json
│   ├── model_versions.csv
│   ├── recommended_travel_windows.csv
│   ├── feature_drift.csv
│   └── monitoring_report.json
│
└── figures/
    ├── neural_network_training.png
    └── shap_feature_importance.png