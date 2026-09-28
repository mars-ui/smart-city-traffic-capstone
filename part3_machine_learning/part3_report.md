# Part 3 – Machine Learning and AI: Intelligent Mobility Solution

## 1. Overview

Part 3 extends the traffic analytics and reproducible Python pipeline developed in Parts 1 and 2 into an intelligent mobility solution.

The work covers supervised machine learning, unsupervised learning, deep learning, model explainability, advanced AI using MLflow, travel-time recommendations, MLOps deployment and monitoring, and responsible and sustainable AI.

The dataset used is the cleaned Metro Interstate Traffic Volume dataset produced in Part 2.

After data cleaning, 48,187 observations were available for modelling.

No real accident dataset was provided. Therefore, in accordance with the capstone instructions, a proxy accident-risk label was created solely to demonstrate the classification workflow. It must not be interpreted as a model of actual accident occurrence.

---

# Task 1 – Supervised Machine Learning

## 1.1 Feature Engineering and Proxy Risk Label

The Part 3 preparation workflow was implemented in:

`prepare_ml_data.py`

A common engineered feature set was created containing:

- Hour
- Day of week
- Weekend indicator
- Cyclical hour encoding
- Cyclical day-of-week encoding
- Holiday indicator
- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- Weather encoding
- Weather severity
- Severe-weather indicator
- Low-visibility indicator

Traffic congestion categories were calculated from traffic-volume quartiles:

| Category | Definition |
|---|---|
| Low | Traffic volume <= Q1 |
| Medium | Q1 < traffic volume <= Q2 |
| High | Q2 < traffic volume <= Q3 |
| Severe | Traffic volume > Q3 |

The proxy high-risk classification target was defined as:

**High/Severe congestion AND severe or low-visibility weather.**

The resulting proxy-label distribution was:

| Proxy class | Records |
|---|---:|
| Non-high-risk | 39,834 |
| High-risk | 8,353 |

Traffic volume and congestion category were excluded from the classification predictor set to avoid direct target leakage.

---

## 1.2 Classification Models

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest Classifier

The evaluation metrics were:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9567 | 0.8095 | 0.9814 | 0.8872 | 0.9915 |
| Random Forest Classifier | 0.9799 | 0.9357 | 0.9491 | 0.9424 | 0.9964 |

The Random Forest Classifier produced the strongest overall classification performance, with an F1-score of 0.9424 and ROC AUC of 0.9964.

The Logistic Regression model achieved slightly higher recall, but with lower precision and overall F1-score.

These results demonstrate strong ability to reproduce the engineered proxy target. They do not demonstrate actual accident-prediction ability because the target itself is an artificial proxy.

---

## 1.3 Regression Models

Traffic volume was predicted using:

1. Linear Regression
2. Random Forest Regressor

Results were:

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 813.47 | 0.7260 |
| Random Forest Regressor | 268.67 | 0.9454 |

The Random Forest Regressor substantially outperformed Linear Regression.

An R² of 0.9454 indicates that the Random Forest captured most of the variation in traffic demand within the evaluation data, while the MAE of approximately 269 vehicles was considerably lower than the Linear Regression result.

The improvement indicates that traffic demand contains nonlinear patterns that are better captured by an ensemble tree model.

---

# Task 2 – Unsupervised Machine Learning

## 2.1 K-Means Clustering

K-means clustering was applied to traffic operating conditions using:

- Hour
- Traffic volume
- Weather severity
- Weekend indicator

Four clusters were generated.

| Cluster | Records | Avg Hour | Avg Traffic | Avg Weather Severity | Weekend Share |
|---|---:|---:|---:|---:|---:|
| 0 | 15,331 | 14.87 | 4,485.55 | 0.00 | 0.00 |
| 1 | 8,623 | 2.51 | 824.65 | 0.63 | 0.00 |
| 2 | 13,548 | 11.30 | 2,558.80 | 0.57 | 1.00 |
| 3 | 10,685 | 13.71 | 4,354.31 | 1.60 | 0.01 |

The clusters can be interpreted as follows:

| Cluster | Interpretation |
|---|---|
| 0 | Higher weekday traffic under mostly mild weather |
| 1 | Low overnight weekday traffic with mixed weather |
| 2 | Moderate weekend traffic with mixed weather |
| 3 | Higher weekday traffic associated with more severe weather |

The clustering demonstrates that traffic conditions naturally separate according to time, day type, demand and weather characteristics.

---

## 2.2 Association Rule Mining

Association rules were generated using:

- Time of day
- Weekday/weekend
- Weather group
- Congestion category

The strongest rules were ranked by lift.

Examples include:

| Antecedent | Consequent | Confidence | Lift |
|---|---|---:|---:|
| Overnight AND Weekend | Low congestion | 0.886 | 3.544 |
| Overnight AND Severe Weather | Low congestion | 0.854 | 3.418 |
| Overnight AND Mild Weather | Low congestion | 0.852 | 3.407 |
| Overnight | Low congestion | 0.849 | 3.395 |
| Midday AND Weekend | High congestion | 0.824 | 3.292 |

The strongest finding is that overnight travel is substantially associated with low-congestion conditions.

The weekend midday rule also indicates that some weekend periods can still experience high traffic and should not automatically be treated as low-demand periods.

---

# Task 3 – Deep Learning and Explainability

## 3.1 Neural Network Demand Prediction

A feed-forward neural network was developed to predict traffic volume.

The architecture contained:

- Input layer using the engineered traffic features
- Dense layer with 64 ReLU units
- Dropout layer
- Dense layer with 32 ReLU units
- Single continuous output node

Adam optimisation and mean squared error loss were used.

The network completed 25 epochs.

Results were:

| Metric | Result |
|---|---:|
| MAE | 355.93 |
| R² | 0.9250 |

The neural network achieved strong predictive performance but did not outperform the Random Forest Regressor.

The Random Forest therefore remained the preferred model for the deployment simulation.

---

## 3.2 SHAP Explainability

SHAP was used to explain the Random Forest Regressor trained on the same traffic-volume prediction problem.

A tree-based explainability model was selected because TreeSHAP provides efficient feature-attribution values for ensemble tree models.

The highest-impact SHAP features were:

| Feature | Mean Absolute SHAP |
|---|---:|
| hour_cos | 1424.43 |
| hour | 422.59 |
| hour_sin | 195.06 |
| day_of_week | 149.94 |
| is_weekend | 142.27 |
| day_sin | 133.45 |
| temp | 114.43 |
| day_cos | 60.54 |
| clouds_all | 29.14 |
| weather_severity | 26.21 |

The result indicates that traffic-volume predictions are driven primarily by temporal patterns.

This is consistent with the earlier traffic analysis, which showed substantial variation in demand by hour and weekday/weekend status.

SHAP values explain model behaviour but should not be interpreted as evidence of causality.

---

# Task 4 – Advanced AI Technique: MLflow

## 4.1 Technique Selection

MLflow was selected as the advanced AI technique because it directly supports reproducibility, experiment tracking and model lifecycle management.

It also integrates naturally with the MLOps requirements of the capstone.

---

## 4.2 Implementation

A local MLflow experiment named:

`smart_city_traffic_models`

was created.

Four model versions were tracked:

| Version | Model | Task |
|---|---|---|
| classification_v1 | Logistic Regression | Classification |
| classification_v2 | Random Forest Classifier | Classification |
| regression_v1 | Linear Regression | Regression |
| regression_v2 | Random Forest Regressor | Regression |

MLflow records:

- Experiment runs
- Model type
- Model version
- Evaluation metrics
- Model artifacts
- Project metadata

A local SQLite MLflow tracking database is stored as:

`mlflow.db`

Model-version comparisons are also recorded in:

`results/model_versions.csv`

---

## 4.3 Value

MLflow improves reproducibility by preserving a traceable record of model experiments and their evaluation results.

It enables comparison between model versions and provides a foundation for controlled promotion of models from experimentation toward production.

---

## 4.4 Limitations

The MLflow implementation is a local capstone simulation.

A production implementation would normally require:

- Shared or managed tracking infrastructure
- Authentication and access control
- Centralised artifact storage
- Formal model approval workflows
- Production model registry and deployment controls
- Backup and retention policies

---

# Task 5 – Traffic Recommendation System

A travel-time recommendation system was implemented in:

`recommendation_system.py`

Because the dataset represents a single corridor, the system recommends **when to travel** rather than selecting an alternative route.

The system:

- Identifies historically lower-traffic periods
- Distinguishes weekday and weekend behaviour
- Considers weather conditions
- Produces plain-language travel recommendations

A minimum practical travel range of 05:00–22:00 was analysed to avoid recommending impractical overnight travel periods.

Example weekday recommendation:

> For a weekday journey, consider travelling between 22:00 and 23:00. Historical traffic volume during this window averages approximately 2,126 vehicles.

Weather-aware example:

> For a weekday journey during rain weather, consider travelling between 22:00 and 23:00. Historical traffic volume during this window averages approximately 1,999 vehicles.

The recommendations are based on historical traffic behaviour rather than guaranteed future conditions.

---

# Task 6 – MLOps and Deployment Simulation

## 6.1 Model Versioning

Four supervised model versions were documented and compared using their evaluation metrics.

The Random Forest Classifier and Random Forest Regressor were the strongest ensemble versions for their respective tasks.

---

## 6.2 Experiment Tracking

MLflow was used to track:

- Parameters
- Evaluation metrics
- Model versions
- Experiment runs
- Model artifacts

The experiment is stored locally in `mlflow.db`.

---

## 6.3 Deployment Simulation

A FastAPI deployment mock-up was implemented in:

`deployment_api.py`

The API loads the trained Random Forest Regressor and exposes a prediction endpoint.

The input schema includes:

- Hour
- Day of week
- Holiday indicator
- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- Weather condition

The API automatically recreates the feature engineering required by the trained model.

A local deployment self-test produced:

| Test Input | Result |
|---|---|
| Tuesday 10:00, clear weather | Predicted traffic volume = 4,422.83 |

The API returned a successful model prediction, demonstrating how the trained model could be exposed as a prediction service.

---

## 6.4 Model Monitoring

A monitoring simulation was implemented in:

`monitoring.py`

The chronologically ordered observations were divided into:

- Earlier 80%: reference window
- Latest 20%: simulated current monitoring window

Two monitoring approaches were used.

### Prediction Error Monitoring

The Random Forest Regressor baseline MAE was:

**268.67**

An alert threshold was set at 125% of baseline MAE:

**335.84**

The monitored current MAE was:

**132.01**

The prediction-error check therefore passed.

### Feature Distribution Monitoring

Feature distribution drift was evaluated using standardised mean difference.

The alert threshold was:

**SMD > 0.50**

Results were:

| Feature | SMD | Drift |
|---|---:|---|
| Hour | 0.001 | No |
| Temperature | 0.064 | No |
| Rainfall | 0.034 | No |
| Snowfall | 0.030 | No |
| Cloud coverage | 0.072 | No |
| Weather severity | 0.093 | No |

No feature exceeded the drift threshold.

---

## 6.5 Alerting

The monitoring system generates one of two statuses:

- PASS / Normal
- ALERT / Requires investigation

For the current monitoring simulation:

**SYSTEM STATUS: PASS / Normal**

In production, an ALERT would trigger investigation, validation of incoming data and model performance, and possible retraining or rollback.

---

# Task 7 – Responsible and Sustainable AI

A separate detailed report is provided in:

`bias_fairness_report.md`

The principal risks identified include:

- Incomplete historical coverage for some years
- Single-corridor geographic limitation
- Unequal representation of weather conditions
- Temporal performance differences
- Proxy-label bias
- Potential distribution shift
- Risk of misinterpreting the proxy classifier as an actual accident model

Human oversight should be retained for any operational or safety-related decision.

The model should be treated as decision support rather than an autonomous traffic-control mechanism.

The Random Forest Regressor also demonstrates an important sustainability consideration. It achieved better prediction accuracy than the neural network while using a simpler modelling and deployment workflow.

This suggests that higher computational complexity should only be adopted when it provides measurable operational value.

---

# 8. End-to-End Solution Integration

The final Part 3 solution integrates:

| Layer | Implementation |
|---|---|
| Data preparation | `prepare_ml_data.py` |
| Supervised ML | `supervised_models.py` |
| Unsupervised ML | `unsupervised_models.py` |
| Deep learning | `deep_learning_explainability.py` |
| Explainability | SHAP |
| Advanced AI | MLflow |
| Recommendation | `recommendation_system.py` |
| Deployment | `deployment_api.py` |
| Monitoring | `monitoring.py` |
| Responsible AI | `bias_fairness_report.md` |

Together, the components demonstrate how traffic data can progress from analytics and reproducible data engineering into modelling, explainability, decision support, deployment simulation and monitored AI operations.

---

# 9. Overall Findings

The project produced several consistent findings.

Traffic demand is strongly time-dependent. Time-of-day and weekday/weekend features were repeatedly important across exploratory analysis, clustering, recommendation generation and SHAP explainability.

The Random Forest models provided the strongest supervised-learning performance.

For traffic-volume prediction:

- Random Forest R² = 0.9454
- Neural Network R² = 0.9250
- Linear Regression R² = 0.7260

This suggests that nonlinear relationships are important in traffic demand.

Association-rule analysis also showed that overnight periods are strongly associated with low congestion.

The travel recommendation system translates these historical patterns into practical timing recommendations.

Finally, the MLOps components demonstrate that predictive performance alone is insufficient for operational AI. Model versioning, experiment tracking, deployment controls, monitoring, alerting, explainability and governance are all required to support a reliable intelligent mobility solution.

---

# 10. Conclusion

Part 3 successfully extends the earlier traffic analytics and Python pipeline into an integrated intelligent mobility prototype.

The solution demonstrates:

- Supervised classification and regression
- Traffic-condition clustering
- Association-rule mining
- Neural-network demand prediction
- SHAP explainability
- MLflow experiment tracking
- Travel-time recommendations
- FastAPI deployment simulation
- Model monitoring and alerting
- Responsible and sustainable AI considerations

The strongest traffic-volume model was the Random Forest Regressor with an MAE of 268.67 and R² of 0.9454.

However, the project remains a demonstration rather than a production mobility system.

Before real-world use, the models would require representative operational data, broader geographic coverage, verified safety outcomes, independent validation, formal governance, continuous monitoring and human oversight.