# Part 3 – Machine Learning and AI: Intelligent Mobility Solution

## Overview

Part 3 builds on the traffic analysis and Python data pipeline developed in Parts 1 and 2. The cleaned Metro Interstate Traffic Volume dataset contained 48,187 observations and was used to develop the machine-learning, explainability, recommendation and MLOps components of the project.

No real accident dataset was provided. Following the capstone instructions, a proxy accident-risk label was therefore created only to demonstrate the classification workflow. The resulting classifier should not be interpreted as a model of actual accident occurrence.

## Task 1 – Supervised Machine Learning

The Part 3 preparation workflow was implemented in `prepare_ml_data.py`. The common feature set included hour, day of week, weekend status, cyclical hour and day encodings, holiday information, temperature, rainfall, snowfall, cloud cover and several encoded weather indicators.

Traffic volume was first divided into quartile-based congestion categories:

| Category | Definition |
|---|---|
| Low | Traffic volume <= Q1 |
| Medium | Q1 < traffic volume <= Q2 |
| High | Q2 < traffic volume <= Q3 |
| Severe | Traffic volume > Q3 |

The proxy high-risk label was then defined as High or Severe congestion occurring together with severe or low-visibility weather. This produced 39,834 non-high-risk records and 8,353 high-risk records.

`traffic_volume` and `congestion_category` were excluded from the classification predictor set to avoid direct target leakage.

### Classification

Two classifiers were compared: Logistic Regression and Random Forest.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9567 | 0.8095 | 0.9814 | 0.8872 | 0.9915 |
| Random Forest Classifier | 0.9799 | 0.9357 | 0.9491 | 0.9424 | 0.9964 |

The Random Forest produced the stronger overall result, with an F1-score of 0.9424 and ROC AUC of 0.9964. Logistic Regression achieved slightly higher recall, but at the cost of lower precision and F1.

These results show that the models reproduce the engineered proxy label well. They should not be interpreted as real accident-prediction performance. The proxy itself is partly constructed from weather conditions, and weather-derived variables are also used as predictors, so some conceptual overlap remains between the target definition and the model inputs.

### Regression

Traffic volume was predicted using Linear Regression and Random Forest Regression.

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 813.47 | 0.7260 |
| Random Forest Regressor | 268.67 | 0.9454 |

The Random Forest Regressor performed substantially better. Its MAE was about 269 vehicles compared with more than 813 for Linear Regression, while its R² of 0.9454 indicates that most of the variation in traffic volume was captured within the evaluation data.

The difference between the two models also suggests that traffic demand contains nonlinear relationships that are not handled as effectively by a simple linear model.

## Task 2 – Unsupervised Machine Learning

### K-means Clustering

K-means was used to group traffic operating conditions using hour, traffic volume, weather severity and weekend status. All clustering variables were standardised before fitting.

Rather than selecting the number of clusters arbitrarily, candidate values from k = 2 to k = 8 were compared using both inertia and silhouette score.

| k | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 135,118.40 | 0.3361 |
| 3 | 101,271.04 | 0.3860 |
| 4 | 79,766.43 | 0.3738 |
| 5 | 64,533.49 | 0.4098 |
| 6 | 49,778.57 | 0.4530 |
| 7 | 43,078.49 | 0.4670 |
| 8 | 36,537.27 | 0.4822 |

Among the tested values, k = 8 gave the highest silhouette score, 0.4822, and was therefore used for the final clustering analysis. This only identifies the best result within the tested range rather than claiming that eight clusters are universally optimal.

The eight clusters showed clear differences in traffic level, time of day, day type and weather severity.

| Cluster | Records | Avg Hour | Avg Traffic | Avg Weather Severity | Weekend Share |
|---|---:|---:|---:|---:|---:|
| 0 | 2,792 | 11.11 | 2,523.71 | 2.02 | 1.00 |
| 1 | 14,033 | 11.75 | 5,338.65 | 0.26 | 0.00 |
| 2 | 4,070 | 2.68 | 894.59 | 1.48 | 0.00 |
| 3 | 6,229 | 20.70 | 2,594.72 | 0.21 | 0.00 |
| 4 | 5,273 | 14.48 | 4,332.01 | 2.10 | 0.00 |
| 5 | 4,283 | 4.06 | 994.19 | 0.31 | 1.00 |
| 6 | 6,625 | 16.20 | 3,609.70 | 0.18 | 1.00 |
| 7 | 4,882 | 2.57 | 878.58 | 0.00 | 0.00 |

For example, Cluster 1 represents high weekday traffic under mostly mild weather, while Clusters 5 and 7 represent low early-hour or overnight traffic. Clusters 0 and 4 contain noticeably higher weather-severity values.

The evaluation results are stored in `results/kmeans_evaluation.csv` and visualised in `figures/kmeans_evaluation.png`. The final model and scaler are also saved so that the clustering can be reproduced.

### Association Rule Mining

Association rules were generated from time of day, weekday/weekend status, weather group and congestion category. The rules were ranked by lift.

| Antecedent | Consequent | Confidence | Lift |
|---|---|---:|---:|
| Overnight AND Weekend | Low congestion | 0.886 | 3.544 |
| Overnight AND Severe Weather | Low congestion | 0.854 | 3.418 |
| Overnight AND Mild Weather | Low congestion | 0.852 | 3.407 |
| Overnight | Low congestion | 0.849 | 3.395 |
| Midday AND Weekend | High congestion | 0.824 | 3.292 |

The strongest pattern is the association between overnight travel and low congestion. The weekend-midday rule is also useful because it shows that weekends should not automatically be treated as low-demand periods.

## Task 3 – Deep Learning and Explainability

A feed-forward neural network was developed to predict traffic volume. The network used the engineered traffic features as input, followed by a 64-unit ReLU layer, dropout, a 32-unit ReLU layer and a single continuous output node. Adam optimisation and mean squared error loss were used.

After 25 epochs, the neural network achieved:

| Metric | Result |
|---|---:|
| MAE | 355.93 |
| R² | 0.9250 |

The neural network performed well, but it did not outperform the Random Forest Regressor. For that reason, the Random Forest remained the preferred model for the deployment simulation.

### SHAP Explainability

SHAP was used to explain the Random Forest Regressor trained on the same traffic-volume prediction problem. TreeSHAP was selected because it provides an efficient way to calculate feature attributions for tree-based ensemble models.

The features with the highest mean absolute SHAP values were:

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

The strongest contributions came from time-related variables, particularly hour and cyclical time features. This agrees with the earlier analytics work, where traffic volume changed substantially by time of day and between weekdays and weekends.

SHAP values describe how the model makes predictions. They should not be interpreted as evidence that these variables cause changes in traffic.

## Task 4 – Advanced AI Technique: MLflow

MLflow was selected because experiment tracking and model versioning fit naturally with the MLOps requirements of the project. It also provides a practical way to compare models without relying on manually recorded results.

A local MLflow experiment called `smart_city_traffic_models` was created. Four supervised model versions were tracked:

| Version | Model | Task |
|---|---|---|
| classification_v1 | Logistic Regression | Classification |
| classification_v2 | Random Forest Classifier | Classification |
| regression_v1 | Linear Regression | Regression |
| regression_v2 | Random Forest Regressor | Regression |

The experiment records model parameters, evaluation metrics, model versions, run information, artifacts and project metadata.

MLflow uses a local SQLite database, `mlflow.db`, during execution. The raw database and local artifact store are excluded from GitHub because they contain environment-specific paths and potentially large artifacts. Instead, sanitized evidence of the completed runs is exported to `results/mlflow_runs.csv`, while `results/model_versions.csv` records the model-version comparison.

This approach keeps the public repository reproducible without exposing local filesystem details.

MLflow adds value by keeping model experiments traceable and making model comparisons easier. In a production environment, however, the current local setup would need to be replaced or extended with shared tracking infrastructure, authentication, centralised artifact storage, formal model approval, registry controls, backup and retention policies.

## Task 5 – Traffic Recommendation System

Because the dataset represents a single road corridor, the recommendation component focuses on **when to travel** rather than suggesting alternative routes.

`recommendation_system.py` identifies historically lower-traffic periods while considering day type and, where relevant, weather conditions. Candidate one-hour windows use start hours from 05:00 through 22:00, so the latest recommendation can run from 22:00 to 23:00 without selecting impractical overnight periods.

For a typical weekday, the system produced:

> For a weekday journey, consider travelling between 22:00 and 23:00. Historical traffic volume during this window averages approximately 2,126 vehicles.

When rain was included as a condition, the result was:

> For a weekday journey during rain weather, consider travelling between 22:00 and 23:00. Historical traffic volume during this window averages approximately 1,999 vehicles.

These recommendations are based on historical averages. They are intended as decision support and do not guarantee future traffic conditions.

## Task 6 – MLOps and Deployment Simulation

### Model Versioning and Experiment Tracking

The four supervised model versions were documented and compared using the same evaluation metrics used during modelling. MLflow records the experiment runs, parameters, metrics, versions and artifacts, while sanitized evidence is kept in the repository.

### FastAPI Deployment Simulation

A FastAPI mock-up was developed in `deployment_api.py`. It loads the trained Random Forest Regressor and exposes a prediction endpoint.

The input includes hour, day of week, holiday status, temperature, rainfall, snowfall, cloud cover and weather condition. The API recreates the feature engineering required by the trained model before returning the prediction.

A local self-test using a Tuesday at 10:00 under clear weather produced a predicted traffic volume of **4,422.83 vehicles**. This confirms that the trained model can be loaded and served through an API-style interface.

### Monitoring

A chronological monitoring simulation was used so that monitoring performance would not be evaluated on observations already used to train the monitoring model.

The data was divided into:

- earliest 70% for monitoring-model training: **33,730 records**
- next 10% as an unseen baseline window: **4,819 records**
- latest 20% as an unseen current window: **9,638 records**

A fresh Random Forest model with the same configuration was trained only on the earliest 70%.

The baseline MAE was **263.16**. An error-drift threshold was set at 125% of baseline MAE, giving a threshold of **328.95**. The current-window MAE was **321.24**, representing an increase of **22.07%**, but it remained below the configured error threshold.

Feature distribution drift was also checked using standardised mean difference, with an alert threshold of SMD > 0.50.

| Feature | SMD | Drift |
|---|---:|---|
| Hour | 0.004 | No |
| Temperature | 1.445 | Yes |
| Rainfall | 0.000 | No |
| Snowfall | 0.000 | No |
| Cloud coverage | 0.022 | No |
| Weather severity | 0.027 | No |

Temperature clearly exceeded the drift threshold. Because the windows are chronological, this may partly reflect seasonal or temporal changes rather than an immediate model failure, but it is still a condition that should be investigated.

The monitoring system therefore returned:

**SYSTEM STATUS: ALERT / Requires investigation**

The alert was caused by feature-distribution drift rather than prediction-error drift. In a production system, this type of alert would trigger checks of incoming data, seasonal changes and model performance before deciding whether retraining, recalibration or rollback was necessary.

## Task 7 – Responsible and Sustainable AI

A more detailed discussion is provided in `bias_fairness_report.md`.

The main risks identified in this project are incomplete coverage for some years, use of data from only one road corridor, uneven representation of weather conditions, possible differences in model performance across periods, proxy-label bias and future distribution shift.

The proxy accident-risk classifier is a particularly important limitation. It should not be used as if it predicts actual accidents, since no verified accident outcome was available.

Human oversight would therefore be required before any model output was used for safety-related or operational decisions. The system should support decision-making rather than automatically control traffic operations.

There is also a sustainability consideration in model selection. The Random Forest Regressor achieved better predictive performance than the neural network while using a simpler modelling and deployment workflow. More computationally intensive methods should therefore be adopted only when they provide a measurable improvement in operational value.

## Overall Findings

Several patterns were consistent across the different parts of the project.

Time-related features were repeatedly important. They appeared in the exploratory analysis, clustering results, recommendation system and SHAP explanations.

Among the traffic-volume models, the results were:

- Random Forest Regressor: R² = **0.9454**
- Neural Network: R² = **0.9250**
- Linear Regression: R² = **0.7260**

This supports the view that nonlinear relationships are important in traffic-demand modelling.

The unsupervised analysis also produced useful operational patterns. K-means selected k = 8 from the tested k = 2–8 range, while the association rules showed a strong relationship between overnight periods and low congestion.

The recommendation component then translated these historical patterns into practical travel-time suggestions.

The MLOps work showed a different but equally important point: strong model accuracy is not enough by itself. The monitoring simulation raised an alert because temperature distribution changed substantially, even though prediction error had not yet crossed its threshold. This is why deployment, monitoring, alerting, explainability and governance need to be considered together.

## Conclusion

Part 3 extends the earlier analytics and data-engineering work into an end-to-end intelligent mobility prototype. It combines supervised and unsupervised learning, neural-network modelling, SHAP explainability, MLflow experiment tracking, travel recommendations, FastAPI deployment and monitoring.

The strongest traffic-volume model was the Random Forest Regressor, with an MAE of **268.67** and R² of **0.9454**.

The project is still a demonstration rather than a production mobility system. Real-world use would require broader and more representative data, verified safety outcomes, independent validation, stronger governance controls, continuous monitoring and human oversight.