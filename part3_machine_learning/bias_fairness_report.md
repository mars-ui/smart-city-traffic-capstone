# Bias, Fairness, Governance and Sustainability Report

## 1. Purpose

This intelligent mobility solution uses the Metro Interstate Traffic Volume dataset to analyse traffic patterns, predict traffic demand, identify traffic-condition clusters, generate travel-time recommendations, and demonstrate an end-to-end machine learning and MLOps workflow.

The solution does not use a real accident dataset. In accordance with the capstone instructions, a proxy accident-risk label was created to demonstrate a classification workflow. The proxy must therefore not be interpreted as an actual accident prediction.

---

## 2. Data Coverage and Sampling Limitations

The traffic dataset contains 48,187 cleaned observations after duplicate removal and data-quality treatment.

The historical data does not provide equally complete coverage for every year. In the earlier data analysis, some years were found to contain only partial-year observations, while other years had substantially more complete coverage. Therefore, raw yearly totals should not be interpreted as directly comparable measures of annual traffic growth without considering the available observation period.

The dataset also represents traffic on a single transport corridor. As a result:

- The findings may not generalise to other roads, cities or transport networks.
- Road geometry, local events, construction, incidents and nearby land-use patterns may differ elsewhere.
- Historical traffic behaviour may not represent future mobility patterns after major infrastructure, policy or behavioural changes.
- Certain weather conditions occur less frequently than common conditions, so model performance may be less reliable for rare weather scenarios.

The recommendation component therefore focuses on travel timing rather than route selection because no alternative route network data is available.

---

## 3. Proxy Accident-Risk Label

No real accident dataset was supplied for the capstone.

A proxy high-risk label was therefore constructed using two conditions:

1. Traffic congestion must fall into the High or Severe quartile-based congestion categories.
2. Severe or low-visibility weather conditions must also be present.

The proxy label produced:

- 39,834 non-high-risk records
- 8,353 high-risk records

This label is useful for demonstrating classification techniques, but it introduces important limitations.

High congestion combined with poor weather does not necessarily mean that an accident occurred. Likewise, accidents can occur during low traffic or clear weather.

The classification models therefore estimate the engineered proxy condition only. They must not be described or deployed as models that predict actual road accidents.

A production accident-risk system would require verified accident records, including accident time, location, severity and potentially road, vehicle and behavioural factors.

---

## 4. Bias and Fairness Risks

### 4.1 Temporal Bias

Traffic varies substantially by hour and day of week. The SHAP analysis showed that time-related variables were among the most influential features, especially:

- hour_cos
- hour
- hour_sin
- day_of_week
- is_weekend

This means model quality may differ across peak, off-peak, weekday and weekend periods.

Performance should therefore be monitored separately across major time windows instead of relying only on aggregate accuracy.

### 4.2 Weather Representation Bias

Common weather conditions have more observations than rare conditions.

The model may therefore learn common conditions more reliably than unusual events such as severe storms, snow or squalls.

Performance for rare weather conditions should be reviewed separately before operational use.

### 4.3 Proxy-Label Bias

The high-risk target is directly constructed from congestion and weather rules.

This means the label reflects assumptions selected for the capstone rather than independently observed accident outcomes.

A model trained on this label may reproduce those assumptions very accurately without learning true accident risk.

Therefore, high classification performance must not be interpreted as evidence that the model accurately predicts real-world accidents.

### 4.4 Geographic Bias

The dataset represents one corridor and does not contain a representative sample of multiple road environments.

Models should not be transferred directly to other locations without validation using local data.

---

## 5. Uneven Distribution of Errors

Prediction errors may not be distributed equally across all operating conditions.

Possible higher-error situations include:

- Rare weather conditions
- Major holidays
- Extreme congestion
- Unusual temperature or precipitation conditions
- Time periods with limited historical observations
- Future traffic conditions that differ from the historical training distribution

The monitoring component therefore evaluates both prediction error and feature-distribution drift.

In the current monitoring simulation:

- Baseline Random Forest MAE: 268.67
- Current monitored MAE: 132.01
- Alert threshold: 335.84
- No monitored feature exceeded the feature-drift threshold
- Overall system status: PASS / Normal

A production system should additionally calculate performance by time period, weather condition and congestion category.

---

## 6. Governance Requirements

The model should not automatically control traffic operations or make safety-critical decisions without human oversight.

Before real-world deployment, the following governance controls should be established:

1. **Data governance**
   - Define approved data sources.
   - Monitor data quality and lineage.
   - Document transformation and feature-engineering logic.

2. **Model validation**
   - Perform independent testing before deployment.
   - Validate performance across different traffic and weather conditions.
   - Define minimum acceptable performance thresholds.

3. **Model versioning**
   - Record each model version and its evaluation metrics.
   - Maintain traceability between training data, code and deployed models.

4. **Human oversight**
   - Require human review for decisions that could affect public safety or traffic operations.
   - Treat recommendations as decision support rather than mandatory instructions.

5. **Monitoring and alerting**
   - Monitor prediction error and feature drift.
   - Trigger investigation when predefined thresholds are exceeded.
   - Suspend or retrain models if performance becomes unreliable.

6. **Change management**
   - Revalidate models after material changes to data, road infrastructure or modelling logic.

MLflow was used in this project to support experiment tracking, model version comparison and reproducibility.

---

## 7. Explainability and Transparency

SHAP was used to explain the Random Forest traffic-volume model.

The analysis indicated that time-related features were dominant drivers of predictions. This is consistent with the observed strong daily traffic cycle.

Explainability is important because mobility-system operators should understand the main factors influencing model predictions before relying on them.

However, SHAP explains relationships learned by the model. It does not establish that the identified features cause traffic changes.

---

## 8. Sustainability

The project considered computational efficiency when selecting models and techniques.

The Random Forest Regressor achieved strong predictive performance with:

- MAE: 268.67
- R²: 0.9454

The neural network achieved:

- MAE: 355.93
- R²: 0.9250

Although the neural network provided a valid deep-learning demonstration, the Random Forest achieved better predictive performance for this dataset while requiring a simpler deployment workflow.

For a production solution, model complexity should therefore be justified by measurable value.

Sustainability practices should include:

- Avoiding unnecessary model retraining.
- Using smaller models when performance is comparable.
- Monitoring models to retrain only when meaningful drift occurs.
- Reusing validated models and feature pipelines.
- Selecting computing resources appropriate to the problem size.
- Tracking the operational benefit generated relative to computational cost.

---

## 9. Conclusion

The project demonstrates that machine learning can support traffic-demand prediction, traffic-pattern analysis and travel-time recommendations.

However, the system has important limitations arising from single-corridor data, uneven historical coverage and the use of a proxy accident-risk label.

The classification model should therefore be regarded as a demonstration of a machine learning workflow rather than a real accident-prediction system.

Real-world use would require representative operational data, verified safety outcomes, independent validation, ongoing monitoring, human oversight and formal governance controls.