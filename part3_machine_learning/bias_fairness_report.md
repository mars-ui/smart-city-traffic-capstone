# Bias, Fairness, Governance and Sustainability Report

## Purpose and Scope

The intelligent mobility solution developed in this capstone uses the Metro Interstate Traffic Volume dataset to predict traffic demand, identify traffic patterns, generate travel-time recommendations and demonstrate how a machine-learning solution could be monitored after deployment.

The dataset contains 48,187 cleaned observations after duplicate removal and data-quality treatment.

One important limitation is that no real accident dataset was provided. A proxy accident-risk label was therefore created only to demonstrate the required classification workflow. The classifier should not be interpreted as predicting whether an actual accident will occur.

## Data and Coverage Limitations

The historical dataset does not provide equal coverage across all years. Some years contain only part of the year, while others have much more complete coverage. This means that annual traffic totals cannot always be compared directly without considering how many months or observations are available.

The data also comes from a single traffic corridor. Traffic behaviour on this road may be different from other roads or cities because of differences in road design, surrounding land use, commuting behaviour, local events, construction and transport policy.

For the same reason, the travel recommendation component recommends **when to travel** rather than suggesting alternative routes. The dataset does not contain a wider road network that would support route selection.

Weather conditions are also unevenly represented. Common conditions appear much more frequently than unusual conditions such as severe storms, snow or squalls. A model may therefore perform well overall while being less reliable during rare operating conditions.

These limitations mean that the models should not be transferred directly to another location or used for long-term forecasting without further validation.

## Proxy Accident-Risk Label

The proxy high-risk label was created using two conditions: traffic had to fall within the High or Severe congestion categories, and severe or low-visibility weather also had to be present.

This produced:

| Proxy Class | Records |
|---|---:|
| Non-high-risk | 39,834 |
| High-risk | 8,353 |

This label is useful for demonstrating classification, but it is not equivalent to an observed accident outcome.

High congestion and difficult weather can occur without an accident, while accidents can also occur in light traffic and clear weather. The label therefore reflects the assumptions used to construct the capstone proxy rather than independently observed road-safety events.

There is also some conceptual overlap between the proxy definition and the classifier inputs because weather-related variables contribute to both. High classification performance may therefore partly reflect how well the model reproduces the engineered rule.

For real accident-risk modelling, verified accident records would be required, including accident time, location and severity, together with relevant road, vehicle and environmental information.

## Bias and Uneven Model Performance

Several forms of bias may affect the results.

Traffic demand changes substantially by hour and between weekdays and weekends. SHAP analysis also showed that time-related variables were among the strongest contributors to traffic-volume predictions. Model performance should therefore be checked separately across peak, off-peak, weekday and weekend periods instead of relying only on one overall accuracy figure.

Weather representation is another concern. Rare weather conditions have fewer training examples, so errors may be higher during unusual conditions even when average model performance is strong.

Geographic bias is also present because the data represents only one corridor. A model that performs well here may not behave in the same way on another road with different traffic patterns.

Other situations that may produce higher error include holidays, extreme congestion, unusual temperatures and future traffic conditions that differ from the historical training data.

For these reasons, fairness in this project is mainly concerned with whether model quality is reasonably consistent across different **operating conditions and time periods**, rather than demographic groups, since the dataset does not contain personal demographic information.

## Monitoring and Operational Risk

The monitoring simulation was designed chronologically so that the monitoring model was not evaluated on data it had already seen during training.

The earliest 70% of the observations were used for training, the next 10% formed the baseline period, and the most recent 20% represented simulated current traffic.

The baseline MAE was **263.16**. A prediction-error alert threshold was set at **328.95**, equal to 125% of the baseline MAE. The current MAE increased to **321.24**, which was a **22.07% increase**, but it remained below the error threshold.

Feature-distribution monitoring produced a different result. Temperature had a standardised mean difference of **1.445**, well above the drift threshold of 0.50.

The final monitoring status was therefore:

**ALERT / Requires investigation**

The alert does not automatically mean the model has failed. Because the comparison is chronological, the temperature shift may partly reflect seasonal changes. It does, however, show why incoming data needs to be monitored rather than assuming that historical model performance will remain stable.

A production system should investigate an alert before deciding whether recalibration, retraining or rollback is required.

## Governance and Transparency

The models in this project should be treated as decision-support tools rather than autonomous traffic-control systems.

Before any real-world deployment, there should be clear ownership of approved data sources, data-quality checks and feature-engineering logic. Each model version should also be traceable to its training data, source code and evaluation results.

Independent validation would be required before a model was promoted into operational use. Performance should be reviewed not only overall, but also across different traffic periods, weather conditions and congestion levels.

Human oversight is particularly important for safety-related decisions. An operator should be able to review recommendations and monitoring alerts rather than allowing the model to make an automatic safety-critical decision.

MLflow was used in this project to support experiment tracking and model-version comparison. SHAP was also used to provide greater transparency into the traffic-volume model. The explainability results showed that temporal features were major contributors to predictions, but these explanations describe model behaviour rather than proving causal relationships.

## Sustainability

Model selection also has a resource and sustainability dimension.

The Random Forest Regressor achieved:

- MAE: **268.67**
- R²: **0.9454**

The neural network achieved:

- MAE: **355.93**
- R²: **0.9250**

Although the neural network provided the required deep-learning implementation, it did not improve predictive performance over the Random Forest for this dataset.

Using a more computationally complex model is therefore not automatically better. Where a simpler model provides equal or better performance, it may also reduce training effort, deployment complexity and computing requirements.

For a production system, unnecessary retraining should be avoided. Monitoring can help determine when model performance or data distributions have changed enough to justify retraining rather than retraining on a fixed schedule without evidence of need.

## Conclusion

The intelligent mobility solution demonstrates useful applications of machine learning for traffic-demand prediction, traffic-pattern analysis and travel-time recommendations, but its results need to be interpreted within the limitations of the available data.

The most important risks are incomplete historical coverage, single-corridor data, uneven representation of operating conditions and the use of an engineered accident-risk proxy rather than real accident outcomes.

The monitoring simulation also showed that a model can require investigation even before prediction error crosses its alert threshold, because the characteristics of incoming data may change.

Real-world use would therefore require broader and more representative data, verified safety outcomes, independent validation, ongoing monitoring, clear governance and human oversight. Model complexity should also be justified by measurable operational benefit rather than adopted simply because a more advanced technique is available.