# Part 2 – Python Data Engineering and Traffic Analytics

## Overview

This part of the Smart City Traffic capstone develops a reproducible Python workflow for cleaning, transforming, analysing and querying the Metro Interstate Traffic Volume dataset.

## Project Files

- `pipeline.py` – loads, validates and cleans the raw dataset
- `feature_engineering.py` – creates ML-ready time, weather and congestion features
- `visualizations.py` – generates traffic visualisations and interpretations
- `traffic_app.py` – command-line traffic analytics application
- `pipeline.log` – sample processing and application log
- `figures/` – generated Matplotlib figures
- `output/` – generated processed datasets; excluded from Git

## Data Pipeline

The pipeline performs:

1. CSV loading with exception handling
2. Schema validation
3. Categorical value standardisation
4. Date/time parsing
5. Duplicate removal
6. Detection of impossible temperature and rainfall values
7. Monthly median imputation
8. Saving of the cleaned dataset

Run:

```bash
python part2_python/pipeline.py