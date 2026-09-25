# Part 2 – Python Data Engineering and Traffic Analytics Report

## 1. Methodology

A Python-based data engineering workflow was developed to prepare the Metro Interstate Traffic Volume dataset for analytics and subsequent machine learning tasks.

The raw dataset contained 48,204 rows and 9 columns. The pipeline first validated that all required columns were present before performing any transformation. Categorical fields including holiday, weather type and weather description were standardised to improve consistency.

The `date_time` field was converted into a valid datetime format. Duplicate records were identified and 17 duplicate rows were removed. Data-quality checks were also applied to physically impossible or missing sensor values. Ten invalid or missing temperature observations and one invalid or missing rainfall observation were handled using monthly median imputation. After cleaning, the dataset contained 48,187 rows.

Logging was implemented throughout the workflow using Python's `logging` module. INFO messages record successful processing stages and dataset dimensions, WARNING messages identify modified, removed or imputed records, DEBUG messages record intermediate calculations, and ERROR messages capture failures or invalid user input.

## 2. Feature Engineering

The cleaned dataset was transformed into an ML-ready dataset containing 29 columns.

Time-based features included hour of day, day of week and a weekend indicator. Hour was also represented using sine and cosine cyclical encoding.

Weather information was enhanced through an adverse-weather indicator and one-hot encoding of weather categories.

Continuous variables including temperature, rainfall and cloud cover were standardised using `StandardScaler`.

A data-driven congestion category was created from traffic-volume percentile thresholds to classify traffic demand as low, medium or high.

## 3. Traffic Pattern Findings

Three Matplotlib visualisations were produced.

Average traffic volume showed a strong hourly pattern. The highest average traffic occurred at approximately 16:00, with around 5,664 vehicles, while the lowest average traffic occurred around 03:00, with approximately 371 vehicles.

Average weekday traffic was approximately 3,533 vehicles compared with 2,571 vehicles during weekends, a difference of approximately 963 vehicles.

The relationship between temperature and traffic volume was weak. The correlation was approximately 0.132, indicating only a weak positive linear relationship.

## 4. Mini Traffic Analytics Application

A command-line traffic analytics application was developed with three functions:

- Query traffic information for a specified date.
- Identify periods where traffic exceeds a user-defined threshold.
- Compare weekday and weekend traffic patterns.

The application validates user input and logs errors clearly instead of exposing an unhandled traceback.

## 5. Conclusion

The Part 2 workflow provides a reproducible foundation for Part 3 machine learning work. The raw traffic data is validated, cleaned, transformed, logged and converted into ML-ready features.

The analysis indicates that time-related variables are likely to be important predictors of traffic demand, while temperature alone has only a weak relationship with traffic volume.