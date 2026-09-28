# Part 2 – Python Data Engineering and Traffic Analytics Report

## Methodology

Part 2 focused on building a reproducible Python workflow around the Metro Interstate Traffic Volume dataset. The raw file contained 48,204 rows and 9 columns.

The pipeline first checked that all required columns were present before any cleaning was carried out. Categorical fields such as holiday, weather type and weather description were standardised, and the `date_time` field was converted to a proper datetime format.

During cleaning, 17 duplicate rows were identified and removed. Data-quality checks also found 10 invalid or missing temperature values and one invalid or missing rainfall value. These were handled using monthly median values so that the replacement reflected the relevant period rather than using one overall value for the whole dataset. After cleaning, 48,187 rows remained.

Logging was included throughout the workflow so that each stage could be traced. INFO messages record normal processing steps, WARNING messages record changes such as removed or imputed values, DEBUG messages capture intermediate calculations, and ERROR messages are used when processing cannot continue or when invalid input is supplied.

## Feature Engineering

The cleaned data was then prepared for later analysis and machine-learning work.

New time-related features included hour of day, day of week and a weekend indicator. Hour was also represented using sine and cosine values so that the cyclical nature of time could be retained.

Weather information was expanded using an adverse-weather indicator and one-hot encoded weather categories. Continuous variables such as temperature, rainfall and cloud cover were also standardised.

A data-driven congestion category was created from the traffic-volume distribution so that traffic conditions could be grouped into low, medium and high demand levels.

## Traffic Patterns

Three Matplotlib visualisations were produced to examine the main traffic patterns.

The clearest result was the variation by time of day. Average traffic was highest at around 16:00, at approximately 5,664 vehicles, while the lowest average occurred around 03:00, at about 371 vehicles.

There was also a noticeable difference between weekdays and weekends. Average weekday traffic was approximately 3,533 vehicles compared with about 2,571 vehicles on weekends, a difference of roughly 963 vehicles.

Temperature showed only a weak relationship with traffic volume. The correlation was approximately 0.132, so temperature by itself does not explain much of the variation in traffic demand.

## Mini Traffic Analytics Application

A small command-line application was developed to make the processed data easier to query. It supports three main types of analysis:

- retrieving traffic information for a selected date,
- identifying periods above a user-defined traffic threshold,
- comparing weekday and weekend traffic patterns.

The application also checks user input and logs errors clearly instead of exposing an unhandled traceback.

## Conclusion

Part 2 produced a cleaned and reproducible traffic dataset that could be used directly in Part 3. The main finding from the Python analysis was that time-related behaviour is much more pronounced than the relationship between temperature and traffic volume.

The workflow also provides a traceable processing history through logging and a simple way for users to query the data through the command-line application. These outputs formed the practical data-engineering base for the machine-learning work in the next stage of the capstone.