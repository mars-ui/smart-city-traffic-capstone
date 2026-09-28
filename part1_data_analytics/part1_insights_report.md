# Part 1 – Data Analytics Insights Report

## Overview

The Metro Interstate Traffic Volume dataset was analysed using SQLite, descriptive statistics, probability analysis and Power BI. The aim was to identify important traffic patterns and understand how factors such as time, temperature, weather and holidays relate to congestion.

The dataset covers 2012 to 2017, but data coverage is not complete for every year. This is important when comparing annual totals.

## Main Findings

Recorded yearly traffic volumes were:

| Year | Total Traffic Volume | % Change |
|---|---:|---:|
| 2012 | 8,208,767 | – |
| 2013 | 28,177,412 | +243.26% |
| 2014 | 15,731,289 | -44.17% |
| 2015 | 14,181,206 | -9.85% |
| 2016 | 29,494,821 | +107.99% |
| 2017 | 35,428,156 | +20.12% |

The large year-to-year changes are affected by incomplete coverage. For example, 2012 contains only the later months of the year, 2014 mainly covers January to August, and 2015 starts around June. In contrast, 2016 and 2017 have much more complete coverage. The totals therefore should not be treated as direct evidence of traffic growth or decline without considering the missing periods.

Traffic demand also varied strongly within the dataset. Average traffic volume was **3,259.82 vehicles**, with a median of **3,380** and a standard deviation of **1,986.86**. The wide range of **7,280 vehicles** reflects the difference between quiet overnight periods and busy commuting hours.

Temperature had only a weak linear relationship with traffic volume. The Pearson correlation was **0.1303**. This is also visible in the Power BI scatter plot, where traffic observations are widely spread across most temperatures instead of following a clear trend. Holiday observations showed a similar pattern: comparable Labor Day temperatures across 2015–2017 did not produce identical traffic volumes. Time of day, day type and travel behaviour are therefore likely to be more useful than temperature alone.

## Congestion and Weather

Congestion was defined as traffic volume above **5,500 vehicles**, and around **14.7%** of observations met this condition.

| Probability Measure | Result |
|---|---:|
| P(Congestion) | 0.1473 |
| P(Clear Weather) | 0.2778 |
| P(Congestion AND Clear Weather) | 0.0366 |
| P(Clear Weather \| Congestion) | 0.2483 |
| P(Temperature > 292K \| Congestion) | 0.2630 |

If congestion and clear weather were independent, the expected joint probability would be about **0.0409**. The observed value was **0.0366**, so the two events were not perfectly independent in this sample.

The odds ratio for congestion in clear weather compared with cloudy weather was approximately **0.735**. This means congestion had lower observed odds in clear conditions than cloudy conditions, but the result should be treated as an association rather than proof of causation.

## Power BI Dashboard

The dashboard supports the statistical findings visually. The hourly analysis shows a clear daily traffic cycle, while the weather chart shows noticeable differences between conditions.

**Clouds** had the highest average traffic volume at about **3,618.45 vehicles**, while **Squalls** had the lowest at about **2,061.75**, giving a difference of approximately **1,556.70 vehicles**.

The dashboard KPI values are approximately:

- **48.2K hours analysed**
- **3.26K average traffic volume**
- **8.06°C average temperature**

Slicers for hour, weather condition and traffic category allow users to examine specific operating conditions.

## Implications for Mobility Planning

The main practical finding is that traffic demand changes more strongly by operating period than by temperature alone. Hourly and weekday/weekend patterns should therefore receive more attention in traffic monitoring and forecasting.

The analysis also shows why data completeness must be checked before comparing yearly totals. Weather has some relationship with traffic conditions, but it is more useful when considered together with time, calendar and traffic-history information.

Overall, Part 1 provides the analytical foundation for the Python pipeline in Part 2 and the machine-learning work in Part 3.