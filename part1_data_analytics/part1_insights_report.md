# Part 1 – Data Analytics Insights Report

## 1. Overview

Part 1 analysed the Metro Interstate Traffic Volume dataset using SQLite, descriptive statistics, probability analysis and Power BI.

The objective was to identify important traffic-demand patterns, understand relationships between traffic, temperature and weather conditions, and translate the findings into useful insights for smart-city mobility planning.

The dataset contains hourly traffic observations from 2012 to 2017. However, data coverage is not equally complete for every year. This is important when interpreting annual totals.

---

## 2. Annual Traffic Trends

The total recorded traffic volume by year was:

| Year | Total Traffic Volume | Change from Previous Year | % Change |
|---|---:|---:|---:|
| 2012 | 8,208,767 | – | – |
| 2013 | 28,177,412 | +19,968,645 | +243.26% |
| 2014 | 15,731,289 | -12,446,123 | -44.17% |
| 2015 | 14,181,206 | -1,550,083 | -9.85% |
| 2016 | 29,494,821 | +15,313,615 | +107.99% |
| 2017 | 35,428,156 | +5,933,335 | +20.12% |

At first sight, the totals appear to show large year-to-year increases and decreases. However, these figures must be interpreted carefully because the observation coverage differs substantially between years.

For example:

- 2012 contains observations only from approximately October to December.
- 2013 has close to full-year coverage.
- 2014 contains observations mainly from January to August.
- 2015 begins only around June.
- 2016 and 2017 have substantially more complete coverage.

Therefore, the large increase from 2012 to 2013 and the decline from 2013 to 2014 do not necessarily represent genuine changes in annual traffic demand. A significant portion of the difference is likely explained by incomplete data coverage.

For mobility planning, this demonstrates the importance of checking data completeness before using annual traffic totals for trend-based decisions.

---

## 3. Holiday and Temperature Analysis

Temperature patterns were examined for New Year's Day and Labor Day across 2015–2017.

Observed examples included:

| Year | Holiday | Temperature (K) | Traffic Volume |
|---|---|---:|---:|
| 2015 | Labor Day | 295.02 | 973 |
| 2016 | Labor Day | 293.17 | 1,064 |
| 2017 | Labor Day | 295.54 | 1,026 |
| 2016 | New Year's Day | 265.94 | 1,513 |
| 2017 | New Year's Day | 270.62 | 798 |

New Year's Day 2015 was not available because of the incomplete 2015 coverage.

The available observations show that temperatures differ considerably between summer and winter holidays, but traffic volume does not move consistently with temperature alone.

For example, Labor Day temperatures remained relatively similar across the observed years, while traffic volumes still varied. This suggests that calendar effects, holiday travel behaviour and time-of-day patterns may have greater operational importance than temperature by itself.

---

## 4. Descriptive Statistics

Traffic-volume descriptive statistics were:

| Statistic | Result |
|---|---:|
| Mean | 3,259.82 |
| Median | 3,380 |
| Standard deviation | 1,986.86 |
| Variance | 3,947,615.32 |
| Range | 7,280 |

The mean and median are relatively close, indicating that the overall traffic distribution is not dominated by a small number of extreme observations.

However, the standard deviation is large relative to the mean. This demonstrates substantial variation in traffic demand across different hours and operating conditions.

The large range also reflects the strong difference between low-demand overnight periods and high-volume commuting periods.

For smart-city mobility operations, average traffic alone is therefore insufficient. Traffic should be analysed by hour, weekday/weekend status and other operating conditions.

---

## 5. Temperature and Traffic Relationship

The Pearson correlation between temperature and traffic volume was:

**r = 0.1303**

This represents a **weak positive relationship**.

As temperature increases, traffic volume tends to increase slightly on average, but the relationship is too weak for temperature to be considered a strong standalone predictor of demand.

The Power BI scatter plot supports this finding. Traffic observations are widely distributed across the temperature range rather than forming a strong linear pattern.

The result also does not imply causation. Temperature may be associated with season, time of year and travel behaviour, while other variables such as hour of day, weekday/weekend status and holidays can have a stronger direct relationship with traffic patterns.

---

## 6. Congestion and Probability Analysis

Congestion was defined as:

**Traffic volume > 5,500 vehicles**

The analysis produced:

| Measure | Probability |
|---|---:|
| P(Congestion) | 0.1473 |
| P(Clear Weather) | 0.2778 |
| P(Congestion AND Clear Weather) | 0.0366 |
| P(Clear Weather \| Congestion) | 0.2483 |
| P(Temperature > 292K \| Congestion) | 0.2630 |

Approximately **14.7% of observations** met the congestion definition.

To test independence between clear weather and congestion:

- Observed joint probability = 0.0366
- P(Congestion) × P(Clear Weather) = 0.0409

Because these values are not equal, clear weather and congestion are not perfectly independent in the observed sample.

The odds ratio comparing congestion under clear versus cloudy weather was approximately:

**0.735**

An odds ratio below 1 indicates that congestion occurred with lower odds during clear conditions than during cloudy conditions in this historical dataset.

However, this should not be interpreted as proof that cloudy weather causes congestion. Time patterns, commuting behaviour and other variables may influence both conditions.

---

## 7. Power BI Dashboard Findings

The Power BI dashboard provides interactive views of traffic demand, weather and temperature.

### Daily Traffic

Daily traffic trends for 2015–2017 demonstrate substantial variation over time. The dashboard makes it possible to distinguish temporary fluctuations from more persistent demand patterns.

### Hourly Traffic

The hourly analysis shows a pronounced daily traffic cycle, confirming that time of day is one of the most important dimensions for mobility analysis.

This finding was later reinforced by the machine-learning work in Part 3, where hour-related features became some of the strongest traffic-demand predictors.

### Weather Impact

Average traffic volume by weather condition showed:

- Highest average: **Clouds – approximately 3,618.45**
- Lowest average: **Squalls – approximately 2,061.75**
- Difference: **approximately 1,556.70 vehicles**

Weather categories therefore show meaningful differences in observed average traffic, although some rare weather conditions may contain fewer observations and should be interpreted carefully.

### KPI Summary

The Power BI dashboard includes:

- Total hours analysed: approximately **48.2K**
- Average traffic volume: approximately **3.26K**
- Average temperature: approximately **8.06°C**

Interactive slicers allow analysis by:

- Hour
- Weather condition
- Traffic category

---

## 8. Implications for Smart-City Mobility

The analysis leads to several practical implications.

First, **time of day is a major driver of traffic demand**. Traffic-management strategies should therefore prioritise hourly demand patterns rather than relying only on daily or annual averages.

Second, **data completeness must be considered before comparing annual totals**. Missing months can create misleading apparent increases or decreases.

Third, **temperature alone has limited predictive value**. Mobility forecasting should combine time, calendar, weather and traffic-history variables rather than relying on a single environmental factor.

Fourth, congestion occurs in a meaningful but minority share of observations. Monitoring systems can therefore focus operational attention on identifiable high-demand periods while maintaining normal operations during lower-demand periods.

Finally, the dashboard demonstrates the value of integrating analytical results into an interactive decision-support environment. Mobility teams can use filters and KPI views to examine conditions relevant to specific operating periods rather than depending on static averages.

---

## 9. Conclusion

Part 1 demonstrates that traffic demand is highly variable and strongly influenced by temporal patterns.

The strongest conclusions are:

1. Annual traffic totals must be interpreted alongside data-coverage completeness.
2. Traffic volume shows high variability across operating periods.
3. Temperature has only a weak positive relationship with traffic demand.
4. Weather categories show differences in average traffic, but these relationships should not be interpreted as causal without further analysis.
5. Congestion represents approximately 14.7% of the observed records.
6. Interactive Power BI analysis provides a practical way for mobility teams to examine traffic, weather and congestion patterns.

These findings provide the analytical foundation for the reproducible Python pipeline in Part 2 and the machine-learning and intelligent mobility solution developed in Part 3.