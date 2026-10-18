# Charlotte Car Crash Analysis

The project uses public data from the City of Charlotte open data portal. The purpose is to analyze historical crash patterns, holiday car crash count changes, and factors associated with injury outcomes in Charlotte, North Carolina. The project merges available crash data with hourly weather data from Meteostat and uses Python for descriptive statistics, statistical hypothesis testing, and machine learning.

Data scope: This analysis work uses crash data from 2010–2024. The reason for excluding 2025 data is due to unusually low counts observed in September to December of said year; the completeness of those records has not been confirmed.


## Research questions

1. When do the most crashes occur (by hour, weekday, month, and year)?
2. Are there more crashes during holidays compared to the non-holiday days in the same month, and compared to the counts adjusted by weekdays?
3. For crashes, how well can crash characteristics (time, location, and weather) predict whether an injury occurs or not?
4. Does forcing precipitation (`prcp`) into a feature-selected logistic regression model improve its performance since prcp is intuitively related to crashes?


## Data sources

1. Crash data: City of Charlotte Open Data (https://data.charlottenc.gov/datasets/e48e2174b51040dca22e9aec4fd894f6_0/explore). Download and save it as `Crashes.csv`.
2. Hourly weather: Meteostat, external API and retrieved in Python using a weather station near Charlotte (station: 72314). Note： The ML notebook first converts hourly timestamps in weather data from UTC to `Crashes.csv`, and then merges crash data with weather data by local date and hour with a left join.



## Main Analysis Results
## 1. Historical crash analysis

The notebook: car_crash.ipynb studies the crash patterns. The data used in the study contains 482,929 records from 2010–2024. The output below identifies the categories with the highest recorded counts:

| Measure | Highest-count category | Recorded crashes |
| --- | --- | ---: |
| Hour of day | 5:00–5:59 p.m. | 41,422 |
| Day of week | Friday | 79,900 |
| Month | October | 46,484 |
| Year | 2023 | 44,472 |


### Holiday comparisons

This part verifies my assumption that more car crashes would occur during the holidays due to the increased commute volume. Six holidays are examined: New Year's Day, Independence Day, Halloween, Veterans Day, Thanksgiving, and Christmas. Each holiday is represented by a three-day window: the day before, the holiday itself, and the day after.

The analysis compares daily crashes from each holiday window with (a) other dates in the same month to rule out monthly effect and (b) other dates in that month with same weekdays to rule out weekday effect. For a window that crosses a month or year, the comparison is still based on the holiday's original month and year.

The saved results below present the average annual ratio to the weekday-matched baseline. 1.00 means the observed count was approximately equal to the estimated baseline; less than 1.00 means less crashes observed in the holiday, and greater than 1.00 means more crashes observed in the holiday.


| Holiday | Years analyzed | Average ratio vs. weekday baseline |
| --- | ---: | ---: |
| Halloween | 15 | 1.050 |
| Veterans Day | 15 | 0.987 |
| New Year's Day | 14 | 0.904 |
| Independence Day | 15 | 0.734 |
| Thanksgiving | 15 | 0.712 |
| Christmas | 15 | 0.615 |

The results indicate that out of 6 holidays, only Halloween has more crashes observed than the usual days.

### Poisson significance testing

The prior step only shows the descriptive statistics, but it begs the question: is there more crash data during the holidays that need statistical tests? The Poisson test serves to answer this question. The one-sided Poisson upper-tail test is applied to compare the observed crash count with the adjusted weekday-matched baseline. The alternative hypothesis is that the holiday-window count is higher than expected.

However, there are some limitations in this method. The p-values rely on a simplified Poisson model, treating estimated baselines as fixed values. Additionally, multiple tests are generated with different years. When many statistical tests are performed, some may produce p-values below 0.05 simply by chance. The last disadvantage is that this is for upper-tail tests which test the higher crashes during the holiday, not lower-tail tests for the less crashes.



## 2. Crash injury classification

The machine-learning notebook: car_crash_ML.ipynb studies possible factors associated with crash severity not the probability that a driver will experience a crash.

### Data preparation

1. Crash severity values 1,2,3,4 are grouped into one category: injury/fatality (`IS_INJURY = 1` as event); value of 5 is the other class (`IS_INJURY = 0`, property damage, no injury). Two recoards with a value of 6 and missing severity labels are excluded from model data.
2. Candidate predictors include location, weekday, hour, rush-hour/weekend indicators,  and weather-related variables (`temp`, `rhum`, `prcp`, `wspd`, `pres`, `cldc`), and one-hot encoded `CRASH_TYPE` variables.
3. Data are randomly assigned into training and testing data with 80% and 20%.
4. There is missing data in the candiate predictors, ~25% missing for `cldc`. The median is used for imputation. Scaling is also fitted on the training set for logistic regression.
5. Logistic regression uses RFE to select signifcant predictors.
6. Random forests are used as an alternative apprach to logistic regression. 


The testing dataset has **96,549 crashes**, including **25,415 labeled injury/fatality outcomes** and **71,134 non-injury outcomes**.

### Model results from logistic regression and random forest

The results shown below are from both modelling approaches with the 2010–2024 analysis period. Three trials are done for logistic regresison. The first one fixes the number of predictors equal to 12. All the varaibles selected in this round are crash types. The number of predictors is increased to 20 in the second round, and logitude, rush_hour and weekend and selected besides crash types. In the third round, precipitation was forced into the model.

Two trials were tested with random forest with differnt values for parameters.

| Model | Accuracy | Injury precision | Injury recall | Injury F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression + RFE (12 features) | 0.56 | 0.36 | 0.86 | 0.51 | 0.6982 |
| Logistic regression + RFE (20 features) | 0.57 | 0.36 | 0.84 | 0.51 | 0.7109 |
| Logistic regression + RFE (20 features, forced `prcp`) | 0.57 | 0.36 | 0.84 | 0.51 | 0.7103 |
| Random forest (100 trees) | 0.57 | 0.36 | 0.86 | 0.51 | 0.7180 |
| Random forest (200 trees) | **0.58** | **0.37** | 0.84 | 0.51 | **0.7207** |

Interpretation: The tested random forest configuration with 200 trees achieved the highest ROC-AUC in this comparison, although differences between models were modest. Class-weighted models recovered many injury cases but generated many false positives, as reflected by injury precision near 0.36–0.37. Accuracy alone is not a sufficient measure because non-injury crashes form the majority class.

### Precipitation experiment

Precipitation was intentionally included in one 20-feature logistic regression because rain was assumed to be relevant to crash injury outcomes. Its recorded ROC-AUC was 0.7103, versus 0.7109 for the standard 20-feature RFE model.

This comparison did not show a performance improvement from the forced-feature model. However, forcing `prcp` also changed the selected feature set (in this saved run, `prcp` replaced `LONGITUDE`). It does not demonstrate that rainfall is unrelated to injury outcomes.



## Steps to run

1. Download the official crash dataset and save it as Crashes.csv in the same folder as the notebooks.
2. Use a Python environment and install the requirement.txt to load in the packages:

   python -m pip install -r requirements.txt

3. Run notbook car_crash.ipynb to reproduce the descriptive and holiday analyses.
4. Run notebook car_crash_ML.ipynb to build the machine learning models and generate the results.

## Limitations and future work

1. Study Purpose：The crash dataset only includes the information after a crash occured, such as time, location etc. herefore, the analyze is only to study the post-crash information, not to predict how likley a crash will happen.
2. Data constraints：The analysis excludes data from 2025 and beyond due to unusually low counts observed in late 2025. Older records may also reflect changes in reporting practices.
3. Holiday crash analysis: The three-day windows and same month/weekday adjustments are approximations. Poisson p-values are exploratory and unadjusted with multiple tests.
4. Weather data analysis: Results show that some variables have a considerable missing percentage of observations, such as precipitation (prcp) with around 6% missing, and cloud cover (cldc) with almost 25% missing. Imputation using the median value may obscure real weather information.
5. Severity level：Due to rare events with a severity level of 1 (<1%), 2 (<1%), and 3 (5%), severity levels of 1-4 were grouped into one category representing injury/fatality in the model. Future development may work on how to model multiple categories with imbalanced data.

