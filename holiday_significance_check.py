import pandas as pd
import numpy as np
from scipy.stats import poisson

# Load data
# This file is only for calculating holiday significance; it can be folded into the notebook later.
df = pd.read_csv('Crashes.csv')

if 'DATE_VAL' in df.columns:
    valid = df.copy()
    valid['CrashDate'] = pd.to_datetime(valid['DATE_VAL'], errors='coerce', utc=True).dt.tz_localize(None)
else:
    month_map = {
        'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
        'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
    }
    if 'DATE_VAL_MONTH_DESC' in df.columns:
        df['DATE_VAL_MONTH'] = df['DATE_VAL_MONTH_DESC'].str.upper().map(month_map)
    if 'DATE_VAL_DAY_OF_MONTH' in df.columns:
        day_col = 'DATE_VAL_DAY_OF_MONTH'
    else:
        day_col = 'DATE_VAL_DAY'
    df['DATE_VAL_MONTH'] = pd.to_numeric(df['DATE_VAL_MONTH'], errors='coerce')
    df[day_col] = pd.to_numeric(df[day_col], errors='coerce')
    df['DATE_VAL_YEAR'] = pd.to_numeric(df['DATE_VAL_YEAR'], errors='coerce')

    valid = df.dropna(subset=['DATE_VAL_YEAR', 'DATE_VAL_MONTH', day_col]).copy()
    valid['CrashDate'] = pd.to_datetime(
        valid['DATE_VAL_YEAR'].astype(int).astype(str) + '-' +
        valid['DATE_VAL_MONTH'].astype(int).astype(str).str.zfill(2) + '-' +
        valid[day_col].astype(int).astype(str).str.zfill(2),
        errors='coerce', utc=True
    ).dt.tz_localize(None)

valid = valid.dropna(subset=['CrashDate']).copy()
holiday_map = {
    "New Year's Day": (1, 1),
    "Independence Day": (7, 4),
    "Halloween": (10, 31),
    "Veterans Day": (11, 11),
    "Christmas": (12, 25),
}

rows = []
for holiday_name, (month, day) in holiday_map.items():
    for year in sorted(valid['DATE_VAL_YEAR'].dropna().unique().astype(int)):
        year_df = valid[valid['DATE_VAL_YEAR'] == year].copy()
        if year_df.empty:
            continue
        total_days = len(pd.date_range(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'), freq='D'))
        avg_daily = len(year_df) / total_days
        target_date = pd.Timestamp(f'{year}-{month:02d}-{day:02d}')
        window_start = target_date - pd.Timedelta(days=2)
        window_end = target_date + pd.Timedelta(days=2)
        window_df = year_df[(year_df['CrashDate'] >= window_start) & (year_df['CrashDate'] <= window_end)]
        observed = len(window_df)
        expected = avg_daily * 5
        rate_ratio = (observed / 5) / avg_daily if avg_daily else np.nan
        p_value = float(poisson.sf(observed - 1, expected)) if expected > 0 else 1.0
        rows.append({
            'Holiday': holiday_name,
            'Year': year,
            'ObservedCrashCount': observed,
            'ExpectedCrashCount': expected,
            'RateRatioVsYearAvg': rate_ratio,
            'p_value': p_value,
            'is_significant_05': p_value < 0.05,
            'is_significant_01': p_value < 0.01,
        })

res = pd.DataFrame(rows)
print('Summary by holiday:')
summary = (
    res.groupby('Holiday')
       .agg(
           Years=('Year', 'count'),
           AvgRateRatio=('RateRatioVsYearAvg', 'mean'),
           MaxRateRatio=('RateRatioVsYearAvg', 'max'),
           MinPValue=('p_value', 'min'),
           MedianPValue=('p_value', 'median'),
           SignificantYears05=('is_significant_05', 'sum'),
           SignificantYears01=('is_significant_01', 'sum')
       )
       .sort_values(['AvgRateRatio', 'MinPValue'], ascending=[False, True])
)
print(summary.to_string())
print('\nDetailed yearly results:')
print(res.sort_values(['Holiday', 'Year']).to_string(index=False))
