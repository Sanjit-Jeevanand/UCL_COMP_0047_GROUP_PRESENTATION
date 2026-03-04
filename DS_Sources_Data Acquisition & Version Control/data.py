import pandas as pd
from fredapi import Fred
import datetime
import os
from scipy.stats import ttest_ind, levene

# 1. Initialize FRED API
API_KEY = '8cd3eb50e74f8a0dd16f9476130f00ec'
fred = Fred(api_key=API_KEY)

# Macroeconomic indicators from the task list
series_dict = {
    'USRECD': 'USRECD',       # NBER Recession Indicator (Prediction Target)
    'T10Y2Y': 'T10Y2Y',       # 10-Year minus 2-Year Treasury Yield Spread
    'UNRATE': 'UNRATE',       # Unemployment Rate
    'INDPRO': 'INDPRO',       # Industrial Production Index
    'CPIAUCSL': 'CPIAUCSL',   # Consumer Price Index (CPI)
    'TEDRATE': 'TEDRATE',     # TED Spread
    'BAA10Y': 'BAA10Y',       # BAA Corporate Bond Yield relative to Yield on 10-Year Treasury (Credit Spread)
    'FEDFUNDS': 'FEDFUNDS'    # Federal Funds Effective Rate (Core representative for H.15 rates)
}

# ==========================================
# Task 1: Data Acquisition
# ==========================================
data_frames = []
for name, series_id in series_dict.items():
    s = fred.get_series(series_id)
    s.name = name
    data_frames.append(s)

# Merge all indicators into a single raw DataFrame by date
raw_df = pd.concat(data_frames, axis=1)
raw_df.index.name = 'Date'

# ==========================================
# Task 2: Frequency Alignment & Label Engineering
# ==========================================
quarterly_df = pd.DataFrame()

# Alignment Logic:
# - Macro features: Use the quarterly mean
# - Recession labels: If any month in the quarter is in recession (value 1), the whole quarter is marked as recession (max)
for col in raw_df.columns:
    if col == 'USRECD':
        quarterly_df[col] = raw_df[col].resample('QS').max() # QS stands for Quarter Start
    else:
        quarterly_df[col] = raw_df[col].resample('QS').mean()

# Drop overly old years with excessive missing values (retain post-WWII core economic data)
quarterly_df = quarterly_df.dropna(subset=['USRECD', 'UNRATE'])

# Construct future prediction labels; shift(-1) brings the next quarter's recession status up to align with current quarter's features
quarterly_df['Target_1Q_ahead'] = quarterly_df['USRECD'].shift(-1)
quarterly_df['Target_2Q_ahead'] = quarterly_df['USRECD'].shift(-2)
quarterly_df['Target_3Q_ahead'] = quarterly_df['USRECD'].shift(-3)


# Create a dedicated output directory for data
output_dir = 'DS_SOURCES_PIPELINE'
os.makedirs(output_dir, exist_ok=True)

# Generate a timestamp with today's date to prevent silent data drift
today_str = datetime.datetime.today().strftime('%Y%m%d')
output_filename = os.path.join(output_dir, f'master_dataset_v{today_str}.csv')

# Export to CSV
quarterly_df.to_csv(output_filename)


# ==========================
# Task 3: Missing Data Strategy & Structural Break Tests
# ==========================

df = quarterly_df.reset_index()
df['Year'] = pd.to_datetime(df['Date']).dt.year

R1 = df[df['Year'] < 2000]
R2 = df[(df['Year'] >= 2000) & (df['Year'] < 2008)]
R3 = df[df['Year'] >= 2008]

variables = ['FEDFUNDS', 'T10Y2Y', 'BAA10Y']

# --- Part A: Structural Break Tests (on raw data, before any imputation) ---
print("=== Structural Break Tests (raw data) ===")
print(f"{'Variable':<10} {'Comparison':<18} {'t-test':<10} {'Levene':<10} {'Mean':<6} {'Vol':<6}")

for var in variables:
    r1, r2, r3 = R1[var].dropna(), R2[var].dropna(), R3[var].dropna()
    pre_08 = pd.concat([r1, r2])
    
    t_p = ttest_ind(pre_08, r3, equal_var=False)[1]
    l_p = levene(pre_08, r3)[1]
    print(f"{var:<10} {'Pre-08 vs Post-08':<18} {t_p:<10.4f} {l_p:<10.4f} {'Y' if t_p<0.05 else 'N':<6} {'Y' if l_p<0.05 else 'N':<6}")
    
    if len(r1) > 0 and len(r2) > 0:
        t_p2 = ttest_ind(r1, r2, equal_var=False)[1]
        l_p2 = levene(r1, r2)[1]
        print(f"{'':<10} {'R1 vs R2':<18} {t_p2:<10.4f} {l_p2:<10.4f} {'Y' if t_p2<0.05 else 'N':<6} {'Y' if l_p2<0.05 else 'N':<6}")
    
    if len(r1) > 0 and len(r3) > 0:
        t_p3 = ttest_ind(r1, r3, equal_var=False)[1]
        l_p3 = levene(r1, r3)[1]
        print(f"{'':<10} {'R1 vs R3':<18} {t_p3:<10.4f} {l_p3:<10.4f} {'Y' if t_p3<0.05 else 'N':<6} {'Y' if l_p3<0.05 else 'N':<6}")

print("\n=== Conclusion ===")
breaks_08 = sum(1 for var in variables if ttest_ind(pd.concat([R1[var].dropna(), R2[var].dropna()]), R3[var].dropna(), equal_var=False)[1] < 0.05)
print(f"Pre-08 vs Post-08: {breaks_08}/3 variables show significant mean change -> {'Primary break at 2008' if breaks_08 >= 2 else 'No clear break'}")

# --- Part B: Missing Indicator (for modeling only) ---
missing_cols = ['BAA10Y', 'T10Y2Y', 'FEDFUNDS']

print("\n=== Missing Data Handling (for modeling) ===")
for col in missing_cols:
    n_missing = quarterly_df[col].isnull().sum()
    pct = n_missing / len(quarterly_df) * 100
    median_val = quarterly_df[col].median()
    
    quarterly_df[f'{col}_missing'] = quarterly_df[col].isnull().astype(int)
    quarterly_df[col] = quarterly_df[col].fillna(median_val)
    
    print(f"{col}: {n_missing} ({pct:.1f}%) -> flag + median placeholder")

# ==========================
# Export
# ==========================
quarterly_df.to_csv(output_filename)
print(f"\nData saved to: {output_filename}")


# task 4 - stationarity stuff
# need to check if variables are stationary before putting them in a model
# used ADF and KPSS because they test opposite null hypotheses so together they give a clearer picture

from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import warnings
warnings.filterwarnings('ignore')

my_cols = ['T10Y2Y', 'UNRATE', 'INDPRO', 'CPIAUCSL', 'TEDRATE', 'BAA10Y', 'FEDFUNDS']
# note: the three above already got median-filled in task 3 so should not have NaN problems

def adf_kpss_check(series, varname):
    s = series.dropna()
    
    # ADF: H0 = has unit root, reject if p < 0.05 → stationary
    adf_res = adfuller(s, autolag='AIC')
    p1 = adf_res[1]
    
    # KPSS: H0 = stationary, reject if p < 0.05 → NOT stationary  
    # opposite of ADF which is why we need both
    kpss_res = kpss(s, regression='c', nlags='auto')
    p2 = kpss_res[1]

    stat1 = (p1 < 0.05)   # adf says stationary
    stat2 = (p2 >= 0.05)  # kpss says stationary

    if stat1 and stat2:
        result = 'STATIONARY'
    elif not stat1 and not stat2:
        result = 'NON-STATIONARY'
    else:
        result = 'INCONCLUSIVE'  # two tests disagree, need to use judgement

    print(f'\n{varname}')
    print(f'  adf p-val:  {p1:.4f}  ->  {"ok" if stat1 else "not stationary"}')
    print(f'  kpss p-val: {p2:.4f}  ->  {"ok" if stat2 else "not stationary"}')
    print(f'  verdict: {result}')
    return result

# run tests on raw series first
test_results = {}
print('\n--- Testing original series ---')
for col in my_cols:
    test_results[col] = adf_kpss_check(quarterly_df[col], col)

# decided on transformations based on the results + economic reasoning
# writing down why for each one so its clear
my_transforms = {
    'T10Y2Y'  : ('none',     'its a spread (10yr minus 2yr) so already difference-stationary by construction'),
    'UNRATE'  : ('diff',     'unemployment clearly trends, first difference removes the unit root'),
    'INDPRO'  : ('log_yoy',  'level grows over time so need yoy log change = industrial growth rate'),
    'CPIAUCSL': ('log_yoy',  'price level is I(1) with trend; yoy log diff gives us inflation rate which is interpretable'),
    'TEDRATE' : ('none',     'risk premium spread, similar reasoning to T10Y2Y'),
    'BAA10Y'  : ('none',     'credit spread mean-reverts around a long-run level'),
    'FEDFUNDS': ('diff',     'fed funds rate is very persistent; first diff = change in policy stance'),
}

df2 = quarterly_df.copy()

for col, (how, why) in my_transforms.items():
    if col not in df2.columns:
        continue
    if how == 'none':
        pass  # keep as is
    elif how == 'diff':
        df2[col + '_d1'] = quarterly_df[col].diff()
        df2.drop(columns=[col], inplace=True)
    elif how == 'log_yoy':
        # diff(4) on quarterly data = year-over-year change
        df2[col + '_lgyoy'] = np.log(quarterly_df[col]).diff(4)
        df2.drop(columns=[col], inplace=True)

# keep the _missing flag columns from task 3, they dont need transforming
flag_cols = [c for c in df2.columns if '_missing' in c]
print(f'\nkept {len(flag_cols)} missing indicator cols from task3: {flag_cols}')

# drop NaN rows caused by diff/log_yoy at beginning of series
# dont drop based on target or flag cols though
dont_check = ['Target_1Q_ahead', 'Target_2Q_ahead', 'Target_3Q_ahead'] + flag_cols
check_these = [c for c in df2.columns if c not in dont_check]
df2.dropna(subset=check_these, inplace=True)

print('\n--- transformed df ---')
print('shape:', df2.shape)
print(df2.head())

# re-test after transforming to make sure it worked
to_retest = [c for c in df2.columns
             if c not in ['USRECD', 'Target_1Q_ahead', 'Target_2Q_ahead', 'Target_3Q_ahead']
             and '_missing' not in c]
print('\n--- re-testing after transforms ---')
for c in to_retest:
    adf_kpss_check(df2[c], c)

# save it
save_path = os.path.join(output_dir, f'master_dataset_transformed_v{today_str}.csv')
df2.to_csv(save_path)
print(f'\nsaved transformed dataset to: {save_path}')

print('\n--- transform summary ---')
for col, (how, why) in my_transforms.items():
    print(f'  {col:12s}  {how:10s}  {why}')