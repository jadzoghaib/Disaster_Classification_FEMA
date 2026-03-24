"""
utils.py - Shared constants and helper functions
FEMA Disaster Forecasting System v3.2
"""

PROCESSED = 'data/processed/'
RAW       = 'data/raw/'

# Canonical incident type mapping (v3.2: Winter Storm is now its own category)
TYPE_MAP = {
    'Flood':               'Flood',         'Flooding':           'Flood',
    'Dam/Levee Break':     'Flood',
    'Hurricane':           'Hurricane',     'Typhoon':            'Hurricane',
    'Tropical Storm':      'Hurricane',     'Coastal Storm':      'Hurricane',
    'Severe Storm(s)':     'Severe Storm',  'Severe Storms':      'Severe Storm',
    'Severe Storm':        'Severe Storm',  'Thunderstorms':      'Severe Storm',
    'Thunderstorm Winds':  'Severe Storm',  'Thunderstorm':       'Severe Storm',
    'Tornado':             'Tornado',       'Tornadoes':          'Tornado',
    'Fire':                'Wildfire',      'Wildfire':           'Wildfire',
    'Forest Fires':        'Wildfire',
    'Earthquake':          'Earthquake',    'Tsunami':            'Earthquake',
    # Winter Storm is now a separate canonical category (v3.2)
    # Previously these were mapped to Other - separating them gives the models
    # a distinct signal for cold-weather events (geographic/seasonal pattern
    # completely different from the Other catchall)
    'Snow':                'Winter Storm',  'Winter Storm':       'Winter Storm',
    'Winter Storms':       'Winter Storm',  'Ice Storm':          'Winter Storm',
    'Freezing':            'Winter Storm',  'Snowstorm':          'Winter Storm',
    'Blizzard':            'Winter Storm',  'Cold Wave':          'Winter Storm',
    'Severe Ice Storm':    'Winter Storm',
    'Drought':             'Other',         'Mud/Landslide':      'Other',
    'Landslide':           'Other',         'Chemical':           'Other',
    'Biological':          'Other',         'Toxic Substances':   'Other',
    'Terrorist':           'Other',         'Human Cause':        'Other',
    'Volcanic Eruption':   'Other',         'Unknown':            'Other',
}

# 8 canonical types (v3.2 adds Winter Storm)
CANONICAL_TYPES = [
    'Flood', 'Hurricane', 'Severe Storm', 'Tornado',
    'Wildfire', 'Earthquake', 'Winter Storm', 'Other',
]

# Tier definitions anchored to 2019 real dollars
# Note: Tier 3 (Catastrophic) is retained in the label map for display
# purposes and for the probability model, but the tier CLASSIFIER (NB05)
# excludes it from training scope (DRF scope = Tiers 0-2 only).
DISASTER_BINS   = [0, 1_000_000, 50_000_000, 500_000_000, float('inf')]
DISASTER_LABELS = {0: 'Minor', 1: 'Moderate', 2: 'Major', 3: 'Catastrophic'}
TIER_NAMES      = ['Minor', 'Moderate', 'Major', 'Catastrophic']

# CPI-U annual averages (BLS series CUUR0000SA0, 1982-84=100)
CPI_BY_YEAR = {
    1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9,
    2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3,
    2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6,
    2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1,
    2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7,
    2023: 304.7, 2024: 314.2,
}
CPI_2019 = CPI_BY_YEAR[2019]
CPI_2024 = CPI_BY_YEAR[2024]

def get_cpi_factor_to_2024(year: int) -> float:
    return CPI_2024 / CPI_BY_YEAR.get(year, CPI_2024)

def get_cpi_factor_to_2019(year: int) -> float:
    return CPI_2019 / CPI_BY_YEAR.get(year, CPI_2019)

GDP_PER_CAPITA = {
    'TX': 72_000, 'FL': 67_000, 'CA': 92_000, 'LA': 58_000, 'OK': 59_000,
    'MO': 61_000, 'KY': 55_000, 'TN': 63_000, 'NY': 101_000, 'AL': 54_000,
    'VA': 79_000,
}

STATE_TO_CPC_REGION = {
    'TX': 'South',     'OK': 'South',     'AR': 'South',    'LA': 'South',
    'FL': 'Southeast', 'GA': 'Southeast', 'SC': 'Southeast','AL': 'Southeast',
    'CA': 'West',      'OR': 'West',      'WA': 'West',     'NV': 'West',
    'NY': 'Northeast', 'PA': 'Northeast', 'NJ': 'Northeast','CT': 'Northeast',
    'IL': 'Central',   'MO': 'Central',   'KY': 'Central',  'TN': 'Central',
}

COLOURS = {
    'navy':       '#1B3A5C',
    'blue':       '#2E75B6',
    'light_blue': '#D5E8F0',
    'orange':     '#C55A11',
    'green':      '#375623',
    'amber':      '#7F6000',
    'red':        '#7B1818',
    'gray':       '#404040',
}
TIER_COLOURS = {
    'Minor':        '#4CAF50',
    'Moderate':     '#FFC107',
    'Major':        '#FF5722',
    'Catastrophic': '#B71C1C',
}


def data_summary(df, label: str = 'DataFrame') -> None:
    import pandas as pd
    print(f'\n{"_"*55}')
    print(f'  {label}: {df.shape[0]:,} rows x {df.shape[1]} columns')
    null_total = int(df.isnull().sum().sum())
    print(f'  Nulls: {null_total:,}  |  Dtypes: {dict(df.dtypes.value_counts())}')
    print(f'{"_"*55}')


def get_season(month: int) -> str:
    if month in (12, 1, 2):  return 'Winter'
    if month in (3, 4, 5):   return 'Spring'
    if month in (6, 7, 8):   return 'Summer'
    return 'Fall'


def add_prior_disasters(df, state_col: str = 'stateAbbreviation',
                         date_col: str = 'incidentBeginDate',
                         window_years: int = 5):
    import pandas as pd
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    counts = []
    for _, row in df.iterrows():
        if pd.isnull(row[date_col]):
            counts.append(0)
            continue
        cutoff = row[date_col] - pd.DateOffset(years=window_years)
        mask = (
            (df[state_col] == row[state_col]) &
            (df[date_col] >= cutoff) &
            (df[date_col] < row[date_col])
        )
        counts.append(int(mask.sum()))
    df['prior_disasters_5yr'] = counts
    return df


def classification_metrics(y_true, y_pred, label: str = 'Model',
                            target_names=None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'label':       label,
        'Accuracy':    round(accuracy_score(y_true, y_pred), 4),
        'F1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'F1_macro':    round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
    }


def time_based_split(df, year_col: str, train_end: int, val_end: int):
    train = df[df[year_col] <= train_end].copy()
    val   = df[(df[year_col] > train_end) & (df[year_col] <= val_end)].copy()
    test  = df[df[year_col] > val_end].copy()
    return train, val, test


def fmt_dollars(v) -> str:
    import numpy as np
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    if v >= 1e9:  return f'${v/1e9:.2f}B'
    if v >= 1e6:  return f'${v/1e6:.1f}M'
    return f'${v:,.0f}'
