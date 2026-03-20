from pathlib import Path

#paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
EDA_LONG_PATH = DATA_DIR / 'eda_long.csv'
EDA_RAW_PATH = RESULTS_DIR / 'eda_raw.pkl'

CUTTED_DATA_PATH = DATA_DIR / 'cutted.csv'
SELECTED_DATA_PATH = DATA_DIR / 'selected.csv'
SELECTED_SERIES_PATH = DATA_DIR / 'selected_series.csv'

SERIES_FEATURES_PATH = RESULTS_DIR / 'series_features.csv'
GROUP_SUMMARY_PATH = RESULTS_DIR / 'group_summary.csv'
EDA_SERIES_IDS_PATH = RESULTS_DIR / 'eda_series_ids.csv'

#dataset
DATASET_NAME = 'electricity_hourly_dataset'
FREQ = 'h'


RANDOM_SEED = 42
N_EDA_SERIES = 20
GROUP_COL = 'acf_168'
GROUP_SIZE = 20

TRAIN_LENGTH = 12 * 7 * 24
HORIZON = 7 * 24
TOTAL_LENGTH = TRAIN_LENGTH + HORIZON

#catboost
CB_NUM_LAGS = 24
CB_SEASON_PERIODS = (24, 168)
CB_NUM_SEASONAL_LAGS = 3
CB_NUM_HARMONICS = 3

CB_ITERATIONS = 700
CB_LEARNING_RATE = 0.05
CB_DEPTH = 6

CB_CONFIGS = [
    ('CB_Lags', False, False, False),
    ('CB_Lags_Seasonal', True, False, False),
    ('CB_Lags_Calendar', False, True, False),
    ('CB_Lags_Fourier', False, False, True),
    ('CB_Lags_Seasonal_Calendar', True, True, False),
    ('CB_Lags_Seasonal_Fourier', True, False, True),
    ('CB_Lags_Seasonal_Calendar_Fourier', True, True, True),
]