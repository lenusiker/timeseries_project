import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sktime.datasets import load_forecastingdata

from config import (
    DATASET_NAME, FREQ, RANDOM_SEED, N_EDA_SERIES,
    GROUP_COL, GROUP_SIZE, TOTAL_LENGTH, HORIZON,
    DATA_DIR, RESULTS_DIR,
    CUTTED_DATA_PATH, SELECTED_DATA_PATH, SELECTED_SERIES_PATH,
    SERIES_FEATURES_PATH, GROUP_SUMMARY_PATH,
    EDA_SERIES_IDS_PATH, EDA_LONG_PATH, EDA_RAW_PATH)

from data_utils import to_long, prepare_selected_long_data, make_train_test_split
from seasonality import get_series_features, compute_series_features, seasonality_groups, sample_series_by_group


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)

    #1. загружаем сырой датасет
    df_raw, meta = load_forecastingdata(DATASET_NAME)

    #2. сэмплируем ряды
    eda_ids = df_raw['series_name'].sample(N_EDA_SERIES, random_state=RANDOM_SEED).tolist()
    df_eda_raw = df_raw[df_raw['series_name'].isin(eda_ids)].copy()

    #3. переделываем сырой датасет в long формат
    df_eda = df_raw[df_raw['series_name'].isin(eda_ids)].copy()
    df_eda_long = to_long(df_eda, freq=FREQ)

    #4.считаем сезонность на трейне
    series_features = compute_series_features(df=df_raw, total_length=TOTAL_LENGTH, horizon=HORIZON, feature_func=get_series_features) #достаем фичи для каждого ряда и считаем ACF

    #5. делим ряды по группам
    series_features = seasonality_groups(series_features, group_col=GROUP_COL)

    #6. саммари для каждой группы
    group_summary = series_features.groupby('seasonality_group')[['acf_24', 'acf_168']].agg(['count', 'mean', 'median', 'min', 'max'])

    #7. перем подвыборку из датасета
    selected_series = sample_series_by_group(series_features, group_size=GROUP_SIZE, random_state=RANDOM_SEED)

    #8. подготавливаем данные для моделей
    df_long_cut, df_long = prepare_selected_long_data(df=df_raw, selected_series=selected_series, freq=FREQ, total_length=TOTAL_LENGTH, to_long_func=to_long)

    #9. сохраняем все df
    pd.DataFrame({'series_name': eda_ids}).to_csv(EDA_SERIES_IDS_PATH, index=False)
    df_eda_long.to_csv(EDA_LONG_PATH, index=False)

    series_features.to_csv(SERIES_FEATURES_PATH, index=False)
    group_summary.to_csv(GROUP_SUMMARY_PATH)

    selected_series.to_csv(SELECTED_SERIES_PATH, index=False)
    df_long.to_csv(SELECTED_DATA_PATH, index=False)
    df_long_cut.to_csv(CUTTED_DATA_PATH, index=False)
    df_eda_raw.to_pickle(EDA_RAW_PATH)

if __name__ == '__main__':
    main()