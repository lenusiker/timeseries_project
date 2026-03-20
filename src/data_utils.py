import pandas as pd
import numpy as np

def to_long(df, freq='h'):
    """
    Преобразует датасет в long формат

    Параметры:
    ----------
    df: pd.DataFrame - DataFrame с колонками
        - series_name
        - start_timestamp
        - series_value

    freq: str, default='h' - частота временного ряда

    Возвращает:
    ----------
    pd.DataFrame - таблица в long формате с колонками
        - timestamp
        - series_id
        - value
    """

    rows = []

    for _, row in df.iterrows():

        series_id = row['series_name']
        start = pd.to_datetime(row['start_timestamp'])
        values = np.asarray(row['series_value'], dtype=float)

        timestamps = pd.date_range(start=start, periods=len(values), freq=freq)

        tmp = pd.DataFrame({'timestamp': timestamps, 'series_id': series_id, 'value': values})
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)

def prepare_selected_long_data(df, selected_series, freq, total_length, to_long_func):
    """
    Готовит long таблицу только для выбранных рядов и обрезает последние total_length наблюдений

    Параметры:
    ----------
    df: pd.DataFrame - исходный DataFrame с временными рядами.
    selected_series : pd.DataFrame - таблица выбранных рядов с колонками
        - series_name
        - seasonality_group
        - acf_24
        - acf_168
    freq: str - частота временного ряда.
    total_length: int - длина окна, которое сохраняется для каждого ряда
    to_long_func: - функция преобразования данных в long формат

    Возвращает:
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        - df_long_cut: обрезанная long таблица с метаинформацией
        - df_long: полная long таблица для выбранных рядов
    """
    selected_ids = selected_series['series_name'].tolist()
    df_selected = df[df['series_name'].isin(selected_ids)].copy()
    df_long = to_long_func(df_selected, freq=freq)

    df_long_cut = df_long.groupby('series_id', group_keys=False).apply(lambda x: x.sort_values('timestamp').iloc[-total_length:]).reset_index(drop=True)
    seasonality_map = selected_series[['series_name', 'seasonality_group', 'acf_24', 'acf_168']].copy()
    df_long_cut = df_long_cut.merge(seasonality_map, left_on='series_id', right_on='series_name', how='left').drop(columns='series_name')

    return df_long_cut, df_long


def make_train_test_split(df, horizon):
    df_sf = df.rename(columns={'series_id': 'unique_id', 'timestamp': 'ds', 'value': 'y'}).copy()

    train_df = df_sf.groupby('unique_id', group_keys=False).apply(lambda x: x.sort_values('ds').iloc[:-horizon]).reset_index(drop=True)
    test_df = df_sf.groupby('unique_id', group_keys=False).apply(lambda x: x.sort_values('ds').iloc[-horizon:]).reset_index(drop=True)

    series_meta = df_sf[['unique_id', 'seasonality_group', 'acf_24', 'acf_168']].drop_duplicates('unique_id')

    return df_sf, train_df, test_df, series_meta