import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

def get_series_features(values):
    """
    Вычисляет статистические и сезонные признаки одного временного ряда

    Параметры:
    -----------
    values: np.ndarray |list[float]: значения временного ряда

    Возвращает:
    --------
    pd.Series(
        - length: длина ряда
        - mean: среднее значение
        - std: стандартное отклонение
        - acf_24: автокорреляция на лаге 24
        - acf_168: автокорреляция на лаге 168)
    """
    values = np.asarray(values, dtype=float)
    acf_vals = acf(values, nlags=168, fft=True)

    return pd.Series({
        'length': len(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'acf_24': acf_vals[24],
        'acf_168': acf_vals[168]
    })


def compute_series_features(df, total_length, horizon, feature_func):
    """
    Считает признаки временных рядов только для train

    Параметры:
    ----------
    df: pd.DataFrame - исходный DF с колонками 'series_name' и 'series_value'
    total_length: int - общая длина окна (train + test)
    horizon: int - длина горизонта
    feature_func - функция, вычисляющая признаки одного ряда
    ------
    Возвращает:
    pd.DataFrame - таблица признаков по всем рядам
    """
    df_for_features = df.copy()
    df_for_features['series_value'] = df_for_features['series_value'].apply(lambda x: np.asarray(x, dtype=float)[-total_length:-horizon])

    series_features = pd.concat([df_for_features[['series_name']], df_for_features['series_value'].apply(feature_func)], axis=1)
    return series_features


def seasonality_groups(series_features, group_col='acf_168'):
    """
    Разбивает ряды на три группы по силе сезонности

    Параметры:
    ---------
    series_features: pd.DataFrame - таблица  с признаками рядов
    group_col: str, default='acf_168' - название колонки, по которой выполняется разбиение на группы

    Возвращает:
    ---------
    pd.DataFrame - таблица признаков с добавленной колонкой 'seasonality_group'
    """
    df = series_features.copy()
    df['seasonality_group'] = pd.qcut( df[group_col], q=3, labels=['weak', 'medium', 'strong'], duplicates='drop')
    return  df


def sample_series_by_group(series_features, group_size=20, random_state=42):
    """
    Выбирает случайную подвыборку рядов из каждой группы сезонности

    Параметры:
    --------
    series_features: pd.DataFrame - таблица с признаками и колонкой 'seasonality_group'
    group_size: int, default=20 - число рядов из каждой группы
    random_state: int, default=42

    Возвращает:
    ---------
    pd.DataFrame - подвыборка рядов из каждой группы сезонности
    """
    selected = (
        series_features.groupby('seasonality_group', group_keys=False)
        .apply(lambda x: x.sample(min(len(x), group_size), random_state=random_state))
        .sort_values(['seasonality_group', 'acf_168'])
        .reset_index(drop=True)
    )
    return selected