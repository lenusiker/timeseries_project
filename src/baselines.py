import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta

def get_baseline_models():
    return [
        Naive(alias='Naive'),
        SeasonalNaive(season_length=24, alias='SeasonalNaive_24'),
        SeasonalNaive(season_length=168, alias='SeasonalNaive_168'),
        AutoETS(season_length=24, alias='AutoETS_24'),
        AutoETS(season_length=168, alias='AutoETS_168'),
        AutoTheta(season_length=24, alias='AutoTheta_24'),
        AutoTheta(season_length=168, alias='AutoTheta_168')
    ]


def run_baselines(train_df, horizon, freq='h'):
    """
    Строит прогнозы baseline-моделей на заданном горизонте

    Параметры:
    --------
    train_df: pd.DataFrame - обучающая выборка в формате StatsForecast c колонками
        'unique_id', 'ds', 'y'
    horizon: int - горизонт прогнозирования
    freq: str, default='h' - частота временного ряда

    Возвращает:
    ---------
    pd.DataFrame - таблица прогнозов для всех моделей
    """
    models = get_baseline_models()
    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
    preds = sf.forecast(df=train_df[['unique_id', 'ds', 'y']], h=horizon)
    return preds


def evaluate_baselines(test_df, preds, train_df, series_meta, metrics_func):
    """
    Оценивает baseline-модели по метрикам на тестовой выборке

    Параметры:
    ----------
    test_df: pd.DataFrame - тестовая выборка с истинными значениями
    preds: pd.DataFrame - прогнозы моделей
    train_df: pd.DataFrame - обучающая выборка, необходимая для расчета масштабируемых метрик
    series_meta: pd.DataFrame - метаинформация по рядам
        'unique_id', 'seasonality_group', 'acf_24', 'acf_168'
    metrics_func - функция, которая принимает 'y_true', 'y_pred', 'train_values' и возвращает набор метрик: (mae, smape, mase24, mase168)

    Возвращает:
    ----------
    tuple[pd.DataFrame, pd.DataFrame]
        - eval_df: тестовая таблица, объединенная с прогнозами
        - results_df: таблица метрик по каждому ряду и каждой модели
    """
    eval_df = test_df.merge(preds, on=['unique_id', 'ds'], how='left')
    model_cols = [col for col in preds.columns if col not in ['unique_id', 'ds']]
    train_map = train_df.groupby('unique_id')['y'].apply(lambda x: np.asarray(x, dtype=float)).to_dict()

    results = []

    for uid, part in eval_df.groupby('unique_id'):
        part = part.sort_values('ds')
        y_true = part['y'].to_numpy()
        train_values = train_map[uid]
        meta_row = series_meta[series_meta['unique_id'] == uid].iloc[0]

        for model in model_cols:
            y_pred = part[model].to_numpy()
            mae_val, smape_val, mase24_val, mase168_val = metrics_func(y_true=y_true, y_pred=y_pred, train_values=train_values)

            results.append({
                'series_id': uid,
                'seasonality_group': meta_row['seasonality_group'],
                'acf_24': meta_row['acf_24'],
                'acf_168': meta_row['acf_168'],
                'model': model,
                'mae': mae_val,
                'smape': smape_val,
                'mase24': mase24_val,
                'mase168': mase168_val
            })

    results_df = pd.DataFrame(results)
    return eval_df, results_df

def summarize_baseline_results(results_df):
    """
     Строит агрегированные сводки по результатам baseline-моделей

    Параметры:
    ----------
    results_df: pd.DataFrame - таблица метрик по каждому ряду и каждой модели

    Возвращает:
    ----------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - baseline_summary_mean: средние значения метрик по моделям
        - baseline_summary_median: медианные значения метрик по моделям
        - baseline_by_group: средние значения метрик по группам сезонности и моделям
    """

    baseline_summary_mean = results_df.groupby('model')[['mae', 'smape', 'mase24', 'mase168']].mean().sort_values('mase168')
    baseline_summary_median = results_df.groupby('model')[['mae', 'smape', 'mase24', 'mase168']].median().sort_values('mase168')
    baseline_by_group = results_df.groupby(['seasonality_group', 'model'])[['mae', 'smape', 'mase24', 'mase168']].mean().reset_index()

    return baseline_summary_mean, baseline_summary_median, baseline_by_group