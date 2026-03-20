import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from config import (
    CUTTED_DATA_PATH,
    RESULTS_DIR,
    HORIZON,
    FREQ,
    RANDOM_SEED,
    CB_CONFIGS,
    CB_NUM_LAGS,
    CB_SEASON_PERIODS,
    CB_NUM_SEASONAL_LAGS,
    CB_NUM_HARMONICS,
    CB_ITERATIONS,
    CB_LEARNING_RATE,
    CB_DEPTH,
)

from data_utils import to_long, prepare_selected_long_data, make_train_test_split
from seasonality import get_series_features, compute_series_features, seasonality_groups, sample_series_by_group
from metrics import metrics
from baselines import run_baselines, evaluate_baselines, summarize_baseline_results
from catboost_model import catboost_forecast_direct, evaluate_catboost_predictions, summarize_results

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    #загрузка данных
    df_long_cut = pd.read_csv(CUTTED_DATA_PATH)
    df_long_cut['timestamp'] = pd.to_datetime(df_long_cut['timestamp'])

    #сплит
    df_sf, train_df, test_df, series_meta = make_train_test_split(df=df_long_cut, horizon=HORIZON)

    #baselines
    preds = run_baselines(train_df=train_df, horizon=HORIZON, freq=FREQ)
    eval_df_base, baseline_results_df = evaluate_baselines(
        test_df=test_df,
        preds=preds,
        train_df=train_df,
        series_meta=series_meta,
        metrics_func=metrics,
    )

    baseline_summary_mean, baseline_summary_median, baseline_by_group = summarize_baseline_results(baseline_results_df)

    baseline_results_df.to_csv(RESULTS_DIR / 'baseline_results.csv', index=False)
    baseline_summary_mean.to_csv(RESULTS_DIR / 'baseline_summary_mean.csv')
    baseline_summary_median.to_csv(RESULTS_DIR / 'baseline_summary_median.csv')
    baseline_by_group.to_csv(RESULTS_DIR / 'baseline_by_group.csv', index=False)

    #catboost
    cb_results_tables = []
    cb_eval_tables = []

    for model_name, seasonal_lags, calendar_features, fourier_features in CB_CONFIGS:
        print(model_name)

        eval_df_cb, _ = catboost_forecast_direct(
            train_df=train_df,
            test_df=test_df,
            horizon=HORIZON,
            model_name=model_name,
            freq=FREQ,
            seasonal_lags=seasonal_lags,
            calendar_features=calendar_features,
            fourier_features=fourier_features,
            num_lags=CB_NUM_LAGS,
            season_periods=CB_SEASON_PERIODS,
            num_seasonal_lags=CB_NUM_SEASONAL_LAGS,
            num_harmonics=CB_NUM_HARMONICS,
            iterations=CB_ITERATIONS,
            learning_rate=CB_LEARNING_RATE,
            depth=CB_DEPTH,
            random_seed=RANDOM_SEED,
        )
        cb_eval_tables.append(eval_df_cb[['unique_id', 'ds', model_name]])

        res_df_cb = evaluate_catboost_predictions(
            eval_df=eval_df_cb,
            train_df=train_df,
            series_meta=series_meta,
            model_name=model_name,
            metrics_func=metrics,
        )

        cb_results_tables.append(res_df_cb)

    catboost_eval_df = cb_eval_tables[0].copy()

    for part in cb_eval_tables[1:]:
        catboost_eval_df = catboost_eval_df.merge(part, on=['unique_id', 'ds'], how='outer')

    catboost_eval_df.to_csv(RESULTS_DIR / 'catboost_eval.csv', index=False)    

    cb_results_df = pd.concat(cb_results_tables, ignore_index=True)
    cb_summary_mean, cb_summary_median, cb_by_group = summarize_results(cb_results_df)

    cb_results_df.to_csv(RESULTS_DIR / 'catboost_results.csv', index=False)
    cb_summary_mean.to_csv(RESULTS_DIR / 'catboost_summary_mean.csv')
    cb_summary_median.to_csv(RESULTS_DIR / 'catboost_summary_median.csv')
    cb_by_group.to_csv(RESULTS_DIR / 'catboost_by_group.csv', index=False)

    #общее сравнение моделей
    all_results_df = pd.concat([baseline_results_df, cb_results_df], ignore_index=True)
    all_summary_mean, all_summary_median, all_by_group = summarize_results(all_results_df)

    all_results_df.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    all_summary_mean.to_csv(RESULTS_DIR / 'all_summary_mean.csv')
    all_summary_median.to_csv(RESULTS_DIR / 'all_summary_median.csv')
    all_by_group.to_csv(RESULTS_DIR / 'all_by_group.csv', index=False)

    print(all_summary_mean)


if __name__ == '__main__':
    main()