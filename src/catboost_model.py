import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from catboost import CatBoostRegressor


def add_history_features(group, seasonal_lags=False, num_lags=24, season_periods=(24, 168), num_seasonal_lags=3):
    group = group.copy().sort_values('ds').reset_index(drop=True)
    group['ds'] = pd.to_datetime(group['ds'])

    #лаги
    for lag in range(1, num_lags + 1):
        group[f'lag_{lag}'] = group['y'].shift(lag)

    #сезонные лаги
    if seasonal_lags:
        for season_period in season_periods:
            for k in range(1, num_seasonal_lags + 1):
                shift_steps = k * season_period
                group[f'season_lag_{season_period}_{k}'] = group['y'].shift(shift_steps)

    return group


def add_future_horizon_features(df, step_start, step_end, freq='h', calendar_features=False, fourier_features=False, season_periods=(24, 168), num_harmonics=3):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    offset = to_offset(freq)

    local_step = 1
    for global_step in range(step_start, step_end + 1):
        future_ds = df['ds'] + global_step * offset

        if calendar_features:
            df[f'cal_hour_{local_step}'] = future_ds.dt.hour.astype(str)
            df[f'cal_dayofweek_{local_step}'] = future_ds.dt.dayofweek.astype(str)
            df[f'cal_month_{local_step}'] = future_ds.dt.month.astype(str)

        if fourier_features:
            t = (future_ds.astype('int64') // 3_600_000_000_000).to_numpy()

            for season_period in season_periods:
                for k in range(1, num_harmonics + 1):
                    angle = 2 * np.pi * k * t / season_period
                    df[f'fourier_sin_{season_period}_{k}_step_{local_step}'] = np.sin(angle)
                    df[f'fourier_cos_{season_period}_{k}_step_{local_step}'] = np.cos(angle)

        local_step += 1

    return df


def make_catboost_train_dataset_block(
        train_df, step_start, step_end, freq='h', 
        seasonal_lags=False, calendar_features=False, fourier_features=False, 
        num_lags=24, season_periods=(24, 168), num_seasonal_lags=3, num_harmonics=3
        ):
    
    groups = []

    for _, group in train_df.groupby('unique_id', sort=False):
        group = add_history_features(
            group=group,
            seasonal_lags=seasonal_lags,
            num_lags=num_lags,
            season_periods=season_periods,
            num_seasonal_lags=num_seasonal_lags,
        )

        group = add_future_horizon_features(
            df=group,
            step_start=step_start,
            step_end=step_end,
            freq=freq,
            calendar_features=calendar_features,
            fourier_features=fourier_features,
            season_periods=season_periods,
            num_harmonics=num_harmonics,
        )

        local_target_cols = []
        local_step = 1
        for global_step in range(step_start, step_end + 1):
            col_name = f'target_{local_step}'
            group[col_name] = group['y'].shift(-global_step)
            local_target_cols.append(col_name)
            local_step += 1

        groups.append(group)

    result_df = pd.concat(groups, ignore_index=True)

    metadata_cols = ['seasonality_group', 'acf_24', 'acf_168']
    feature_cols = [
        col for col in result_df.columns
        if col not in ['y', 'ds'] + local_target_cols + metadata_cols
    ]

    result_df = result_df.reset_index(drop=True)

    return result_df, feature_cols, local_target_cols


def make_catboost_inference_dataset_block(
        train_df, step_start, step_end, freq='h', 
        seasonal_lags=False, calendar_features=False, 
        fourier_features=False, num_lags=24, season_periods=(24, 168), num_seasonal_lags=3, num_harmonics=3
        ):
    groups = []

    for _, group in train_df.groupby('unique_id', sort=False):
        group = add_history_features(
            group=group,
            seasonal_lags=seasonal_lags,
            num_lags=num_lags,
            season_periods=season_periods,
            num_seasonal_lags=num_seasonal_lags,
        )

        group = add_future_horizon_features(
            df=group,
            step_start=step_start,
            step_end=step_end,
            freq=freq,
            calendar_features=calendar_features,
            fourier_features=fourier_features,
            season_periods=season_periods,
            num_harmonics=num_harmonics,
        )

        groups.append(group.tail(1))

    inference_df = pd.concat(groups, ignore_index=True)
    inference_df = inference_df.dropna().reset_index(drop=True)

    return inference_df


def fit_catboost_block(train_df, step_start, step_end, freq='h', seasonal_lags=False, 
                       calendar_features=False, fourier_features=False, num_lags=24, 
                       season_periods=(24, 168), num_seasonal_lags=3, num_harmonics=3, 
                       iterations=600, learning_rate=0.05, depth=6, random_seed=42):
    
    train_supervised, feature_cols, target_cols = make_catboost_train_dataset_block(
        train_df=train_df,
        step_start=step_start,
        step_end=step_end,
        freq=freq,
        seasonal_lags=seasonal_lags,
        calendar_features=calendar_features,
        fourier_features=fourier_features,
        num_lags=num_lags,
        season_periods=season_periods,
        num_seasonal_lags=num_seasonal_lags,
        num_harmonics=num_harmonics,
    )

    inference_df = make_catboost_inference_dataset_block(
        train_df=train_df,
        step_start=step_start,
        step_end=step_end,
        freq=freq,
        seasonal_lags=seasonal_lags,
        calendar_features=calendar_features,
        fourier_features=fourier_features,
        num_lags=num_lags,
        season_periods=season_periods,
        num_seasonal_lags=num_seasonal_lags,
        num_harmonics=num_harmonics,
    )

    X_train = train_supervised[feature_cols]
    y_train = train_supervised[target_cols]

    cat_features = ['unique_id']
    cat_features.extend([col for col in feature_cols if col.startswith('cal_')])

    model = CatBoostRegressor(
        loss_function='MultiRMSE',
        eval_metric='MultiRMSE',
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=5,
        random_seed=random_seed,
        task_type='GPU',
        devices='0',
        boosting_type='Plain',
        allow_writing_files=False,
        verbose=100,
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    return model, feature_cols, inference_df


def predict_catboost_block(model, feature_cols, inference_df, model_name, step_start, step_end, freq='h'):
    X_pred = inference_df[feature_cols]
    preds = model.predict(X_pred)

    n_steps = step_end - step_start + 1
    preds_df = pd.DataFrame(preds, columns=[f'target_{i}' for i in range(1, n_steps + 1)])

    rows = []
    inference_meta = inference_df[['unique_id', 'ds']].reset_index(drop=True)
    offset = to_offset(freq)

    for i, row in inference_meta.iterrows():
        uid = row['unique_id']
        last_train_ds = pd.to_datetime(row['ds'])

        future_ds = [last_train_ds + step * offset for step in range(step_start, step_end + 1)]
        tmp = pd.DataFrame({'unique_id': uid, 'ds': future_ds, model_name: preds_df.iloc[i].to_numpy()})
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def catboost_forecast_direct(train_df, test_df, horizon, model_name,
                            freq='h', seasonal_lags=False, calendar_features=False, 
                            fourier_features=False, num_lags=24, season_periods=(24, 168), 
                            num_seasonal_lags=3, num_harmonics=3, iterations=600, learning_rate=0.05, 
                            depth=6, random_seed=42, block_size=24):
    
    block_predictions = []
    fitted_models = {}

    for step_start in range(1, horizon + 1, block_size):
        step_end = min(step_start + block_size - 1, horizon)

        print(f'{model_name}: [{step_start}, {step_end}]')

        model, feature_cols, inference_df = fit_catboost_block(
            train_df=train_df,
            step_start=step_start,
            step_end=step_end,
            freq=freq,
            seasonal_lags=seasonal_lags,
            calendar_features=calendar_features,
            fourier_features=fourier_features,
            num_lags=num_lags,
            season_periods=season_periods,
            num_seasonal_lags=num_seasonal_lags,
            num_harmonics=num_harmonics,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
        )

        pred_block = predict_catboost_block(
            model=model,
            feature_cols=feature_cols,
            inference_df=inference_df,
            model_name=model_name,
            step_start=step_start,
            step_end=step_end,
            freq=freq,
        )

        block_predictions.append(pred_block)
        fitted_models[(step_start, step_end)] = model

    pred_long = pd.concat(block_predictions, ignore_index=True)
    pred_long = pred_long.sort_values(['unique_id', 'ds']).drop_duplicates(['unique_id', 'ds'], keep='last').reset_index(drop=True)

    eval_df = test_df.merge(pred_long, on=['unique_id', 'ds'], how='left')
    return eval_df, fitted_models


def evaluate_catboost_predictions(eval_df, train_df, series_meta, model_name, metrics_func):
    train_map = train_df.groupby('unique_id')['y'].apply(lambda x: np.asarray(x, dtype=float)).to_dict()
    results = []

    for uid, part in eval_df.groupby('unique_id'):
        part = part.sort_values('ds')

        y_true = part['y'].to_numpy()
        y_pred = part[model_name].to_numpy()
        train_values = train_map[uid]

        meta_row = series_meta[series_meta['unique_id'] == uid].iloc[0]

        mae_val, smape_val, mase24_val, mase168_val = metrics_func(y_true=y_true, y_pred=y_pred, train_values=train_values)

        results.append({
            'series_id': uid,
            'seasonality_group': meta_row['seasonality_group'],
            'acf_24': meta_row['acf_24'],
            'acf_168': meta_row['acf_168'],
            'model': model_name,
            'mae': mae_val,
            'smape': smape_val,
            'mase24': mase24_val,
            'mase168': mase168_val,
        })

    return pd.DataFrame(results)


def summarize_results(results_df):
    summary_mean = results_df.groupby('model')[['mae', 'smape', 'mase24', 'mase168']].mean().sort_values('mase168')
    summary_median = results_df.groupby('model')[['mae', 'smape', 'mase24', 'mase168']].median().sort_values('mase168')
    by_group = results_df.groupby(['seasonality_group', 'model'])[['mae', 'smape', 'mase24', 'mase168']].mean().reset_index()
    return summary_mean, summary_median, by_group