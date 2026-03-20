import numpy as np
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def metrics(y_true, y_pred, train_values):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    train_values = np.asarray(train_values, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))

    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(np.where(denom == 0, 0, 200 * np.abs(y_true - y_pred) / denom))

    scale24 = np.mean(np.abs(train_values[24:] - train_values[:-24]))
    if scale24 == 0:
        mase24 = np.nan
    else:
        mase24 = np.mean(np.abs(y_true - y_pred)) / scale24

    scale168 = np.mean(np.abs(train_values[168:] - train_values[:-168]))
    if scale168 == 0:
        mase168 = np.nan
    else:
        mase168 = np.mean(np.abs(y_true - y_pred)) / scale168

    return mae, smape, mase24, mase168
