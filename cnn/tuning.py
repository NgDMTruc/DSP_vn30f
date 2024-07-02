from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from xgboost import callback
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
import pickle
import joblib 
import os

def choose_position(roi, trade_threshold = 0.01):
    pos =0
    # Predict position base on change in future
    if roi > trade_threshold:
        pos = 1
    elif roi < -trade_threshold:
        pos = -1
    else:
        pos = 0

    return pos

def backtest_position_ps(position, price, periods, percentage=0.01):
    #print(periods)
    # Shift positions to align with future price changes and handle NaN by filling with 0
    pos = pd.Series(position, index=pd.Series(price).index).shift(1).fillna(0)
    pos = pd.Series(pos).rolling(int(periods)).sum() #pos for 10 hour predict

    price_array = pd.Series(price).shift(1).fillna(0)

    pos_diff = pos.diff()
    fee = pos_diff*price_array*0.05*percentage

    # Calculate price changes over the given periods
    ch = pd.Series(price) - price_array

    # Calculate total PnL
    total_pnl = pos*ch - fee
    return total_pnl

def calculate_sharpe_ratio(pnl):
    pnl = np.diff(pnl)
    std = np.std(pnl) if np.std(pnl) != 0 else 0.001
    sharpe = np.mean(pnl)/std*np.sqrt(252)
    return sharpe

def sharpe_for_vn30f(y_pred, y_price, trade_threshold, fee_perc, periods):

    # Predict position base on change in future
    pos = [choose_position(roi, trade_threshold) for roi in y_pred]
    pos = np.array(pos)

    # Calculate PNL
    pnl = backtest_position_ps(pos, y_price, percentage=fee_perc, periods=periods)
    pnl = np.cumsum(pnl)

    # Standardalize PNL to date
    daily_pnl = [pnl.iloc[i] for i in range(0, len(pnl), 241)]
    daily_pnl = pd.Series(daily_pnl).fillna(0)

    # Calculate Sharpe
    sharpe = calculate_sharpe_ratio(daily_pnl)

    return pos, pnl, daily_pnl, sharpe

def calculate_hitrate(pos_predict, pos_true):
    if len(pos_predict) != len(pos_true):
        raise ValueError("Độ dài của hai mảng không khớp")

    # Tính số lượng dự đoán đúng (các phần tử tương ứng giống nhau)
    correct_predictions = np.sum(pos_predict == pos_true)

    # Tính tỷ lệ hit rate
    hit_rate_value = correct_predictions / len(pos_predict)

    return hit_rate_value

def scale_data(data):
    scaler = StandardScaler()
    data = np.where(np.isinf(data), np.nan, data)
    data = pd.DataFrame(data)
    data = data.fillna(0)
    scaler.fit(data)
    data=pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    return data

def split_data(data):
    new_part = np.array_split(data, 3)

    # Access each part individually
    hold_out = new_part[2]
    train_data = pd.concat([new_part[0], new_part[1]], axis=0)

    return train_data, hold_out

def split_optuna_data(data):
        train_data, _ = split_data(data)
        optuna_data = train_data.drop(['close', 'open','high','low','volume', 'Return'], axis=1)
        optuna_data = scale_data(optuna_data)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data, train_data['Return'], test_size=0.5, shuffle=False)

        return X_train, X_valid, y_train, y_valid

min_delta = 0.0001
patience = 30

class CustomEarlyStopping(callback.TrainingCallback):
    def __init__(self, min_delta, patience, verbose=False):
        super().__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.best_score = np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False
        metric_name = next(iter(evals_log['validation_0']))
        score = evals_log['validation_0'][metric_name][-1]
        if score < (self.best_score - self.min_delta):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nStopping. Best score: {self.best_score}")
                self.stopped_epoch = epoch
                return True
        return False

    def get_best_score(self):
        return self.best_score
    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

data = pd.read_csv('save_data.csv')
train_data, hold_out = split_data(data)

with open('top_10_list.pkl', 'rb') as f:
    selected_columns_cluster = pickle.load(f)



def objective_params(trial, X_train, X_valid, y_train, y_valid, y_close):
    # Define the hyperparameter search space
    params = {
        'epochs': trial.suggest_categorical('epochs', [10, 15, 20]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
    }
    trade_threshold  = 0.005

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i

    y_train_flat = y_train.values.flatten()
    y_valid_flat = y_valid.values.flatten()

    # Reshape data for LSTM
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_reshaped = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))


    with tf.device('GPU:0'):  # Ensure the model runs on GPU
        model = create_cnn_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
        model.fit(X_train_reshaped, y_train_flat, epochs=params['epochs'], batch_size=params['batch_size'],validation_data=(X_valid_reshaped, y_valid_flat), verbose=0)
        
        # Forecasting with the trained model
        forecast = model.predict(X_valid_reshaped).flatten().reshape(-1, 1)

        # Reshape the predictions back to their original shape
        y_train_pred = model.predict(X_train_reshaped).flatten().reshape(-1, 1)

    # Calculate Sharpe ratio and other metrics
    _, pos_is, _, sharpe_is = sharpe_for_vn30f(y_train_pred, y_close[:len(y_train_pred)], trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
    _, pos_os, _, sharpe_oos = sharpe_for_vn30f(forecast, y_close[len(y_train_pred):], trade_threshold=trade_threshold, fee_perc=0.01, periods=10)

    return sharpe_oos, abs((abs(sharpe_is / sharpe_oos))-1)

best_params_list = []
for idx, data_item in enumerate(selected_columns_cluster):
    train_cols, _ = split_data(data_item)
    optuna_data = scale_data(train_cols)

    X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                            train_data['Return'],
                                                            test_size=0.5,
                                                            shuffle=False)
    study = optuna.create_study(directions=['maximize', 'minimize'])
    
    y_train_flat = y_train.values.flatten()
    y_valid_flat = y_valid.values.flatten()

    # Reshape data for LSTM
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_reshaped = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))

    unique_trials = 10
    
    while unique_trials > len(set(str(t.params) for t in study.trials)):
        study.optimize(lambda trial: objective_params(trial, X_train, X_valid, y_train, y_valid, train_data['Close']), n_trials=1)
        study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(f'hypertuning{idx}.csv')
        joblib.dump(study, f'{unique_trials}hypertuningcluster{idx}.pkl')

    # Retrieve all trials
    trials = study.trials
    try: 
        completed_trials = [t for t in study.trials if t.values is not None]

        # Sort trials based on objective values
        completed_trials.sort(key=lambda trial: trial.values, reverse=True)

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)
    except: continue

    with tf.device('GPU:0'):  # Ensure the model runs on GPU
        model = create_cnn_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
        model.fit(X_train_reshaped, y_train_flat, epochs=params['epochs'], batch_size=params['batch_size'],validation_data=(X_valid_reshaped, y_valid_flat), verbose=0)
        
    model.save(f'best_in_cluster_{idx}.h5')

with open('best_params_list.pkl', 'wb') as f:
    pickle.dump(best_params_list, f)