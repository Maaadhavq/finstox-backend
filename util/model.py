import requests
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import nsepython as nse 
import os
from datetime import date, timedelta, datetime
import ta
import shap
import lime
import lime.lime_tabular


# fetch historical stock data via yfinance
def fetch_data_nsepython(symbol, days_history=250):
    print(f"> fetching data for {symbol}...")
    
    import yfinance as yf
    
    yf_symbol = symbol if symbol.endswith('.NS') else symbol + '.NS'
    try:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=int(days_history * 1.5))
        
        df = yf.download(yf_symbol, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
        
        if df.empty:
             print(f"> no data for {yf_symbol}, trying without .NS suffix...")
             df = yf.download(symbol, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
        
        if df.empty:
            raise ValueError("No data returned from yfinance.")
            
        df = df.reset_index()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        rename_map = {
            'Date': 'date', 
            'Close': 'close',
            'Volume': 'volume',
            'High': 'high',
            'Low': 'low'
        }
        df.rename(columns=rename_map, inplace=True)
        df.columns = [str(c).lower() if str(c).lower() in rename_map.values() else c for c in df.columns]
        
    except Exception as e:
        print(f"> yfinance failed: {e}")
        return None

    try:
        required_cols = ['date', 'close', 'volume', 'high', 'low']
        extract_cols = [c for c in required_cols if c in df.columns]
        df_processed = df[extract_cols].copy()
        
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        df_processed['close'] = pd.to_numeric(df_processed['close'], errors='coerce')
        df_processed.dropna(subset=['close'], inplace=True)
        
        df_processed = df_processed.sort_values('date').reset_index(drop=True)

        # compute technical indicators
        df_processed['SMA_5'] = ta.trend.sma_indicator(df_processed['close'], window=5)
        df_processed['SMA_20'] = ta.trend.sma_indicator(df_processed['close'], window=20)
        df_processed['RSI_14'] = ta.momentum.rsi(df_processed['close'], window=14)
        df_processed['Volatility'] = df_processed['close'].rolling(window=14).std()
        
        if 'volume' not in df_processed.columns:
            df_processed['volume'] = 0
            
        df_processed.dropna(inplace=True)
        df_processed = df_processed.reset_index(drop=True)

        print(f"> processed {len(df_processed)} records")
        return df_processed

    except Exception as e:
        print(f"> error processing data: {e}")
        return None


# preprocess data into sequences for lstm
def preprocess_data(data, n_steps=60):
    print("> preprocessing data...")

    # close must be first (index 0) for target extraction
    feature_columns = ['close', 'volume', 'SMA_5', 'SMA_20', 'RSI_14', 'Volatility']

    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    features_data = data[feature_columns].values

    if len(features_data) <= n_steps:
        print(f"> not enough data ({len(features_data)} points) for n_steps={n_steps}")
        return None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(features_data)

    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i, :])
        y.append(scaled[i, 0])

    if not X or not y:
         print("> could not create sequences")
         return None, None, None, None, None

    X = np.array(X)
    y = np.array(y)
    print(f"> X: {X.shape}, y: {y.shape}")
    return X, y, scaler, scaled, feature_columns


# build lstm model
def build_model(input_shape):
    print(f"> building model, input_shape={input_shape}")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# predict next n days using sliding window
def predict_next_days(model, scaled_data, scaler, n_steps=60, days=7):
    print(f"> predicting next {days} days...")
    if len(scaled_data) < n_steps:
         print(f"> not enough data ({len(scaled_data)}) for prediction")
         return None

    input_seq = scaled_data[-n_steps:]
    predictions = []
    num_features = input_seq.shape[1]

    for _ in range(days):
        input_reshaped = input_seq.reshape(1, n_steps, num_features)
        pred_scaled = model.predict(input_reshaped, verbose=0)[0, 0]

        # copy last row, update close price (index 0), slide window
        new_row = np.copy(input_seq[-1])
        new_row[0] = pred_scaled
        predictions.append(pred_scaled)
        input_seq = np.append(input_seq[1:], [new_row], axis=0)

    if not predictions:
         return None

    # inverse transform needs full feature width; fill zeros for non-close columns
    predictions_array = np.zeros((len(predictions), num_features))
    predictions_array[:, 0] = predictions

    try:
        inversed = scaler.inverse_transform(predictions_array)
        return inversed[:, 0].reshape(-1, 1)
    except Exception as e:
        print(f"> inverse transform error: {e}")
        return None


# generate shap feature importance
def generate_shap_values(model, X_train, target_instance, feature_columns):
    print("> generating shap values...")
    try:
        background = X_train[-100:]
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(target_instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # sum across time steps for per-feature importance
        feature_importance = np.sum(shap_values[0], axis=0)
        base_value = float(np.mean(model.predict(background, verbose=0)))

        return {
            "features": feature_columns,
            "values": [float(v) for v in feature_importance],
            "base_value": base_value
        }
    except Exception as e:
        print(f"> shap error: {e}")
        return None


# generate lime explanation weights
def generate_lime_weights(model, X_train, target_instance, feature_columns):
    print("> generating lime weights...")
    try:
        n_steps = X_train.shape[1]
        num_features = X_train.shape[2]

        # lime needs 2d; average across time steps
        background_2d = np.mean(X_train[-100:], axis=1)
        target_2d = np.mean(target_instance, axis=1)[0]

        explainer = lime.lime_tabular.LimeTabularExplainer(
            background_2d,
            feature_names=feature_columns,
            class_names=['ClosePrice'],
            mode='regression'
        )

        # wrapper: broadcast 2d back to 3d for lstm
        def predict_fn_2d(X_2d):
            X_3d = np.repeat(X_2d[:, np.newaxis, :], n_steps, axis=1)
            return model.predict(X_3d, verbose=0).flatten()

        exp = explainer.explain_instance(
            target_2d,
            predict_fn_2d,
            num_features=len(feature_columns),
            num_samples=150
        )

        # map lime output back to clean feature names
        extracted_features = []
        extracted_weights = []
        for lime_str, weight in exp.as_list():
            matched_feature = "Unknown"
            for col in feature_columns:
                if col in lime_str:
                    matched_feature = col
                    break
            extracted_features.append(matched_feature)
            extracted_weights.append(float(weight))

        return {
            "features": extracted_features,
            "weights": extracted_weights,
            "fidelity_score": float(exp.score)
        }
    except Exception as e:
         print(f"> lime error: {e}")
         return None


# plotting disabled
def plot_predictions(preds, last_date, symbol):
    return "plot_disabled.png"



# main prediction pipeline
def get_stock_predictions(symbol: str, days_to_predict: int = 7, n_steps: int = 60, epochs: int = 15, days_history: int = 350):
    print(f"> generating prediction for {symbol}")
    symbol = symbol.strip().upper()

    df = fetch_data_nsepython(symbol, days_history=days_history)
    if df is None or df.empty:
        print(f"> could not fetch data for {symbol}")
        return None

    last_day = df['date'].max()

    X, y, scaler, scaled_data, feature_columns = preprocess_data(df, n_steps=n_steps)
    if X is None:
        print(f"> preprocessing failed for {symbol}")
        return None

    model = build_model((X.shape[1], X.shape[2]))

    print(f"> training model for {symbol}...")
    try:
        model.fit(X, y, epochs=epochs, batch_size=64, verbose=0)
        print(f"> training complete for {symbol}")
    except Exception as e:
        print(f"> training error for {symbol}: {e}")
        return None

    predictions_raw = predict_next_days(model, scaled_data, scaler, n_steps, days=days_to_predict)
    if predictions_raw is None:
        print(f"> prediction failed for {symbol}")
        return None

    filename = plot_predictions(predictions_raw, last_day, symbol)
    predictions_list = [round(float(p[0]), 2) for p in predictions_raw]
    print(f"> predictions for {symbol}: {predictions_list}")

    # xai explanations
    target_instance = scaled_data[-n_steps:].reshape(1, n_steps, len(feature_columns))
    shap_data = generate_shap_values(model, X, target_instance, feature_columns)
    lime_data = generate_lime_weights(model, X, target_instance, feature_columns)

    return {
        'predictions': predictions_list,
        'last_date': last_day,
        'filename': f"{filename}",
        'shap_values': shap_data,
        'lime_weights': lime_data
    }
    
if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., SBIN, RELIANCE, INFY): ").strip().upper()

    df = fetch_data_nsepython(symbol, days_history=750)

    if df is not None and not df.empty:
        n_steps = 60
        X, y, scaler, scaled_data, feature_columns = preprocess_data(df, n_steps=n_steps)

        if X is not None and y is not None and scaler is not None and scaled_data is not None:
            model = build_model((X.shape[1], X.shape[2]))

            print("> training model...")
            model.fit(X, y, epochs=90, batch_size=64, verbose=1)
            print("> training complete")

            predictions = predict_next_days(model, scaled_data, scaler, n_steps, days=7)

            if predictions is not None:
                print("> predicted prices for next 7 business days:")
                for i, val in enumerate(predictions, 1):
                    price = val[0] if isinstance(val, (list, np.ndarray)) and len(val) > 0 else val
                    print(f"  day {i}: {price:.2f}")

                last_day = df['date'].max()
                plot_predictions(predictions, last_day)
            else:
                print("> prediction failed")
        else:
            print("> preprocessing failed")
    else:
        print(f"> could not fetch data for {symbol}")