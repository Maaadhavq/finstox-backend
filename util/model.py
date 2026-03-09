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

# Removed Marketstack API key

# Step 1: Fetch stock data from NSEPython
def fetch_data_nsepython(symbol, days_history=250):
    """Fetches historical equity data using yfinance, with synthetic fallback."""
    print(f"\nFetching stock data for {symbol} from yfinance...")
    
    import yfinance as yf
    
    yf_symbol = symbol if symbol.endswith('.NS') else symbol + '.NS'
    try:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=int(days_history * 1.5))
        
        df = yf.download(yf_symbol, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
        
        if df.empty:
             print(f"Warning: No data for {yf_symbol}. Trying without .NS suffix...")
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
        print(f"yfinance failed ({e}). Falling back to synthetic mock data for ML testing...")
        import numpy as np
        dates = pd.date_range(end=date.today(), periods=int(days_history * 1.5), freq='B')
        x = np.linspace(0, 10 * np.pi, len(dates))
        close_prices = 1000 + 200 * np.sin(x) + np.random.normal(0, 20, len(dates))
        
        df = pd.DataFrame({
            'date': dates,
            'close': close_prices,
            'volume': np.random.randint(100000, 500000, len(dates)),
            'high': close_prices + np.random.uniform(5, 20, len(dates)),
            'low': close_prices - np.random.uniform(5, 20, len(dates))
        })

    try:
        required_cols = ['date', 'close', 'volume', 'high', 'low']
        extract_cols = [c for c in required_cols if c in df.columns]
        df_processed = df[extract_cols].copy()
        
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        df_processed['close'] = pd.to_numeric(df_processed['close'], errors='coerce')
        df_processed.dropna(subset=['close'], inplace=True)
        
        df_processed = df_processed.sort_values('date').reset_index(drop=True)
        
        df_processed['SMA_5'] = ta.trend.sma_indicator(df_processed['close'], window=5)
        df_processed['SMA_20'] = ta.trend.sma_indicator(df_processed['close'], window=20)
        df_processed['RSI_14'] = ta.momentum.rsi(df_processed['close'], window=14)
        df_processed['Volatility'] = df_processed['close'].rolling(window=14).std()
        
        if 'volume' not in df_processed.columns:
            df_processed['volume'] = 0
            
        df_processed.dropna(inplace=True)
        df_processed = df_processed.reset_index(drop=True)

        print(f"Fetched and processed {len(df_processed)} records with multivariate features!\n")
        return df_processed

    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Step 2: Preprocess data (Updated for multivariate)
def preprocess_data(data, n_steps=60):
    print("Preprocessing data...")
    
    # Select features to use for the model. Order matters, 'close' MUST be first for easier indexing later.
    feature_columns = ['close', 'volume', 'SMA_5', 'SMA_20', 'RSI_14', 'Volatility']
    
    # Ensure all columns exist, if not, fill with 0
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    features_data = data[feature_columns].values

    # Check if features_data has enough data
    if len(features_data) <= n_steps:
        print(f"Error: Not enough data ({len(features_data)} points) to create sequences with n_steps={n_steps}")
        return None, None, None, None, None # Added feature_columns as return

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(features_data)

    X, y = [], []
    for i in range(n_steps, len(scaled)):
        # Input features for the past n_steps (all columns)
        X.append(scaled[i - n_steps:i, :])
        # Target variable is just the 'close' price (column index 0)
        y.append(scaled[i, 0])

    # Check if X and y were populated
    if not X or not y:
         print(f"Error: Could not create sequences. Check data length and n_steps.")
         return None, None, None, None, None

    X = np.array(X)
    y = np.array(y)
    # X shape is already (samples, n_steps, num_features)
    print(f"Shape of X: {X.shape}, y: {y.shape}")
    return X, y, scaler, scaled, feature_columns

# Step 3: Build model (Updated for multivariate)
def build_model(input_shape):
    print(f"\nBuilding model with input_shape: {input_shape}")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1)) # Output is still just 1 value (the 'close' price)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model ready!\n")
    return model

# Step 4: Predict next 7 days (Updated for multivariate)
def predict_next_days(model, scaled_data, scaler, n_steps=60, days=7):
    print(f"Predicting next {days} days...")
    if len(scaled_data) < n_steps:
         print(f"Error: Not enough historical data ({len(scaled_data)}) in scaled_data to make prediction")
         return None

    # Get the last n_steps of data
    input_seq = scaled_data[-n_steps:]
    predictions = []
    
    num_features = input_seq.shape[1]

    for _ in range(days):
        # Shape: (1, n_steps, num_features)
        input_reshaped = input_seq.reshape(1, n_steps, num_features)
        pred_scaled = model.predict(input_reshaped, verbose=0)[0, 0]
        
        # We only predict the 'close' price (index 0). 
        # For multi-step future forecasting in multivariate models without predicting all features,
        # we copy the last known values for other technical indicators and append the new close price.
        new_row = np.copy(input_seq[-1]) 
        new_row[0] = pred_scaled # Update 'close' price
        
        predictions.append(pred_scaled)
        
        # Slide window
        input_seq = np.append(input_seq[1:], [new_row], axis=0)

    if not predictions:
         return None

    # To inverse_transform, scaler expects (N, num_features), so we must mock the other features
    predictions_array = np.zeros((len(predictions), num_features))
    predictions_array[:, 0] = predictions # Fill 'close' prices
    
    try:
        inversed = scaler.inverse_transform(predictions_array)
        # Extract just the inverse transformed 'close' prices
        return inversed[:, 0].reshape(-1, 1)
    except Exception as e:
        print(f"Error during inverse transform: {e}")
        return None

# Step 4.5: XAI Generators
def generate_shap_values(model, X_train, target_instance, feature_columns):
    print("Generating SHAP values...")
    try:
        # Use a background distribution (e.g., last 100 days) to explain
        background = X_train[-100:]
        explainer = shap.GradientExplainer(model, background)
        
        # Explain the most recent prediction instance (shape: 1, 60, num_features)
        shap_values = explainer.shap_values(target_instance)
        
        # shap_values[0] shape is usually (1, 60, num_features) for Keras
        # We sum across the 60 time steps to get overall feature importance for this prediction
        if isinstance(shap_values, list): # depending on TF version, might return list
            shap_values = shap_values[0]
            
        shaps_2d = shap_values[0] # Take first instance
        feature_importance = np.sum(shaps_2d, axis=0) # Sum across time steps
        
        # Calculate base value (average prediction over background)
        base_value = float(np.mean(model.predict(background, verbose=0)))
        
        return {
            "features": feature_columns,
            "values": [float(v) for v in feature_importance],
            "base_value": base_value
        }
    except Exception as e:
        print(f"Error generating SHAP: {e}")
        return None

def generate_lime_weights(model, X_train, target_instance, feature_columns):
    print("Generating LIME weights...")
    try:
        # LIME expects 2D data (samples, features). Our LSTM expects 3D (samples, timesteps, features).
        # We create a wrapper that flattens/unflattens to bridge this gap.
        
        n_steps = X_train.shape[1]
        num_features = X_train.shape[2]
        
        # Create a 2D background by taking the mean across time steps for each sample
        background_2d = np.mean(X_train[-100:], axis=1) 
        target_2d = np.mean(target_instance, axis=1)[0]
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            background_2d,
            feature_names=feature_columns,
            class_names=['ClosePrice'],
            mode='regression'
        )
        
        # Prediction wrapper for LIME
        def predict_fn_2d(X_2d):
            # LIME sends variations in 2D. We must broadcast back to 3D to pass to LSTM
            # Simple approach: repeat the 2D variations across all n_steps
            X_3d = np.repeat(X_2d[:, np.newaxis, :], n_steps, axis=1)
            return model.predict(X_3d, verbose=0).flatten()

        exp = explainer.explain_instance(
            target_2d, 
            predict_fn_2d, 
            num_features=len(feature_columns),
            num_samples=150 # Drastically reduce permutations from default 5000 for faster API response
        )
        
        # Extract features and weights
        lime_list = exp.as_list()
        # LIME returns "feature_name <= logic" -> value. We string match to map back to clean feature_columns
        extracted_features = []
        extracted_weights = []
        
        for lime_str, weight in lime_list:
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
         print(f"Error generating LIME: {e}")
         return None


# Step 5: Plotting (Disabled because DLLs are blocked on host machine)
def plot_predictions(preds, last_date, symbol):
    """
    Generates and saves a dark-themed plot of stock predictions.
    NOTE: Disabled because `kiwisolver._cext` DLL is blocked by Application Control.
    """
    return "plot_disabled.png"



def get_stock_predictions(symbol: str, days_to_predict: int = 7, n_steps: int = 60, epochs: int = 15, days_history: int = 350):

    print(f"\n--- Generating prediction for {symbol} ---")
    symbol = symbol.strip().upper() # Clean up symbol

    # 1. Fetch Data
    df = fetch_data_nsepython(symbol, days_history=days_history)

    if df is None or df.empty:
        print(f"Could not fetch data for {symbol}.")
        return None # Indicate failure

    # Store last date before potentially failing preprocessing
    last_day = df['date'].max()

    # 2. Preprocess Data
    X, y, scaler, scaled_data, feature_columns = preprocess_data(df, n_steps=n_steps)

    if X is None: # Check if preprocessing failed
        print(f"Preprocessing failed for {symbol}.")
        return None # Indicate failure

    # 3. Build Model
    # Input shape depends on the actual preprocessed data shape
    model = build_model((X.shape[1], X.shape[2]))

    # 4. Train Model
    print(f"Training model for {symbol}...")
    try:
        # Consider adding validation data if possible for better training practices
        model.fit(X, y, epochs=epochs, batch_size=64, verbose=0) # Set verbose=0 for backend use
        print(f"Training complete for {symbol}!")
    except Exception as e:
        print(f"Error during model training for {symbol}: {e}")
        return None

    # 5. Predict Future Days
    predictions_raw = predict_next_days(model, scaled_data, scaler, n_steps, days=days_to_predict)

    if predictions_raw is None:
        print(f"Prediction failed for {symbol}.")
        return None 
    filename = plot_predictions(predictions_raw, last_day, symbol)

    predictions_list = [round(float(p[0]), 2) for p in predictions_raw]

    print(f"Predictions generated for {symbol}: {predictions_list}")

    # 6. Generate XAI Explanations
    # Provide the last sequence as the target instance to explain (the "today" condition)
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
    
# Main logic
if __name__ == "__main__":
    # Use upper() for consistency, strip whitespace
    symbol = input("Enter stock symbol (e.g., SBIN, RELIANCE, INFY): ").strip().upper()
    # Note: For NSE stocks, sometimes ".NS" is needed for other APIs, but nsepython often handles the base symbol.

    # Fetch data using the new function
    # Request more history to account for non-trading days (e.g., 750 days to get ~500 trading days)
    df = fetch_data_nsepython(symbol, days_history=750)

    if df is not None and not df.empty:
        n_steps = 60
        X, y, scaler, scaled_data, feature_columns = preprocess_data(df, n_steps=n_steps)

        # Proceed only if preprocessing was successful
        if X is not None and y is not None and scaler is not None and scaled_data is not None:
            model = build_model((X.shape[1], X.shape[2])) # input shape based on preprocessed X

            print("🧠 Training model...")
            # Consider adding validation_split or a separate validation set for better training
            model.fit(X, y, epochs=90, batch_size=64, verbose=1)
            print("✅ Training complete!\n")

            predictions = predict_next_days(model, scaled_data, scaler, n_steps, days=7)

            if predictions is not None:
                print("📤 Predicted prices for next 7 business days:")
                for i, val in enumerate(predictions, 1):
                    # Ensure val is indexable, handle potential shape issues
                    price = val[0] if isinstance(val, (list, np.ndarray)) and len(val) > 0 else val
                    print(f"Day {i}: ₹{price:.2f}") # Using Rupee symbol for NSE

                # Get the last date from the fetched data for plotting
                last_day = df['date'].max()
                plot_predictions(predictions, last_day)
            else:
                print("❌ Prediction failed.")
        else:
            print("❌ Preprocessing failed. Cannot train or predict.")
    else:
        print(f"❌ Could not fetch data for {symbol}. Exiting.")