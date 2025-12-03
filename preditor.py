import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
import joblib

MODEL_STORAGE_PATH = 'temp_models'
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_features(df):
    df_copy = df.copy()

    for i in range(1, 8):
        df_copy[f'close_lag_{i}'] = df_copy['close'].shift(i)

    df_copy['sma_7'] = df_copy['close'].rolling(window=7).mean()
    df_copy['sma_30'] = df_copy['close'].rolling(window=30).mean()
    df_copy['daily_volatility'] = df_copy['high'] - df_copy['low']
    df_copy['rsi_14'] = calculate_rsi(df_copy['close'], 14)

    return df_copy

def run_prediction(symbol_safe):
    try:
        file_path = f'Historico_Moedas/historico_{symbol_safe}.csv'

        if not os.path.exists(file_path):
            return {"error": f"Arquivo de historico para {symbol_safe} nao encontrado."}

        df_full = pd.read_csv(file_path)
        current_price = float(df_full['close'].iloc[-1])

        model_path = os.path.join(MODEL_STORAGE_PATH, f'model_xgb_{symbol_safe}.pkl')
        regressor_path = os.path.join(MODEL_STORAGE_PATH, f'regressor_rf_{symbol_safe}.pkl')

        # Treina XGBClassifier se não existir
        if os.path.exists(model_path):
            classifier = joblib.load(model_path)
        else:
            df_features = create_features(df_full.copy())
            df_features['target'] = np.where(df_features['close'].shift(-1) > df_features['close'], 1, 0)
            df_features.dropna(inplace=True)

            features = [c for c in df_features.columns if 'lag' in c or 'sma' in c or 'volatility' in c or 'rsi' in c]
            X_train = df_features[features]
            y_train = df_features['target']

            classifier = XGBClassifier(
                n_estimators=50,
                eval_metric='logloss',
                random_state=42
            )
            classifier.fit(X_train, y_train)
            joblib.dump(classifier, model_path)

        # Treina RandomForest se não existir
        if os.path.exists(regressor_path):
            regressor = joblib.load(regressor_path)
        else:
            df_features = create_features(df_full.copy())
            df_features['target_price'] = df_features['close'].shift(-1)
            df_features.dropna(inplace=True)

            features = [c for c in df_features.columns if 'lag' in c or 'sma' in c or 'volatility' in c or 'rsi' in c]
            X_train = df_features[features]
            y_train = df_features['target_price']

            regressor = RandomForestRegressor(n_estimators=30, random_state=42)
            regressor.fit(X_train, y_train)
            joblib.dump(regressor, regressor_path)

        df_features = create_features(df_full)
        features = [c for c in df_features.columns if 'lag' in c or 'sma' in c or 'volatility' in c or 'rsi' in c]

        X_pred = df_features[features].iloc[-1:].copy()
        X_pred.fillna(method='ffill', inplace=True)
        X_pred.fillna(method='bfill', inplace=True)

        if X_pred.isnull().values.any():
            return {"error": "Dados insuficientes para gerar features para previsao."}

        prediction_code = classifier.predict(X_pred)[0]
        prediction_proba = classifier.predict_proba(X_pred)[0]
        direction = "ALTA" if prediction_code == 1 else "BAIXA"
        confidence = prediction_proba[prediction_code] * 100

        predicted_price = regressor.predict(X_pred)[0]

        return {
            "symbol": symbol_safe.replace("_", "/"),
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "prediction_direction": direction,
            "prediction_confidence": round(confidence, 2),
        }

    except Exception as e:
        return {"error": f"Erro no preditor: {str(e)}"}
