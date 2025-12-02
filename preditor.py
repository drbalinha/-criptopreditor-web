# Arquivo: preditor.py (VERSAO FINAL v5 - MAX POWER GRATUITO)

import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import joblib
import json

MODEL_STORAGE_PATH = 'temp_models'
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def calculate_rsi(series, period=14):
    """Calcula o Indice de Forca Relativa (RSI) sem usar bibliotecas externas."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta &lt; 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(df):
    """Cria features (pistas) para o modelo."""
    df_copy = df.copy()
    # Lags (memoria de curto prazo)
    for i in range(1, 8):
        df_copy[f'close_lag_{i}'] = df_copy['close'].shift(i)
    # Medias Moveis (tendencias)
    df_copy['sma_7'] = df_copy['close'].rolling(window=7).mean()
    df_copy['sma_30'] = df_copy['close'].rolling(window=30).mean()
    # Volatilidade
    df_copy['daily_volatility'] = df_copy['high'] - df_copy['low']
    # FEATURE NOVA: RSI (momento do mercado)
    df_copy['rsi_14'] = calculate_rsi(df_copy['close'], 14)
    return df_copy

def run_prediction(symbol):
    symbol_safe = symbol.replace('/', '_').replace('.', '')
    model_path = os.path.join(MODEL_STORAGE_PATH, f'model_xgb_{symbol_safe}.pkl')
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            print(f"Modelo XGBoost para {symbol} nao encontrado. Treinando um novo...")
            file_path = f'Historico_Moedas/historico_{symbol_safe}.csv'
            if not os.path.exists(file_path):
                return {"error": f"Arquivo de historico para {symbol} nao encontrado."}
            
            df = pd.read_csv(file_path)
            df_features = create_features(df)
            
            # --- MUDANCA CRUCIAL: O ALVO DA PREVISAO ---
            # Prever 1 se o preco de amanha for MAIOR que o de hoje, senao 0.
            df_features['target'] = np.where(df_features['close'].shift(-1) > df_features['close'], 1, 0)
            
            df_features.dropna(inplace=True)
            
            features = [col for col in df_features.columns if 'lag' in col or 'sma' in col or 'volatility' in col or 'rsi' in col]
            target_col = 'target'
            
            X_train = df_features[features]
            y_train = df_features[target_col]
            
            # --- MUDANCA CRUCIAL: O MODELO ---
            model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)

        # PREVISAO
        df_full = pd.read_csv(f'Historico_Moedas/historico_{symbol_safe}.csv')
        df_full_features = create_features(df_full)
        
        features_to_predict = [col for col in df_full_features.columns if 'lag' in col or 'sma' in col or 'volatility' in col or 'rsi' in col]
        X_pred_row = df_full_features[features_to_predict].iloc[-1:].copy()
        
        X_pred_row.fillna(method='ffill', inplace=True)
        X_pred_row.fillna(method='bfill', inplace=True)

        prediction_code = model.predict(X_pred_row)[0]
        prediction_proba = model.predict_proba(X_pred_row)[0]

        direction = "ALTA" if prediction_code == 1 else "BAIXA"
        confidence = prediction_proba[prediction_code] * 100

        response = {
            "symbol": symbol,
            "prediction_direction": direction,
            "prediction_confidence": round(confidence, 2),
            "current_price": round(float(df_full['close'].iloc[-1]), 2),
        }
        
        print(f"Previsao XGBoost para {symbol}: {response}")
        return response

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"Ocorreu um erro: {str(e)}"}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        resultado = run_prediction(sys.argv[1])
        print(json.dumps(resultado, indent=4))
