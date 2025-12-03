import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
import joblib

MODEL_STORAGE_PATH = 'temp_models'
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# Cache de DataFrames carregados
df_cache = {}


# ============================
# RSI
# ============================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ============================
# FEATURES
# ============================
def create_features(df):
    df = df.copy()
    
    for i in range(1, 8):
        df[f'close_lag_{i}'] = df['close'].shift(i)

    df['sma7'] = df['close'].rolling(7).mean()
    df['sma30'] = df['close'].rolling(30).mean()
    df['volatility'] = df['high'] - df['low']
    df['rsi14'] = calculate_rsi(df['close'], 14)
    
    return df


# ============================
# CARREGAR OU TREINAR MODELOS
# ============================
def load_models(symbol, df):
    model_path = f"{MODEL_STORAGE_PATH}/model_xgb_{symbol}.pkl"
    reg_path = f"{MODEL_STORAGE_PATH}/reg_rf_{symbol}.pkl"

    if os.path.exists(model_path):
        classifier = joblib.load(model_path)
    else:
        dfX = create_features(df.copy())
        dfX["target"] = (dfX["close"].shift(-1) > dfX["close"]).astype(int)
        dfX = dfX.dropna()
        features = [c for c in dfX.columns if "lag" in c or "sma" in c or "rsi" in c or "volatility" in c]
        
        # XGBoost mais rápido
        classifier = XGBClassifier(
            eval_metric="logloss",
            n_estimators=50,  # Reduzido de 100
            max_depth=3,      # Reduzido de 6
            n_jobs=1
        )
        classifier.fit(dfX[features], dfX["target"])
        joblib.dump(classifier, model_path)

    if os.path.exists(reg_path):
        regressor = joblib.load(reg_path)
    else:
        dfX = create_features(df.copy())
        dfX["target_price"] = dfX["close"].shift(-1)
        dfX = dfX.dropna()
        features = [c for c in dfX.columns if "lag" in c or "sma" in c or "rsi" in c or "volatility" in c]
        
        # RandomForest mais rápido
        regressor = RandomForestRegressor(
            n_estimators=30,  # Reduzido de 100
            max_depth=5,      # Limitado
            n_jobs=1
        )
        regressor.fit(dfX[features], dfX["target_price"])
        joblib.dump(regressor, reg_path)

    return classifier, regressor


# ============================
# PREVISÃO DE 1 DIA (PADRÃO)
# ============================
def predict_single(df, classifier, regressor):
    dfX = create_features(df.copy())
    dfX = dfX.ffill().bfill()

    last = dfX.iloc[-1:]
    features = [c for c in dfX.columns if "lag" in c or "sma" in c or "rsi" in c or "volatility" in c]

    p_class = classifier.predict(last[features])[0]
    p_prob = classifier.predict_proba(last[features])[0]
    p_price = regressor.predict(last[features])[0]

    direction = "ALTA" if p_class == 1 else "BAIXA"
    confidence = float(p_prob[p_class]) * 100

    return p_price, direction, confidence


# ============================
# PREVISÃO MULTI-HORIZONTE (OTIMIZADA)
# ============================
def predict_multi_horizon_fast(df, classifier, regressor):
    """Versão otimizada: calcula apenas 1, 7 e 30 sem iterações completas"""
    
    dfX = create_features(df.copy())
    dfX = dfX.ffill().bfill()
    
    last = dfX.iloc[-1:]
    features = [c for c in dfX.columns if "lag" in c or "sma" in c or "rsi" in c or "volatility" in c]
    
    # Predição base
    pred_1 = regressor.predict(last[features])[0]
    
    # Estimativa rápida para 7 e 30 dias (sem iteração completa)
    current = float(df["close"].iloc[-1])
    change_rate = (pred_1 - current) / current
    
    pred_7 = pred_1 * (1 + change_rate * 3)   # Aproximação
    pred_30 = pred_1 * (1 + change_rate * 8)  # Aproximação
    
    return {
        "1": float(pred_1),
        "7": float(pred_7),
        "30": float(pred_30)
    }


# ============================
# API PÚBLICA
# ============================
def run_prediction(symbol):
    try:
        file_path = f"Historico_Moedas/historico_{symbol}.csv"

        if not os.path.exists(file_path):
            return {"error": "Arquivo não encontrado"}

        # Cache de DataFrame
        if symbol in df_cache:
            df = df_cache[symbol]
        else:
            df = pd.read_csv(file_path)
            df_cache[symbol] = df

        current_price = float(df["close"].iloc[-1])

        classifier, regressor = load_models(symbol, df)

        # Predição 1 Horizonte
        pred1, direction, confidence = predict_single(df, classifier, regressor)

        # Predições Multi-Horizonte (VERSÃO RÁPIDA)
        multi = predict_multi_horizon_fast(df, classifier, regressor)

        return {
            "symbol": symbol.replace("_", "/"),
            "current_price": current_price,
            "prediction_direction": direction,
            "prediction_confidence": round(confidence, 2),
            "predicted_price_1": round(pred1, 2),
            "horizons": {
                "1": round(multi["1"], 2),
                "7": round(multi["7"], 2),
                "30": round(multi["30"], 2)
            }
        }

    except Exception as e:
        return {"error": f"Erro: {str(e)}"}
