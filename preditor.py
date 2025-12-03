import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
import joblib

MODEL_STORAGE_PATH = 'temp_models'
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)


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
        classifier = XGBClassifier(eval_metric="logloss")
        classifier.fit(dfX[features], dfX["target"])
        joblib.dump(classifier, model_path)

    if os.path.exists(reg_path):
        regressor = joblib.load(reg_path)
    else:
        dfX = create_features(df.copy())
        dfX["target_price"] = dfX["close"].shift(-1)
        dfX = dfX.dropna()
        features = [c for c in dfX.columns if "lag" in c or "sma" in c or "rsi" in c or "volatility" in c]
        regressor = RandomForestRegressor()
        regressor.fit(dfX[features], dfX["target_price"])
        joblib.dump(regressor, reg_path)

    return classifier, regressor


# ============================
# PREVISÃO DE 1 DIA (PADRÃO)
# ============================
def predict_single(df, classifier, regressor):
    dfX = create_features(df.copy())
    dfX = dfX.fillna(method="ffill").fillna(method="bfill")

    last = dfX.iloc[-1:]
    features = [c for c in dfX.columns if "lag" in c or "sma" in c or "rsi" in c or "volatility" in c]

    p_class = classifier.predict(last[features])[0]
    p_prob = classifier.predict_proba(last[features])[0]
    p_price = regressor.predict(last[features])[0]

    direction = "ALTA" if p_class == 1 else "BAIXA"
    confidence = float(p_prob[p_class]) * 100

    return p_price, direction, confidence


# ============================
# PREVISÃO MULTI-HORIZONTE
# ============================
def predict_multi_horizon(df, classifier, regressor, horizons=[1,7,30]):
    results = {}

    for h in horizons:
        df_temp = df.copy()

        for _ in range(h):
            next_price, _, _ = predict_single(df_temp, classifier, regressor)
            df_temp.loc[len(df_temp)] = {
                "open": next_price,
                "high": next_price,
                "low": next_price,
                "close": next_price,
                "volume": df_temp["volume"].iloc[-1]
            }

        final_pred = df_temp["close"].iloc[-1]
        results[str(h)] = float(final_pred)

    return results


# ============================
# API PÚBLICA
# ============================
def run_prediction(symbol):
    try:
        file_path = f"Historico_Moedas/historico_{symbol}.csv"

        if not os.path.exists(file_path):
            return {"error": "Arquivo não encontrado"}

        df = pd.read_csv(file_path)
        current_price = float(df["close"].iloc[-1])

        classifier, regressor = load_models(symbol, df)

        # Predição 1 Horizonte
        pred1, direction, confidence = predict_single(df, classifier, regressor)

        # Predições Multi-Horizonte
        multi = predict_multi_horizon(df, classifier, regressor, horizons=[1,7,30])

        return {
            "symbol": symbol.replace("_", "/"),
            "current_price": current_price,
            "prediction_direction": direction,
            "prediction_confidence": round(confidence,2),
            "predicted_price_1": round(pred1,2),
            "horizons": {
                "1": round(multi["1"],2),
                "7": round(multi["7"],2),
                "30": round(multi["30"],2)
            }
        }

    except Exception as e:
        return {"error": f"Erro: {str(e)}"}
