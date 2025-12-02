# Arquivo: preditor.py (VERSAO FINAL v2 - Correcao de Permissao)

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
import json

# --- CORRECAO ESTA AQUI ---
# Usamos diretamente o caminho do disco montado, sem tentar criar uma subpasta que causa erro de permissao.
# O proprio Render garante que este diretorio exista.
MODEL_STORAGE_PATH = '/var/data' 

# Garantimos que o caminho exista, o que eh seguro de fazer no /var/data
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def create_dataset(dataset, look_back=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_prediction(symbol, allow_training=False):
    symbol_safe = symbol.replace('/', '_').replace('.', '')
    
    # O caminho completo do arquivo agora eh construido corretamente
    model_path = os.path.join(MODEL_STORAGE_PATH, f'model_{symbol_safe}.h5')
    scaler_path = os.path.join(MODEL_STORAGE_PATH, f'scaler_{symbol_safe}.pkl')
    
    if os.path.exists(model_path):
        print(f"Modelo para {symbol} encontrado em '{model_path}'. Carregando...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        if not allow_training:
            print(f"Modelo para {symbol} nao encontrado, e o treinamento nao esta permitido nesta chamada.")
            return {"error": f"Modelo para {symbol} ainda nao foi treinado."}
        
        print(f"Modelo para {symbol} nao encontrado. Iniciando treinamento (allow_training=True)...")
        
        file_path = f'Historico_Moedas/historico_{symbol_safe}.csv'
        if not os.path.exists(file_path):
            return {"error": f"Arquivo de historico para {symbol} nao encontrado."}
        
        df = pd.read_csv(file_path)
        data = df.filter(['close'])
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * .95))
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training_data_len), :]
        look_back = 60
        x_train, y_train = create_dataset(train_data, look_back)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(64, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

    # O codigo de previsao continua o mesmo
    file_path = f'Historico_Moedas/historico_{symbol_safe}.csv'
    df = pd.read_csv(file_path)
    data = df.filter(['close'])
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_pred = [last_60_days_scaled]
    X_pred = np.array(X_pred)
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
    pred_price = model.predict(X_pred)
    pred_price = scaler.inverse_transform(pred_price)
    
    response = {
        "symbol": symbol,
        "predicted_price": round(float(pred_price[0][0]), 2),
        "current_price": round(float(data['close'].iloc[-1]), 2),
    }
    return response

if __name__ == '__main__':
    if len(sys.argv) > 1:
        moeda = sys.argv[1]
        resultado = run_prediction(moeda, allow_training=True)
        print(json.dumps(resultado, indent=4))
    else:
        print("Por favor, forneca o simbolo da moeda como argumento.")
