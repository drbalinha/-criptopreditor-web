# Arquivo: preditor.py (VERSAO FINAL - Scikit-learn Gratuito)

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import json

# No plano gratuito, os arquivos sao temporarios.
# Vamos salvá-los em uma pasta local que sera criada a cada deploy.
MODEL_STORAGE_PATH = 'temp_models' 
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

def run_prediction(symbol):
    symbol_safe = symbol.replace('/', '_').replace('.', '')
    
    model_path = os.path.join(MODEL_STORAGE_PATH, f'model_skl_{symbol_safe}.pkl')
    
    try:
        # 1. TENTAR CARREGAR O MODELO
        if os.path.exists(model_path):
            print(f"Modelo Scikit-learn para {symbol} encontrado. Carregando...")
            model = joblib.load(model_path)
        else:
            # 2. SE NAO EXISTE, TREINA UM NOVO (o treinamento e quase instantaneo)
            print(f"Modelo Scikit-learn para {symbol} nao encontrado. Treinando um novo...")
            
            file_path = f'Historico_Moedas/historico_{symbol_safe}.csv'
            if not os.path.exists(file_path):
                return {"error": f"Arquivo de historico para {symbol} nao encontrado."}
            
            df = pd.read_csv(file_path)
            df['prediction'] = df['close'].shift(-1)
            df.dropna(inplace=True)
            
            X = np.array(df['close']).reshape(-1, 1)
            y = np.array(df['prediction'])
            
            model = LinearRegression()
            model.fit(X, y)
            
            joblib.dump(model, model_path)
            print(f"Modelo para {symbol} treinado e salvo em {model_path}")

        # 3. FAZER A PREVISAO
        file_path = f'Historico_Moedas/historico_{symbol_safe}.csv'
        df = pd.read_csv(file_path)
        last_price = np.array(df['close'].iloc[-1]).reshape(-1, 1)
        predicted_price = model.predict(last_price)
        
        response = {
            "symbol": symbol,
            "predicted_price": round(float(predicted_price[0]), 2),
            "current_price": round(float(df['close'].iloc[-1]), 2),
        }
        
        print(f"Previsao Scikit-learn para {symbol}: {response['predicted_price']}")
        return response

    except Exception as e:
        return {"error": f"Ocorreu um erro durante a previsao: {str(e)}"}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        moeda = sys.argv[1]
        resultado = run_prediction(moeda)
        print(json.dumps(resultado, indent=4))
    else:
        print("Por favor, forneca o simbolo da moeda como argumento.")
