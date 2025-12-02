# Arquivo: app.py (VERSAO FINAL - Compativel com Scikit-learn)

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import preditor

app = Flask(__name__)
CORS(app)

print(">>> APLICACAO WEB (Scikit-learn) INICIADA &lt;&lt;&lt;")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<path:symbol>', methods=['GET'])
def predict(symbol):
    print(f"Recebida requisicao de previsao para: {symbol}")
    result = preditor.run_prediction(symbol) 
    
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
