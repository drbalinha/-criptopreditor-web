# Arquivo: app.py (VERSAO FINAL - Com auto-atualizacao)

from flask import Flask, render_template, jsonify, request
import preditor
import subprocess # Biblioteca para executar scripts externos
import os
import shutil     # Biblioteca para deletar pastas

app = Flask(__name__)

# --- CONFIGURACAO DE SEGURANCA ---
# Este token secreto impede que qualquer um acione a atualizacao.
# Mude para qualquer outra coisa, se quiser.
UPDATE_TOKEN = "um_token_muito_secreto_123"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<path:symbol>', methods=['GET'])
def predict_route(symbol):
    try:
        result = preditor.run_prediction(symbol)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Erro inesperado no servidor: {str(e)}"}), 500

# --- NOSSO NOVO "BOTAO VERMELHO SECRETO" ---
@app.route('/trigger-update', methods=['POST'])
def trigger_update_route():
    # 1. Verifica se a chamada tem o token de seguranca
    client_token = request.headers.get('X-Update-Token')
    if client_token != UPDATE_TOKEN:
        return jsonify({"error": "Acesso nao autorizado"}), 403

    try:
        # 2. Executa o script de coleta de dados em segundo plano
        print("Iniciando a execucao do script de coleta de dados...")
        subprocess.run(['python', 'coleta_dados.py'], check=True)
        print("Script de coleta de dados executado com sucesso.")
        
        # 3. APAGA os modelos antigos para forcar o re-treinamento
        if os.path.exists(preditor.MODEL_STORAGE_PATH):
            shutil.rmtree(preditor.MODEL_STORAGE_PATH)
            os.makedirs(preditor.MODEL_STORAGE_PATH, exist_ok=True)
            print("Modelos antigos removidos para forcar novo treinamento.")
        
        return jsonify({"message": "Coleta de dados e limpeza de modelos concluidos com sucesso."}), 200

    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o script de coleta: {e}")
        return jsonify({"error": "Falha ao executar o script de coleta de dados."}), 500
    except Exception as e:
        print(f"Erro inesperado durante a atualizacao: {e}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
