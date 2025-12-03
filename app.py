from flask import Flask, render_template, jsonify, request
import os
import pandas as pd
import ccxt
from supabase import create_client, Client
import preditor


# ========================================
# CONFIG SUPABASE
# ========================================

SUPABASE_URL = "https://ocivodqbfezaouctqydq.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jaXZvZHFiZmV6YW91Y3RxeWRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ3MTQ5NjcsImV4cCI6MjA4MDI5MDk2N30.nCErkNisbwxUGH_5NDSY_4IGFw5frV13FHWx-orvOGU"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = Flask(__name__)


# ========================================
# FUNÇÃO: Buscar OHLC real da Binance
# ========================================
def fetch_ohlcv(symbol, timeframe):
    try:
        exchange = ccxt.binance()
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=500)

        rows = []
        for t, o, h, l, c, v in data:
            rows.append({
                "timestamp": t,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })
        return rows

    except Exception:
        return []


# ========================================
# FUNÇÃO: Salvar no Supabase
# ========================================
def save_to_supabase(symbol, timeframe, rows):
    if not rows:
        return False

    safe = symbol.replace("/", "_")

    for r in rows:
        supabase.table("ohlc").insert({
            "symbol": safe,
            "timeframe": timeframe,
            "timestamp": r["timestamp"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"]
        }).execute()

    return True


# ========================================
# ROTA PRINCIPAL (Dashboard)
# ========================================
@app.route("/")
def index():
    return render_template("index.html")


# ========================================
# ROTA DE PREDIÇÃO (1 horizonte)
# ========================================
@app.route("/predict/<path:symbol>", methods=["GET"])
def predict_route(symbol):
    try:
        safe_symbol = symbol.replace("/", "_").replace(".", "")

        result = preditor.run_prediction(safe_symbol)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Erro inesperado no servidor: {str(e)}"}), 500


# ========================================
# ROTA DE PREVISÃO MULTI-HORIZONTE (1, 7, 30)
# ========================================
@app.route("/predict_multi/<path:symbol>", methods=["GET"])
def multi_route(symbol):
    try:
        safe = symbol.replace("/", "_")

        result = preditor.run_prediction(safe)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Erro inesperado no servidor: {str(e)}"}), 500


# ========================================
# ROTA DE HISTÓRICO MULTI-TIMEFRAME
# ========================================
@app.route("/history/<path:symbol>", methods=["GET"])
def history_route(symbol):

    timeframe = request.args.get("tf", "1d")  # default 1D
    safe = symbol.replace("/", "_").replace(".", "")

    query = (
        supabase.table("ohlc")
        .select("*")
        .eq("symbol", safe)
        .eq("timeframe", timeframe)
        .order("timestamp", desc=False)
        .execute()
    )

    data = query.data

    # Se não existe histórico no Supabase, buscar da Binance e preencher
    if not data:
        rows = fetch_ohlcv(symbol, timeframe)
        if rows:
            save_to_supabase(symbol, timeframe, rows)

        query = (
            supabase.table("ohlc")
            .select("*")
            .eq("symbol", safe)
            .eq("timeframe", timeframe)
            .order("timestamp", desc=False)
            .execute()
        )
        data = query.data

    if not data:
        return jsonify({"history": []})

    df = pd.DataFrame(data)
    df = df.sort_values("timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["sma7"] = df["close"].rolling(7).mean()
    df["sma30"] = df["close"].rolling(30).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    df["rsi14"] = 100 - (100 / (1 + RS))

    history = []
    for _, r in df.iterrows():
        history.append({
            "timestamp": str(r["timestamp"]),
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "sma7": None if pd.isna(r["sma7"]) else float(r["sma7"]),
            "sma30": None if pd.isna(r["sma30"]) else float(r["sma30"]),
            "rsi14": None if pd.isna(r["rsi14"]) else float(r["rsi14"]),
        })

    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "history": history
    })


# ========================================
# EXECUTAR SERVIDOR
# ========================================
if __name__ == "__main__":
    app.run()
