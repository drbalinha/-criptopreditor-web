from flask import Flask, render_template, jsonify, request
import os
import pandas as pd
import ccxt
from supabase import create_client, Client
import preditor


# ========================================
# SUPABASE CONFIG
# ========================================
SUPABASE_URL = "https://ocivodqbfezaouctqydq.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jaXZvZHFiZmV6YW91Y3RxeWRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ3MTQ5NjcsImV4cCI6MjA4MDI5MDk2N30."
    "nCErkNisbwxUGH_5NDSY_4IGFw5frV13FHWx-orvOGU"
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = Flask(__name__)


# ========================================
# FUNÇÕES AUXILIARES
# ========================================

def fetch_ohlcv(symbol, timeframe):
    """Busca dados reais do Binance via CCXT."""
    try:
        exchange = ccxt.binance()
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=500)

        rows = []
        for t, o, h, l, c, v in raw:
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


def save_to_supabase(symbol, timeframe, rows):
    """Insere candles no Supabase."""
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
            "volume": r["volume"],
        }).execute()

    return True


def save_prediction(symbol, timeframe, horizon, real_price, predicted_price):
    """Salva cada previsão (multi‑horizonte) no Supabase."""
    supabase.table("predictions").insert({
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": horizon,
        "timestamp": int(pd.Timestamp.utcnow().timestamp() * 1000),
        "real_price": real_price,
        "predicted_price": predicted_price,
    }).execute()


# ========================================
# ROTAS PRINCIPAIS
# ========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history_page")
def history_page():
    return render_template("history.html")


# ========================================
# PREDIÇÃO NORMAL (1 CANDLE)
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
# PREDIÇÃO MULTI-HORIZONTE (1, 7, 30)
# ========================================
@app.route("/predict_multi/<path:symbol>", methods=["GET"])
def multi_route(symbol):
    try:
        safe = symbol.replace("/", "_")
        timeframe = "1d"

        result = preditor.run_prediction(safe)

        if "error" in result:
            return jsonify(result), 500

        # Salvar previsões no Supabase
        current_price = result["current_price"]

        save_prediction(safe, timeframe, 1, current_price, result["horizons"]["1"])
        save_prediction(safe, timeframe, 7, current_price, result["horizons"]["7"])
        save_prediction(safe, timeframe, 30, current_price, result["horizons"]["30"])

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500


# ========================================
# HISTÓRICO DE PREDIÇÕES (USADO PELO PAINEL)
# ========================================
@app.route("/prediction_history/<path:symbol>", methods=["GET"])
def prediction_history(symbol):
    safe = symbol.replace("/", "_")

    query = (
        supabase.table("predictions")
        .select("*")
        .eq("symbol", safe)
        .order("timestamp", desc=False)
        .execute()
    )

    return jsonify({"history": query.data})


# ========================================
# HISTÓRICO OHLC MULTI-TIMEFRAME (BINANCE + SUPABASE)
# ========================================
@app.route("/history/<path:symbol>", methods=["GET"])
def history_route(symbol):

    timeframe = request.args.get("tf", "1d")
    safe = symbol.replace("/", "_").replace(".", "")

    # Buscar no Supabase
    query = (
        supabase.table("ohlc")
        .select("*")
        .eq("symbol", safe)
        .eq("timeframe", timeframe)
        .order("timestamp", desc=False)
        .execute()
    )

    data = query.data

    # Se não existir, busca na Binance e grava
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
# EXECUTAR SERVIDOR LOCAL
# ========================================
if __name__ == "__main__":
    app.run()
