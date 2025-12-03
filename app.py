from flask import Flask, render_template, jsonify, request, redirect, session
import os
import pandas as pd
import ccxt
from supabase import create_client, Client
import bcrypt
import preditor
from datetime import datetime, timedelta


# 
# CACHE DE PREDIÇÕES (5 minutos)
# 
prediction_cache = {}
CACHE_DURATION = timedelta(minutes=5)

def get_cached_prediction(symbol):
    if symbol in prediction_cache:
        cached_data, cached_time = prediction_cache[symbol]
        if datetime.now() - cached_time < CACHE_DURATION:
            return cached_data
    return None

def set_cached_prediction(symbol, data):
    prediction_cache[symbol] = (data, datetime.now())


# 
# CONFIG FLASK SESSIONS
# 
app = Flask(__name__)
app.secret_key = "chave-super-secreta-trocar-depois"


# 
# CONFIG SUPABASE
# 
SUPABASE_URL = "https://ocivodqbfezaouctqydq.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9jaXZvZHFiZmV6YW91Y3RxeWRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ3MTQ5NjcsImV4cCI6MjA4MDI5MDk2N30."
    "nCErkNisbwxUGH_5NDSY_4IGFw5frV13FHWx-orvOGU"
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# 
# FUNÇÃO: SALVA PREVISÕES
# 
def save_prediction(symbol, timeframe, horizon, real_price, predicted_price):
    supabase.table("predictions").insert({
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": horizon,
        "timestamp": int(pd.Timestamp.utcnow().timestamp() * 1000),
        "real_price": real_price,
        "predicted_price": predicted_price
    }).execute()


# 
# FUNÇÃO: BUSCAR OHLC
# 
def fetch_ohlcv(symbol, timeframe):
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


# 
# FUNÇÃO: SALVAR NO SUPABASE
# 
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
            "volume": r["volume"],
        }).execute()

    return True


# 
# LOGIN PAGE
# 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        user_query = supabase.table("users").select("*").eq("email", email).execute()
        if not user_query.data:
            return "Usuário não encontrado"

        user = user_query.data[0]
        hashed = user["password_hash"]

        if bcrypt.checkpw(password.encode(), hashed.encode()):
            session["user"] = email
            return redirect("/")

        return "Senha incorreta"

    return render_template("login.html")


# 
# REGISTER PAGE
# 
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        supabase.table("users").insert({
            "email": email,
            "password_hash": hashed
        }).execute()

        return redirect("/login")

    return render_template("register.html")


# 
# LOGOUT
# 
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# 
# PÁGINA PRINCIPAL (PROTEGIDA)
# 
@app.route("/")
def index():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")


# 
# HISTÓRICO (PROTEGIDO)
# 
@app.route("/history_page")
def history_page():
    if "user" not in session:
        return redirect("/login")
    return render_template("history.html")


# 
# PREDIÇÃO 1 DIA (COM CACHE)
# 
@app.route("/predict/<path:symbol>", methods=["GET"])
def predict_route(symbol):
    safe = symbol.replace("/", "_")

    # Verificar cache
    cached = get_cached_prediction(safe)
    if cached:
        return jsonify(cached)

    # Gerar nova predição
    result = preditor.run_prediction(safe)

    # Salvar no cache
    if "error" not in result:
        set_cached_prediction(safe, result)

    return jsonify(result)


# 
# PREDIÇÃO MULTI-HORIZONTE (1, 7, 30)
# 
@app.route("/predict_multi/<path:symbol>", methods=["GET"])
def multi_route(symbol):
    safe = symbol.replace("/", "_")
    timeframe = "1d"

    # Verificar cache
    cached = get_cached_prediction(safe)
    if cached:
        return jsonify(cached)

    result = preditor.run_prediction(safe)

    if "error" in result:
        return jsonify(result), 500

    # Salvar previsões no Supabase
    current_price = result["current_price"]

    save_prediction(safe, timeframe, 1, current_price, result["horizons"]["1"])
    save_prediction(safe, timeframe, 7, current_price, result["horizons"]["7"])
    save_prediction(safe, timeframe, 30, current_price, result["horizons"]["30"])

    # Salvar no cache
    set_cached_prediction(safe, result)

    return jsonify(result)


# 
# HISTÓRICO DE PREVISÕES
# 
@app.route("/prediction_history/<path:symbol>")
def prediction_history(symbol):
    safe = symbol.replace("/", "_")
    q = supabase.table("predictions").select("*").eq("symbol", safe).execute()
    return jsonify({"history": q.data})


# 
# HISTÓRICO OHLC MULTI-TIMEFRAME
# 
@app.route("/history/<path:symbol>")
def history(symbol):

    timeframe = request.args.get("tf", "1d")
    safe = symbol.replace("/", "_")

    q = (
        supabase.table("ohlc")
        .select("*")
        .eq("symbol", safe)
        .eq("timeframe", timeframe)
        .order("timestamp", desc=False)
        .execute()
    )

    data = q.data

    if not data:
        rows = fetch_ohlcv(symbol, timeframe)
        save_to_supabase(symbol, timeframe, rows)

        q = (
            supabase.table("ohlc")
            .select("*")
            .eq("symbol", safe)
            .eq("timeframe", timeframe)
            .order("timestamp", desc=False)
            .execute()
        )
        data = q.data

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

    return jsonify({"history": history})


# 
# RUN
# 
if __name__ == "__main__":
    app.run()
