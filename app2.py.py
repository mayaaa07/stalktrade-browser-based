import os
import time
import random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import yfinance as yf
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import en_core_web_sm
from flask import render_template
import pickle
import sqlite3
from flask import Flask, request, jsonify, session
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = "supersecretkey"
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*") 

@socketio.on("subscribe")
def handle_subscribe(symbol):
    print(f"Client subscribed to {symbol}")
    while True:
        price = round(150 + random.uniform(-2, 2), 2)
        socketio.emit("price_update", {"symbol": symbol, "price": price})
        time.sleep(1)


# --- Global In-Memory Store ---
app.users = {}  # Store registered users
app.alerts = []  # Store price alerts

# --- NLP & Chatbot Setup ---
nlp = en_core_web_sm.load()
sia = SentimentIntensityAnalyzer()
trading_bot = ChatBot("TradingBot")
trainer = ChatterBotCorpusTrainer(trading_bot)
trainer.train("chatterbot.corpus.english")




# Initialize DB
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT UNIQUE  
        )
    """)
    conn.commit()
    conn.close()

init_db()  # call once at startup


# --- Registration ---
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return jsonify({"message": "Registration successful"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 400


# --- Login ---
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()

    if user:
        session["user"] = username
        return jsonify({"message": "Login successful", "user": username})
    return jsonify({"error": "Invalid credentials"}), 401


# --- Logout ---
@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"message": "Logged out"})


# --- Get current user ---
@app.route("/api/user")
def get_user():
    user = session.get("user")
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Not logged in"}), 401

# --- Chart Data (OHLCV) ---
period_map = {
    "1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo",
    "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y"
}

@app.route("/api/ohlcv/<symbol>/<period>")
def get_ohlcv(symbol, period):
    try:
        data = yf.download(symbol, period=period_map.get(period, "1mo"))
        candles = [{
            "t": t.isoformat(),
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": float(row["Volume"])
        } for t, row in data.iterrows() if not any(row.isnull())]
        return jsonify({"candles": candles or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Watchlist Ticker Tape ---
@app.route("/api/ticker_tape")
def ticker_tape():
    symbols = request.args.get("symbols", "AAPL,MSFT,TSLA,GOOG").split(",")
    tape = []
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        info = stock.info
        tape.append({
            "symbol": symbol,
            "price": info.get("currentPrice", 0),
            "change": info.get("regularMarketChange", 0),
            "percent": info.get("regularMarketChangePercent", 0)
        })
    return jsonify({"ticker_tape": tape})

# --- Market Movers ---
@app.route("/api/movers")
def market_movers():
    gainers = [
        {"symbol": "NVDA", "name": "NVIDIA", "price": 128.32, "change": 4.2, "percent": 3.4},
        {"symbol": "AMD", "name": "AMD", "price": 89.45, "change": 2.9, "percent": 3.3}
    ]
    losers = [
        {"symbol": "NFLX", "name": "Netflix", "price": 480.67, "change": -10.3, "percent": -2.1}
    ]
    return jsonify({"gainers": gainers, "losers": losers})

# --- Indices Data ---
@app.route("/api/indices")
def market_indices():
    indices = [
        {"symbol": "^GSPC", "name": "S&P 500", "price": 5450.12, "change": 15.4, "percent": 0.29},
        {"symbol": "^IXIC", "name": "Nasdaq", "price": 17890.45, "change": 30.5, "percent": 0.17}
    ]
    return jsonify({"indices": indices})

# --- Market News ---
@app.route("/api/news")
def financial_news():
    news_items = [
        {
            "title": "Tech Stocks Surge",
            "content": "Nasdaq rallies as chip stocks soar.",
            "source": "Reuters",
            "sentiment": sia.polarity_scores("great growth positive")["compound"]
        },
        {
            "title": "Oil Prices Plunge",
            "content": "Crude dips as supply rises.",
            "source": "Bloomberg",
            "sentiment": sia.polarity_scores("falling oil negative")["compound"]
        }
    ]
    return jsonify({"news": news_items})

with open("nlp_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("qa_data.pkl", "rb") as f:
    qa_data = pickle.load(f)


stock_map = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "netflix": "NFLX",
    "reliance": "RELIANCE.NS",  # NSE stock
    "tata": "TCS.NS",
    "meta": "META",
    "nvidia": "NVDA"
}

@app.route("/api/assistant", methods=["POST"])
def assistant():
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"response": "Please ask something."})

    # 1️⃣ Check QA Knowledge Base first
    for q, a in qa_data:
        if q.lower() in query.lower():
            return jsonify({"response": a})

    # 2️⃣ Otherwise use intent model
    X = vectorizer.transform([query])
    intent = model.predict(X)[0]

    if intent == "add_watchlist":
        for name, symbol in stock_map.items():
            if name in query.lower():
                if symbol not in watchlist:
                    watchlist.append(symbol)
                return jsonify({"response": f"{symbol} has been added to your watchlist."})
        return jsonify({"response": "I didn’t recognize the stock you want to add."})

    elif intent == "remove_watchlist":
        for name, symbol in stock_map.items():
            if name in query.lower():
                if symbol in watchlist:
                    watchlist.remove(symbol)
                return jsonify({"response": f"{symbol} has been removed from your watchlist."})
        return jsonify({"response": "I didn’t recognize the stock you want to remove."})

    elif intent == "check_watchlist":
        if not watchlist:
            return jsonify({"response": "Your watchlist is empty."})
        return jsonify({"response": f"Here’s your current watchlist: {watchlist}"})

    elif intent == "check_portfolio":
        return jsonify({"response": "Your portfolio contains Apple and Tesla with gains today."})

    else:
        return jsonify({"response": f"You asked: {query}. I’ll learn more soon."})


# --- Sentiment Info ---
@app.route("/api/sentiment")
def market_sentiment():
    return jsonify({
        "overall": "Positive",
        "score": 0.67,
        "trending": ["TSLA", "NVDA", "AAPL"],
        "fear_and_greed": 72,
        "comment": "Investors optimistic after earnings."
    })

# --- Place Order ---
@app.route("/api/order", methods=["POST"])
def place_order():
    data = request.json
    print(f"Order received: {data}")
    return jsonify({"message": "Order placed."})

# --- Voice Command ---
@app.route("/process_voice_command", methods=["POST"])
def process_voice_command():
    command = request.json.get("command", "").lower()
    if "buy" in command and "shares" in command:
        return jsonify({"action": "buy", "symbol": "AAPL", "shares": 10})
    elif "show" in command and "chart" in command:
        return jsonify({"action": "show_chart", "symbol": "AAPL"})
    return jsonify({"error": "Command not understood"})

# --- News Analysis ---
@app.route("/analyze_news", methods=["POST"])
def analyze_news():
    news_text = request.json.get("news", "")
    scores = sia.polarity_scores(news_text)
    mentioned = [s for s in ["AAPL", "TSLA", "MSFT"] if s.lower() in news_text.lower()]
    return jsonify({"sentiment": scores, "mentioned_stocks": mentioned})

# --- Alerts ---
@app.route("/api/alerts", methods=["GET", "POST", "DELETE"])
def price_alerts():
    if request.method == "GET":
        return jsonify({"alerts": app.alerts})
    elif request.method == "POST":
        data = request.json
        new_alert = {**data, "id": int(np.random.randint(100000)), "triggered": False}
        app.alerts.append(new_alert)
        return jsonify({"success": True, "alert": new_alert})
    elif request.method == "DELETE":
        alert_id = int(request.args.get("id"))
        app.alerts = [a for a in app.alerts if a["id"] != alert_id]
        return jsonify({"success": True})

# --- Stock Info ---
@app.route("/api/stock/<symbol>")
def get_stock(symbol):
    stock = yf.Ticker(symbol)
    
    # Try to get recent history
    df = stock.history(period="1d", interval="1m")
    if df.empty:
        return jsonify({"error": f"No information available for {symbol.upper()}"}), 404
    
    latest = df.iloc[-1]
    price = latest["Close"]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else price
    change = price - prev_close
    percent = (change / prev_close) * 100 if prev_close != 0 else 0
    
    return jsonify({
        "symbol": symbol.upper(),
        "price": float(price),
        "change": float(change),
        "percent": float(percent)

    })

@app.route("/api/nlp", methods=["POST"])
def nlp():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "Please enter a question."}), 400

    # 1️⃣ Sentiment analysis
    sentiment = sia.polarity_scores(query)
    sentiment_label = "Neutral"
    if sentiment["compound"] > 0.2:
        sentiment_label = "Positive"
    elif sentiment["compound"] < -0.2:
        sentiment_label = "Negative"

    # 2️⃣ Named entity recognition with spaCy
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]

    # 3️⃣ Stock lookup if an entity looks like a ticker/company
    stock_info = ""
    if entities:
        try:
            symbol = entities[0].upper()
            stock = yf.Ticker(symbol)
            price = stock.info.get("currentPrice")
            if price:
                stock_info = f"{symbol} is trading at ${price}"
        except Exception:
            stock_info = f"Couldn't fetch stock info for {entities[0]}"

    # 4️⃣ ChatterBot response
    bot_response = trading_bot.get_response(query)

    final_answer = f"{bot_response} (Sentiment: {sentiment_label}) {stock_info}"

    return jsonify({"answer": final_answer})




watchlist = [
    {"name": "Apple", "price": 175.0},
    {"name": "Microsoft", "price": 300.0},
    {"name": "Amazon", "price": 3300.0},
    {"name": "Google", "price": 2900.0},
    {"name": "Tesla", "price": 700.0},
    {"name": "Meta", "price": 340.0},
    {"name": "Netflix", "price": 450.0},
    {"name": "NVIDIA", "price": 500.0},
    {"name": "Intel", "price": 55.0},
    {"name": "AMD", "price": 95.0},
    {"name": "IBM", "price": 135.0},
    {"name": "Oracle", "price": 85.0},
    {"name": "Salesforce", "price": 220.0},
    {"name": "Adobe", "price": 520.0},
    {"name": "Spotify", "price": 220.0},
    {"name": "Snap", "price": 15.0},
    {"name": "PayPal", "price": 120.0},
    {"name": "Uber", "price": 50.0},
    {"name": "Lyft", "price": 30.0},
    {"name": "Shopify", "price": 75.0},
    {"name": "Alibaba", "price": 120.0},
    {"name": "Tencent", "price": 55.0},
    {"name": "Baidu", "price": 160.0},
    {"name": "Sony", "price": 100.0},
    {"name": "Samsung", "price": 60.0},
    {"name": "LG", "price": 80.0},
    {"name": "Panasonic", "price": 40.0},
    {"name": "Honda", "price": 35.0},
    {"name": "Toyota", "price": 150.0},
    {"name": "Ford", "price": 14.0},
    {"name": "General Motors", "price": 40.0},
    {"name": "Coca-Cola", "price": 60.0},
    {"name": "PepsiCo", "price": 180.0},
    {"name": "Walmart", "price": 155.0},
    {"name": "Costco", "price": 600.0},
    {"name": "McDonald's", "price": 280.0},
    {"name": "Starbucks", "price": 110.0},
    {"name": "Visa", "price": 225.0},
    {"name": "Mastercard", "price": 360.0},
    {"name": "American Express", "price": 180.0},
    {"name": "JPMorgan", "price": 150.0},
    {"name": "Goldman Sachs", "price": 400.0},
    {"name": "Bank of America", "price": 35.0},
    {"name": "Chevron", "price": 170.0},
    {"name": "ExxonMobil", "price": 120.0},
    {"name": "Shell", "price": 50.0},
    {"name": "BP", "price": 35.0},
    {"name": "Unilever", "price": 60.0},
    {"name": "Procter & Gamble", "price": 155.0},
    {"name": "Johnson & Johnson", "price": 180.0},
    {"name": "Pfizer", "price": 45.0},
    {"name": "Moderna", "price": 125.0}
]


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    return jsonify({"watchlist": watchlist})

@app.route("/api/watchlist/add", methods=["POST"])
def add_watchlist():
    data = request.json
    symbol = data.get("symbol")
    if symbol and symbol not in watchlist:
        watchlist.append(symbol)
    return jsonify({"watchlist": watchlist})

@app.route("/api/watchlist/remove", methods=["POST"])
def remove_watchlist():
    data = request.json
    symbol = data.get("symbol")
    if symbol in watchlist:
        watchlist.remove(symbol)
    return jsonify({"watchlist": watchlist})




@app.route('/api/yfinance-info/<symbol>')
def yfinance_info(symbol):
    return jsonify({"symbol": symbol, "price": 150, "change": 1.2, "changePercent": 0.8})

portfolio = []
@app.route("/api/buy", methods=["POST"])

def buy_stock():
    data = request.json
    symbol = data.get("symbol")
    quantity = int(data.get("quantity", 1))
    price = float(data.get("price", 100))  # you can replace with live price later

    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    # Add to portfolio
    portfolio.append({
        "symbol": symbol,
        "quantity": quantity,
        "price": price
    })

    print(f"[BUY] {quantity} shares of {symbol} at {price}")
    return jsonify({"message": f"Bought {quantity} shares of {symbol} at {price}."})



@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    return jsonify(portfolio)


@app.route("/")
def index():
    return render_template("working_candle.html")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)


