"""
AI Trading Monitoring Agent (Telegram Only)
- Fully conversational (no /start needed)
- Auto greetings
- /stop or 'stop' halts monitoring
- Multi-user support (no hardcoded chat_id)
"""

import os, datetime, json, time, threading, requests, yfinance as yf, pandas as pd
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)
from scorer import EnsembleScorer
from groq import Groq

load_dotenv()

# =============== ENVIRONMENT KEYS ===================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

groq_client = Groq(api_key=GROQ_API_KEY)
user_monitors = {}  # {chat_id: {"thread": thread_object, "stop_flag": threading.Event()}}

# =====================================================
# GROQ + SERPER HELPERS
# =====================================================
def groq_explain(prompt: str) -> str:
    try:
        chat = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"

def explain_anomaly_context(symbol, anomaly_time, anomaly_type="spike"):
    context_snippets = ""
    if SERPER_API_KEY:
        try:
            url = "https://google.serper.dev/news"
            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            payload = {"q": f"{symbol} stock OR earnings OR news", "num": 3}
            resp = requests.post(url, headers=headers, json=payload)
            articles = resp.json().get("news", [])
            context_snippets = "\n".join([f"{a.get('title')}: {a.get('snippet')}" for a in articles])
        except Exception as e:
            context_snippets = f"News fetch error: {e}"
    prompt = f"""
The stock {symbol} showed an {anomaly_type} anomaly around {anomaly_time}.
Recent news headlines:
{context_snippets}

Explain in 3–5 sentences the most likely reason for this anomaly based on the news context.
Be direct, confident, and analytical — no uncertainty, disclaimers, or filler phrases.
Start immediately with the reason.
"""
    return groq_explain(prompt)

# =====================================================
# TELEGRAM HELPERS
# =====================================================
def send_alert(chat_id, payload: dict):
    msg = f"""
🚨 Anomaly Detected!
Symbol: {payload['symbol']}
Price: {payload['price']}
Reason: {payload['reason']}
Details: {payload['anomaly_explanation']}
"""
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": chat_id, "text": msg}
        )
    except Exception as e:
        print(f"❌ Telegram send error for {chat_id}: {e}")

# =====================================================
# CORE AGENT
# =====================================================
def get_latest_tick(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty: return None
        price = round(float(data["Close"].iloc[-1]), 2)
        ts = data.index[-1].to_pydatetime().isoformat()
        return {"symbol": symbol, "price": price, "ts": ts}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

def log_event(payload):
    payload["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("events_log.jsonl", "a") as f:
        f.write(json.dumps(payload) + "\n")

def run_monitor(chat_id, symbols, context, mode="Custom", cycle_seconds=900):
    """
    Main monitoring loop for a user and a set of symbols.
    """
    scorer = EnsembleScorer(window_size=60)
    stop_flag = user_monitors[chat_id]["stop_flag"]

    # Start log
    print(f"[{chat_id}] Monitoring started for {mode} — {len(symbols)} symbols")
    context.bot.send_message(
        chat_id=chat_id,
        text=f"✅ Monitoring started for {mode}. You’ll receive anomaly alerts if detected.\nType 'stop' or /stop to halt."
    )

    while not stop_flag.is_set():
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{chat_id}] Monitoring cycle started at {start_time} | Interval: {cycle_seconds // 60} min")

        for sym in symbols:
            tick = get_latest_tick(sym)
            if not tick or "price" not in tick:
                continue

            score = scorer.update_and_score(sym, tick["price"])

            if score.get("ready") and score.get("final_anomaly"):
                explanation = explain_anomaly_context(sym, tick["ts"])
                payload = {
                    "symbol": sym,
                    "price": tick["price"],
                    "reason": "Anomaly detected",
                    "anomaly_explanation": explanation,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                send_alert(chat_id, payload)
                log_event(payload)
                print(f"[{chat_id}] Anomaly detected for {sym} — alert sent.")

        time.sleep(cycle_seconds)

    # Stop log after thread ends
    print(f"[{chat_id}] Monitoring stopped for {mode}")
    context.bot.send_message(chat_id=chat_id, text=f"🛑 Monitoring stopped for {mode}.")




# =====================================================
# SYMBOL UNIVERSES
# =====================================================
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_dynamic_symbols():
    """Fetch S&P 500 and NIFTY 50 symbols reliably, fallback to static lists if scraping fails."""
    try:
        # -------------------- S&P 500 --------------------
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(sp500_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        sp500_table = soup.find("table", {"id": "constituents"})
        sp500_df = pd.read_html(str(sp500_table))[0]
        sp500_symbols = sp500_df["Symbol"].tolist()
        # Clean any periods (BRK.B -> BRK-B)
        sp500_symbols = [s.replace(".", "-") for s in sp500_symbols]

        # -------------------- NIFTY 50 --------------------
        nifty_url = "https://en.wikipedia.org/wiki/NIFTY_50"
        r = requests.get(nifty_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        nifty_table = soup.find("table", {"class": "wikitable"})
        nifty_df = pd.read_html(str(nifty_table))[0]
        nifty_symbols = nifty_df["Symbol"].tolist()
        nifty_symbols = [s + ".NS" for s in nifty_symbols]  # add NSE suffix

        return {"largecap": sp500_symbols[:25] + nifty_symbols[:25],
                "midcap": sp500_symbols[25:50] + nifty_symbols[25:50]}

    except Exception as e:
        print(f"⚠️ Fallback to static lists due to: {e}")
        # Static fallback
        US_LARGECAP = ["AAPL","MSFT","AMZN","GOOG","META","TSLA","NVDA","NFLX","ADBE","INTC",
                        "ORCL","CRM","CSCO","QCOM","TXN","AVGO","AMD","IBM","HON","PYPL",
                        "PEP","KO","NKE","MCD","INTU"]
        IN_LARGECAP = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                       "HINDUNILVR.NS","KOTAKBANK.NS","LT.NS","SBIN.NS","BAJFINANCE.NS",
                       "MARUTI.NS","ITC.NS","HCLTECH.NS","AXISBANK.NS","ULTRACEMCO.NS",
                       "TECHM.NS","BAJAJFINSV.NS","NTPC.NS","ICICIPRULI.NS","TITAN.NS",
                       "BPCL.NS","ADANIPORTS.NS","SUNPHARMA.NS","WIPRO.NS","ONGC.NS"]
        US_MIDCAP = ["SNAP","TWTR","DOCU","ZM","FUBO","ROKU","CRWD","OKTA","NET","ETSY",
                     "SQ","SHOP","BMRN","EXPE","JD","PINS","UBER","LYFT","TTD","MDB",
                     "DDOG","PLAN","AFRM","COUP","CRSP"]
        IN_MIDCAP = ["MGL.NS","PAGEIND.NS","ALKEM.NS","APOLLOHOSP.NS","BALKRISIND.NS",
                     "BATAINDIA.NS","BERGEPAINT.NS","CIPLA.NS","DLF.NS","GODREJCP.NS",
                     "ICICIGI.NS","IDFC.NS","INDIGO.NS","LAURUSLABS.NS","LUPIN.NS",
                     "MUTHOOTFIN.NS","NAUKRI.NS","PIDILITIND.NS","SBILIFE.NS","SRF.NS",
                     "TATACHEM.NS","TATACONSUM.NS","TATAMOTORS.NS","TORNTPHARM.NS","ZOMATO.NS"]
        return {"largecap": US_LARGECAP + IN_LARGECAP,
                "midcap": US_MIDCAP + IN_MIDCAP}

import yfinance as yf

def validate_symbols(symbols_dict):
    """
    Validate tickers by checking if they exist on Yahoo Finance.
    Returns only valid symbols.
    """
    validated = {"largecap": [], "midcap": []}
    
    for category, tickers in symbols_dict.items():
        print(f"🔍 Validating {category} tickers...")
        for symbol in tickers:
            try:
                info = yf.Ticker(symbol).info
                if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    validated[category].append(symbol)
            except Exception:
                pass  # skip invalid/delisted tickers
        print(f"✅ {category.capitalize()} validation complete. {len(validated[category])} valid symbols found.")
    
    return validated


symbols = get_dynamic_symbols()
symbols = validate_symbols(symbols) 
LARGECAP = symbols["largecap"]
MIDCAP = symbols["midcap"]

# =====================================================
# TELEGRAM BOT HANDLERS
# =====================================================
async def greet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 LargeCap", callback_data="largecap")],
        [InlineKeyboardButton("📈 MidCap", callback_data="midcap")],
        [InlineKeyboardButton("💹 SingleStock", callback_data="single")]
    ]
    await update.message.reply_text(
        "🤖 Hello! I’m your AI Trading Bot.\nPlease select mode:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode = query.data
    chat_id = query.message.chat.id

    # Stop existing monitor if any
    if chat_id in user_monitors:
        old_monitor = user_monitors[chat_id]
        old_monitor["stop_flag"].set()
        print(f"[{chat_id}] Previous monitoring stopped before starting new one ({old_monitor['mode']})")
        context.bot.send_message(
            chat_id=chat_id,
            text=f"🛑 Monitoring stopped for {old_monitor['mode']} before starting new one."
        )
        del user_monitors[chat_id]

    # Prepare new stop flag
    stop_flag = threading.Event()

    if mode == "largecap":
        await query.edit_message_text(
            "✅ Running for Large Cap stocks...\nYou’ll receive anomaly alerts if detected.\nType 'stop' or /stop to halt."
        )
        t = threading.Thread(target=run_monitor, args=(chat_id, LARGECAP, context, "LargeCap"), daemon=True)
    elif mode == "midcap":
        await query.edit_message_text(
            "✅ Running for Mid Cap stocks...\nYou’ll receive anomaly alerts if detected.\nType 'stop' or /stop to halt."
        )
        t = threading.Thread(target=run_monitor, args=(chat_id, MIDCAP, context, "MidCap"), daemon=True)
    else:
        await query.edit_message_text("Please send the stock symbol (e.g. AAPL or RELIANCE.NS)")
        context.user_data["await_symbol"] = True
        return

    # Store new monitor
    user_monitors[chat_id] = {"thread": t, "stop_flag": stop_flag, "mode": mode}
    t.start()



async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat.id
    text = update.message.text.strip().lower()

    # Handle stop command
    if text in ["stop", "/stop"]:
        if chat_id in user_monitors:
            user_monitors[chat_id]["stop_flag"].set()
            del user_monitors[chat_id]
            await update.message.reply_text("🛑 Monitoring stopped.")
        else:
            await update.message.reply_text("❌ No active monitoring found.")
        return

    # Handle stock input
    if context.user_data.get("await_symbol"):
        symbol = text.upper()
        context.user_data["await_symbol"] = False
        stop_flag = threading.Event()
        user_monitors[chat_id] = {"stop_flag": stop_flag}
        await update.message.reply_text(f"✅ Running for {symbol}...\nYou’ll receive anomaly alerts if detected.\nType 'stop' or /stop to halt.")
        t = threading.Thread(target=run_monitor, args=(chat_id, [symbol], context), daemon=True)
        user_monitors[chat_id]["thread"] = t
        t.start()
        return

    # For any new user input (auto greet)
    await greet(update, context)

# =====================================================
# MAIN
# =====================================================
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in user_monitors:
        monitor = user_monitors[chat_id]
        monitor["stop_flag"].set()
        print(f"[{chat_id}] Stop command received — terminating {monitor['mode']} monitoring thread.")
        del user_monitors[chat_id]
        await update.message.reply_text("🛑 Monitoring stopped.")
    else:
        await update.message.reply_text("❌ No active monitoring found.")


def start_bot():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CallbackQueryHandler(handle_choice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_handler(CommandHandler("stop", stop_command))
    print("🤖 Telegram bot running...")
    app.run_polling()


if __name__ == "__main__":
    start_bot()
