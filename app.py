"""Gold Bot - single-file server (API + phone UI + trading engine).
Start: uvicorn app:app --host 0.0.0.0 --port $PORT      Env: BOT_TOKEN (required), BROKER=paper|metaapi
"""
from __future__ import annotations
"""Technical indicators (pure pandas/numpy, no TA-Lib needed)."""
import numpy as np
import pandas as pd


def ta_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def ta_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def ta_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def ta_macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    line = ta_ema(close, fast) - ta_ema(close, slow)
    return line - ta_ema(line, sig)


def ta_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = ta_atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def ta_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

# ---------------------------------------------------------------
"""Trading-session 'intelligence' card (all times UTC)."""
from datetime import datetime, timezone

# (name, start_hour, end_hour, liquidity, volatility, base quality score 0-100)
SESSIONS = [
    ("Asian Session", 0, 7, "Low", "Low", 35),
    ("London Session", 7, 12, "High", "High", 75),
    ("London/NY Overlap", 12, 16, "Very High", "High", 95),
    ("New York Session", 16, 21, "High", "Medium", 65),
    ("Late NY / Rollover", 21, 24, "Low", "Low", 20),
]


def current_session(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    h = now.hour + now.minute / 60
    for name, a, b, liq, vol, score in SESSIONS:
        if a <= h < b:
            mins_left = int((b - h) * 60)
            return {
                "name": name, "liquidity": liq, "volatility": vol, "score": score,
                "remaining": f"{mins_left // 60}h {mins_left % 60}m",
                "prime": score >= 75,
                "tradeable": score >= 60,
            }
    raise RuntimeError("unreachable")

# ---------------------------------------------------------------
"""Fundamental layer for gold.

Two things:
1. News blackout - skip trading around high-impact USD events (NFP, CPI, FOMC...).
2. Macro bias - gold moves inversely to the US dollar and to US yields.
   DXY below its 50-day EMA and falling 10Y yield = tailwind for gold.

Everything fails soft: if a data source is unreachable we return a neutral
bias (0) and no blackout, and say so in the 'notes'.
"""
from datetime import datetime, timedelta, timezone
import time
import requests

CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_cache: dict = {}


def _cached(key: str, ttl: int, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def high_impact_usd_events() -> list[dict]:
    def fetch():
        r = requests.get(CAL_URL, timeout=10, headers={"User-Agent": "goldbot/1.0"})
        r.raise_for_status()
        out = []
        for e in r.json():
            if e.get("country") == "USD" and e.get("impact") == "High":
                out.append({"title": e["title"], "time": datetime.fromisoformat(e["date"]).astimezone(timezone.utc)})
        return out
    try:
        return _cached("cal", 3600, fetch)
    except Exception:
        return []


def news_blackout(now: datetime | None = None, before_min: int = 30, after_min: int = 30,
                  events: list[dict] | None = None) -> tuple[bool, str]:
    now = now or datetime.now(timezone.utc)
    events = high_impact_usd_events() if events is None else events
    for e in events:
        if e["time"] - timedelta(minutes=before_min) <= now <= e["time"] + timedelta(minutes=after_min):
            return True, f"News blackout: {e['title']} at {e['time']:%H:%M} UTC"
    return False, ""


def _yahoo_daily(symbol: str, days: int = 120):
    import pandas as pd
    def fetch():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=6mo&interval=1d"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        return pd.Series(res["indicators"]["quote"][0]["close"], index=pd.to_datetime(res["timestamp"], unit="s")).dropna()
    return _cached(f"y:{symbol}", 1800, fetch)


def macro_bias() -> tuple[int, list[str]]:
    """Return (-2..+2, notes). Positive = bullish for gold."""
    score, notes = 0, []
    try:
        dxy = _yahoo_daily("DX-Y.NYB")
        e50 = dxy.ewm(span=50, adjust=False).mean()
        if dxy.iloc[-1] < e50.iloc[-1] and dxy.iloc[-1] < dxy.iloc[-6]:
            score += 1; notes.append("USD weak (DXY < EMA50 & falling) -> gold tailwind")
        elif dxy.iloc[-1] > e50.iloc[-1] and dxy.iloc[-1] > dxy.iloc[-6]:
            score -= 1; notes.append("USD strong (DXY > EMA50 & rising) -> gold headwind")
        else:
            notes.append("USD neutral")
    except Exception:
        notes.append("DXY unavailable (neutral)")
    try:
        y = _yahoo_daily("^TNX")
        if y.iloc[-1] < y.iloc[-6] - 0.05:
            score += 1; notes.append("US10Y yields falling -> gold tailwind")
        elif y.iloc[-1] > y.iloc[-6] + 0.05:
            score -= 1; notes.append("US10Y yields rising -> gold headwind")
        else:
            notes.append("US10Y flat")
    except Exception:
        notes.append("US10Y unavailable (neutral)")
    return score, notes

# ---------------------------------------------------------------
"""Gold strategy: H1 trend filter + M15 pullback-resumption entry, gated by fundamentals.

Technical score (-5..+5) is combined with the fundamental bias (-2..+2).
A trade needs: technical score >= MIN_TECH in one direction AND fundamentals not opposing
AND no news blackout AND a tradeable session.
"""
from dataclasses import dataclass, field
import pandas as pd

MIN_TECH = 3
SL_ATR = 1.5
TP_ATR = 2.5   # ~1.67R


@dataclass
class Signal:
    action: str = "HOLD"          # BUY / SELL / HOLD
    price: float = 0.0
    sl: float | None = None
    tp: float | None = None
    tech_score: int = 0
    fund_score: int = 0
    reasons: list[str] = field(default_factory=list)
    confidence: int = 0           # 0-100 for the UI


def technical_score(m15: pd.DataFrame) -> tuple[int, list[str], float, float]:
    """m15: OHLC DataFrame with DatetimeIndex, >= 250 rows. Returns (score, reasons, price, atr)."""
    h1 = ta_resample(m15, "1h")
    c = m15["close"]
    reasons: list[str] = []
    score = 0

    # 1) H1 trend (+-2)
    e50, e200 = ta_ema(h1["close"], 50).iloc[-1], ta_ema(h1["close"], 200).iloc[-1]
    if h1["close"].iloc[-1] > e50 > e200:
        score += 2; reasons.append("H1 uptrend (price > EMA50 > EMA200)")
    elif h1["close"].iloc[-1] < e50 < e200:
        score -= 2; reasons.append("H1 downtrend (price < EMA50 < EMA200)")
    else:
        reasons.append("H1 trend unclear")

    # 2) Trend strength: ADX gate
    adx = ta_adx(m15).iloc[-1]
    if adx < 18:
        reasons.append(f"ADX {adx:.0f} - market ranging, no trend trades")
        return 0, reasons, float(c.iloc[-1]), float(ta_atr(m15).iloc[-1])

    # 3) M15 pullback resumption (+-2): was beyond EMA20 recently, now closed back through it
    e20 = ta_ema(c, 20)
    was_below = (c.iloc[-4:-1] < e20.iloc[-4:-1]).any()
    was_above = (c.iloc[-4:-1] > e20.iloc[-4:-1]).any()
    if was_below and c.iloc[-1] > e20.iloc[-1]:
        score += 2; reasons.append("M15 pullback resumed upward through EMA20")
    elif was_above and c.iloc[-1] < e20.iloc[-1]:
        score -= 2; reasons.append("M15 pullback resumed downward through EMA20")

    # 4) Momentum (+-1): MACD histogram sign + RSI not exhausted
    mh = ta_macd_hist(c)
    r = ta_rsi(c).iloc[-1]
    if mh.iloc[-1] > 0 and mh.iloc[-1] > mh.iloc[-2] and r < 70:
        score += 1; reasons.append(f"Momentum up (MACD rising, RSI {r:.0f})")
    elif mh.iloc[-1] < 0 and mh.iloc[-1] < mh.iloc[-2] and r > 30:
        score -= 1; reasons.append(f"Momentum down (MACD falling, RSI {r:.0f})")

    return score, reasons, float(c.iloc[-1]), float(ta_atr(m15).iloc[-1])


def build_signal(m15: pd.DataFrame, fund_score: int = 0, fund_notes: list[str] | None = None,
                 blackout: tuple[bool, str] = (False, ""), session_ok: bool = True) -> Signal:
    tech, reasons, price, atr = technical_score(m15)
    sig = Signal(price=price, tech_score=tech, fund_score=fund_score, reasons=reasons + (fund_notes or []))
    sig.confidence = min(100, int(abs(tech) / 5 * 70 + max(0, fund_score * (1 if tech > 0 else -1)) * 15))

    if blackout[0]:
        sig.reasons.append(blackout[1]); return sig
    if not session_ok:
        sig.reasons.append("Session liquidity too low - waiting"); return sig

    if tech >= MIN_TECH and fund_score >= 0:
        sig.action, sig.sl, sig.tp = "BUY", price - SL_ATR * atr, price + TP_ATR * atr
    elif tech <= -MIN_TECH and fund_score <= 0:
        sig.action, sig.sl, sig.tp = "SELL", price + SL_ATR * atr, price - TP_ATR * atr
    elif abs(tech) >= MIN_TECH:
        sig.reasons.append("Technicals and fundamentals disagree - skipped")
    return sig

# ---------------------------------------------------------------
"""Risk controls. These matter more than the entry logic."""
from dataclasses import dataclass


@dataclass
class RiskConfig:
    lot_size: float = 0.01
    max_open_trades: int = 1          # the 'Trades to open' selector in the app
    max_daily_loss_pct: float = 3.0   # stop for the day after losing this % of balance
    max_spread: float = 0.60          # skip if spread (USD) wider than this
    cooldown_min: int = 30            # min minutes between entries


def can_trade(cfg: RiskConfig, open_trades: int, day_pnl: float, balance: float,
              spread: float, minutes_since_last: float) -> tuple[bool, str]:
    if open_trades >= cfg.max_open_trades:
        return False, f"Max open trades reached ({open_trades}/{cfg.max_open_trades})"
    if balance > 0 and day_pnl <= -balance * cfg.max_daily_loss_pct / 100:
        return False, f"Daily loss limit hit ({cfg.max_daily_loss_pct}%) - trading paused until tomorrow"
    if spread > cfg.max_spread:
        return False, f"Spread too wide ({spread:.2f})"
    if minutes_since_last < cfg.cooldown_min:
        return False, f"Cooldown ({int(cfg.cooldown_min - minutes_since_last)}m left)"
    return True, "ok"

# ---------------------------------------------------------------
"""Market data providers."""
import pandas as pd
import requests


def yahoo_m15(symbol: str = "GC=F") -> pd.DataFrame:
    """Free 15-minute gold futures candles (good for paper trading / testing)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=15m"
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True)).dropna()
    return df

# ---------------------------------------------------------------
"""Broker adapters. PaperBroker is the default; MetaApiBroker talks to a real MT4/MT5 account."""
from datetime import datetime, timezone, timedelta
import os
import pandas as pd
import requests

GOLD_CONTRACT = 100  # 1.00 lot XAUUSD = 100 oz -> $1 move = $100 per lot


class PaperBroker:
    """Simulated account. Uses live Yahoo gold prices, fills at market with a fixed spread."""
    name = "PAPER"

    def __init__(self, balance: float = 1000.0, spread: float = 0.30):
        self.balance, self.start_balance, self.spread = balance, balance, spread
        self.positions: list[dict] = []
        self.closed: list[dict] = []
        self._last: pd.DataFrame | None = None

    def candles(self) -> pd.DataFrame:
        self._last = yahoo_m15()
        return self._last

    def quote(self) -> tuple[float, float]:
        if self._last is None:
            self.candles()
        mid = float(self._last["close"].iloc[-1])
        return mid - self.spread / 2, mid + self.spread / 2

    def spread_now(self) -> float:
        return self.spread

    def place(self, action: str, lot: float, sl: float, tp: float) -> dict:
        bid, ask = self.quote()
        pos = {"id": len(self.closed) + len(self.positions) + 1, "side": action, "lot": lot,
               "entry": ask if action == "BUY" else bid, "sl": sl, "tp": tp,
               "opened": datetime.now(timezone.utc).isoformat()}
        self.positions.append(pos)
        return pos

    def update(self):
        """Check open positions against latest candle range for SL/TP hits."""
        if self._last is None or not self.positions:
            return
        hi, lo = float(self._last["high"].iloc[-1]), float(self._last["low"].iloc[-1])
        for p in list(self.positions):
            exit_px = None
            if p["side"] == "BUY":
                if lo <= p["sl"]: exit_px = p["sl"]
                elif hi >= p["tp"]: exit_px = p["tp"]
            else:
                if hi >= p["sl"]: exit_px = p["sl"]
                elif lo <= p["tp"]: exit_px = p["tp"]
            if exit_px is not None:
                d = (exit_px - p["entry"]) * (1 if p["side"] == "BUY" else -1)
                p["pnl"] = d * p["lot"] * GOLD_CONTRACT
                p["closed"] = datetime.now(timezone.utc).isoformat()
                self.balance += p["pnl"]
                self.positions.remove(p); self.closed.append(p)

    def open_count(self) -> int:
        return len(self.positions)

    def day_pnl(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        return sum(p["pnl"] for p in self.closed if p["closed"][:10] == today)

    def get_balance(self) -> float:
        return self.balance

    def floating(self) -> float:
        bid, ask = self.quote()
        return sum(((bid if p["side"] == "BUY" else ask) - p["entry"]) * (1 if p["side"] == "BUY" else -1)
                   * p["lot"] * GOLD_CONTRACT for p in self.positions)

    def close_all(self):
        bid, ask = self.quote()
        for p in list(self.positions):
            px = bid if p["side"] == "BUY" else ask
            p["pnl"] = (px - p["entry"]) * (1 if p["side"] == "BUY" else -1) * p["lot"] * GOLD_CONTRACT
            p["closed"] = datetime.now(timezone.utc).isoformat()
            self.balance += p["pnl"]; self.positions.remove(p); self.closed.append(p)


class MetaApiBroker:
    """Real MT4/MT5 execution through MetaApi cloud (https://metaapi.cloud).
    Your broker account must be deployed in MetaApi; you need an API token and account id.
    Endpoints follow MetaApi's REST docs - verify against the current docs if a call 404s."""
    name = "METAAPI"

    def __init__(self, token: str, account_id: str, symbol: str = "XAUUSD", region: str = "london"):
        self.h = {"auth-token": token, "Content-Type": "application/json"}
        self.symbol = symbol
        self.base = f"https://mt-client-api-v1.{region}.agiliumtrade.ai/users/current/accounts/{account_id}"
        self.mdata = f"https://mt-market-data-client-api-v1.{region}.agiliumtrade.ai/users/current/accounts/{account_id}"
        self._spread = 0.3

    def _get(self, url, **kw):
        r = requests.get(url, headers=self.h, timeout=20, **kw); r.raise_for_status(); return r.json()

    def candles(self) -> pd.DataFrame:
        rows = self._get(f"{self.mdata}/historical-market-data/symbols/{self.symbol}/timeframes/15m/candles",
                         params={"limit": 1000})
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["time"], utc=True)
        return df[["open", "high", "low", "close"]].sort_index()

    def quote(self) -> tuple[float, float]:
        p = self._get(f"{self.base}/symbols/{self.symbol}/current-price")
        self._spread = p["ask"] - p["bid"]
        return p["bid"], p["ask"]

    def spread_now(self) -> float:
        self.quote(); return self._spread

    def place(self, action: str, lot: float, sl: float, tp: float) -> dict:
        body = {"actionType": "ORDER_TYPE_BUY" if action == "BUY" else "ORDER_TYPE_SELL",
                "symbol": self.symbol, "volume": lot, "stopLoss": round(sl, 2), "takeProfit": round(tp, 2),
                "comment": "goldbot"}
        r = requests.post(f"{self.base}/trade", headers=self.h, json=body, timeout=30)
        r.raise_for_status(); return r.json()

    def update(self): pass

    def open_count(self) -> int:
        return len([p for p in self._get(f"{self.base}/positions") if p.get("symbol") == self.symbol])

    def get_balance(self) -> float:
        return float(self._get(f"{self.base}/account-information")["balance"])

    def day_pnl(self) -> float:
        now = datetime.now(timezone.utc); start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = self._get(f"{self.base}/history-deals/time/{start.isoformat()}/{(now + timedelta(minutes=1)).isoformat()}")
        return sum(d.get("profit", 0) + d.get("commission", 0) + d.get("swap", 0) for d in deals)

    def floating(self) -> float:
        return sum(p.get("profit", 0) for p in self._get(f"{self.base}/positions"))

    def close_all(self):
        for p in self._get(f"{self.base}/positions"):
            requests.post(f"{self.base}/trade", headers=self.h, timeout=30,
                          json={"actionType": "POSITION_CLOSE_ID", "positionId": p["id"]})


def make_broker():
    if os.getenv("BROKER", "paper").lower() == "metaapi":
        return MetaApiBroker(os.environ["METAAPI_TOKEN"], os.environ["METAAPI_ACCOUNT_ID"],
                             os.getenv("SYMBOL", "XAUUSD"), os.getenv("METAAPI_REGION", "london"))
    return PaperBroker(float(os.getenv("PAPER_BALANCE", "1000")))

# ---------------------------------------------------------------
"""The trading loop: each tick = data -> fundamentals -> signal -> risk -> order."""
import asyncio, os, time
from collections import deque
from datetime import datetime, timezone

TICK_SECONDS = int(os.getenv("TICK_SECONDS", "60"))


class Engine:
    def __init__(self):
        self.broker = make_broker()
        self.cfg = RiskConfig()
        self.running = False
        self.log: deque = deque(maxlen=100)
        self.last_signal: dict = {}
        self.last_entry_ts = 0.0
        self.macro = (0, [])
        self.macro_ts = 0.0
        self._task: asyncio.Task | None = None

    def say(self, msg: str):
        self.log.appendleft({"t": datetime.now(timezone.utc).strftime("%H:%M:%S"), "msg": msg})

    def tick(self):
        df = self.broker.candles()
        self.broker.update()
        if time.time() - self.macro_ts > 1800:
            self.macro, self.macro_ts = macro_bias(), time.time()
        fscore, fnotes = self.macro
        sess = current_session()
        sig = build_signal(df, fscore, fnotes, news_blackout(), sess["tradeable"])
        self.last_signal = {"action": sig.action, "price": sig.price, "sl": sig.sl, "tp": sig.tp,
                            "tech": sig.tech_score, "fund": sig.fund_score, "confidence": sig.confidence,
                            "reasons": sig.reasons}
        if not self.running or sig.action == "HOLD":
            return
        ok, why = can_trade(self.cfg, self.broker.open_count(), self.broker.day_pnl(),
                            self.broker.get_balance(), self.broker.spread_now(),
                            (time.time() - self.last_entry_ts) / 60)
        if not ok:
            self.say(f"{sig.action} signal skipped: {why}"); return
        self.broker.place(sig.action, self.cfg.lot_size, sig.sl, sig.tp)
        self.last_entry_ts = time.time()
        self.say(f"{sig.action} {self.cfg.lot_size} lot @ {sig.price:.2f} SL {sig.sl:.2f} TP {sig.tp:.2f}")

    async def loop(self):
        while True:
            try:
                await asyncio.to_thread(self.tick)
            except Exception as e:  # never let the loop die
                self.say(f"Error: {e}")
            await asyncio.sleep(TICK_SECONDS)

    def ensure_loop(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.loop())

    def start(self): self.running = True; self.say("Bot STARTED"); self.ensure_loop()
    def stop(self): self.running = False; self.say("Bot STOPPED (open trades keep their SL/TP)")

    def status(self) -> dict:
        try:
            bid, ask = self.broker.quote()
        except Exception:
            bid = ask = None
        return {"running": self.running, "broker": self.broker.name, "session": current_session(),
                "signal": self.last_signal, "bid": bid, "ask": ask,
                "balance": self.broker.get_balance(), "open": self.broker.open_count(),
                "floating": self.broker.floating(), "day_pnl": self.broker.day_pnl(),
                "lot": self.cfg.lot_size, "max_trades": self.cfg.max_open_trades,
                "blackout": news_blackout()[1], "log": list(self.log)[:25]}

INDEX_HTML = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0a0709">
<title>Gold Bot</title>
<style>
:root{--bg:#0a0709;--panel:#120a0c;--red:#ff2a2a;--redd:#7a1111;--amber:#ffb400;--ok:#25e08a;--tx:#f3e9e9;--mut:#a08a8a}
*{box-sizing:border-box;margin:0}
body{background:radial-gradient(900px 500px at 50% -10%,#2a0b0e,var(--bg));color:var(--tx);font:14px/1.4 system-ui,sans-serif;min-height:100dvh;padding:12px 14px 90px;max-width:520px;margin:auto}
.card{background:var(--panel);border:1px solid var(--redd);border-radius:18px;box-shadow:0 0 18px #ff2a2a22,inset 0 0 14px #ff2a2a14;padding:14px;margin-bottom:12px}
.top{display:flex;justify-content:space-between;align-items:center;font-weight:800;letter-spacing:.12em;font-size:12px}
.pill{background:var(--amber);color:#1b1100;border:0;border-radius:99px;padding:5px 14px;font-weight:800;font-size:11px}
.hero{text-align:center;padding:26px 10px 8px}
.hero h1{font-size:26px;letter-spacing:.06em;text-shadow:0 0 14px var(--red)}
.hero .gold{color:var(--amber)}
.mode{display:inline-block;margin-top:6px;font-size:10px;letter-spacing:.14em;padding:3px 10px;border-radius:99px;border:1px solid var(--redd);color:var(--mut)}
.mode.live{border-color:var(--red);color:var(--red)}
.acts{display:flex;justify-content:space-around;margin-top:18px}
.acts button{background:none;border:0;color:var(--tx);font-size:10px;letter-spacing:.12em;font-weight:700;display:flex;flex-direction:column;align-items:center;gap:4px}
.acts i{font-style:normal;font-size:22px;color:var(--amber)}
.acts .on i{color:var(--ok)}
.lbl{font-size:10px;letter-spacing:.14em;color:var(--red);font-weight:800;margin-bottom:8px}
.row{display:flex;gap:12px;align-items:center}
.ring{width:62px;height:62px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:18px;border:4px solid var(--red)}
.k{font-size:10px;color:var(--mut)}.v{font-weight:700}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.prime{border:1px solid var(--redd);border-radius:12px;padding:10px;margin-top:10px;font-size:12px}
.prime b{color:var(--amber)}
.big{font-size:30px;font-weight:800}.buy{color:var(--ok)}.sell{color:var(--red)}.hold{color:var(--mut)}
.bar{height:6px;background:#2a1a1c;border-radius:9px;overflow:hidden;margin-top:6px}.bar>div{height:100%;background:linear-gradient(90deg,var(--redd),var(--red))}
ul{padding-left:18px;font-size:12px;color:#d9c9c9}li{margin:3px 0}
.log div{font-size:12px;border-bottom:1px solid #2a1a1c;padding:5px 0}.log span{color:var(--mut);margin-right:8px;font-family:monospace}
nav{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);width:min(492px,calc(100% - 28px));display:flex;background:#0d0607;border:1px solid var(--red);border-radius:20px;padding:6px}
nav button{flex:1;background:none;border:0;color:var(--mut);font-size:10px;letter-spacing:.1em;font-weight:700;padding:9px 0;border-radius:14px}
nav button.on{color:var(--tx);background:#2a0d10;box-shadow:inset 0 0 10px #ff2a2a44}
.tab{display:none}.tab.on{display:block}
input[type=range]{width:100%;accent-color:var(--red)}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 14px}
.chips button{background:#1a0c0e;border:1px solid var(--redd);color:var(--tx);border-radius:9px;padding:7px 12px;font-weight:700}
.chips button.on{background:var(--red);color:#fff}
.warn{font-size:11px;color:var(--mut);margin-top:6px}
input[type=password]{width:100%;padding:12px;background:#1a0c0e;color:var(--tx);border:1px solid var(--redd);border-radius:10px;margin:10px 0}
.cta{width:100%;background:linear-gradient(#ffc933,#ffa400);border:0;border-radius:12px;padding:13px;font-weight:800;color:#1b1100}
#err{color:var(--red);font-size:12px;margin-top:6px}
</style></head><body>

<div id="login" class="card" style="display:none;margin-top:30vh">
  <div class="lbl">ACCESS</div>Enter the BOT_TOKEN you set on your server.
  <input id="tok" type="password" placeholder="Token"><button class="cta" onclick="saveTok()">CONNECT</button><div id="err"></div>
</div>

<div id="app" style="display:none">
<div class="card top"><span>● GOLD BOT · XAUUSD</span><button class="pill" onclick="show('set')">MENU</button></div>

<section id="home" class="tab on">
  <div class="card hero">
    <h1>GOLD <span class="gold">BOT</span></h1><div class="mode" id="mode">PAPER</div>
    <div class="acts">
      <button id="bStart" onclick="toggle()"><i>▶</i><span id="bStartT">START</span></button>
      <button onclick="show('sig')"><i>〰</i>SIGNAL</button>
      <button onclick="remove()"><i>🗑</i>CLOSE ALL</button>
    </div>
  </div>
  <div class="card"><div class="lbl">SESSION INTELLIGENCE</div>
    <div class="row"><div class="ring" id="ring">--</div>
      <div style="flex:1"><div class="v" id="sname" style="color:var(--amber)">—</div><div class="k" id="sleft"></div>
      <div class="grid" style="margin-top:6px"><div><div class="k">LIQUIDITY</div><div class="v" id="liq">—</div></div><div><div class="k">VOLATILITY</div><div class="v" id="vol">—</div></div></div></div></div>
    <div class="prime" id="prime"></div>
  </div>
  <div class="card"><div class="lbl">ACCOUNT</div>
    <div class="grid"><div><div class="k">BALANCE</div><div class="v" id="bal">—</div></div><div><div class="k">FLOATING P/L</div><div class="v" id="flt">—</div></div>
    <div><div class="k">TODAY</div><div class="v" id="day">—</div></div><div><div class="k">OPEN TRADES</div><div class="v" id="open">—</div></div></div>
    <div class="warn" id="quotes"></div>
  </div>
</section>

<section id="sig" class="tab">
  <div class="card"><div class="lbl">CURRENT SIGNAL</div>
    <div class="big hold" id="act">HOLD</div>
    <div class="k" style="margin-top:6px">CONFIDENCE</div><div class="bar"><div id="conf" style="width:0"></div></div>
    <div class="grid" style="margin-top:12px"><div><div class="k">TECHNICAL (-5..+5)</div><div class="v" id="tech">0</div></div><div><div class="k">FUNDAMENTAL (-2..+2)</div><div class="v" id="fund">0</div></div>
    <div><div class="k">STOP LOSS</div><div class="v" id="sl">—</div></div><div><div class="k">TAKE PROFIT</div><div class="v" id="tp">—</div></div></div>
  </div>
  <div class="card"><div class="lbl">WHY</div><ul id="why"></ul></div>
</section>

<section id="logs" class="tab"><div class="card log"><div class="lbl">ACTIVITY</div><div id="log"></div></div></section>

<section id="set" class="tab"><div class="card"><div class="lbl">LOT SIZE <span id="lotv" style="float:right;color:var(--amber)">0.01</span></div>
  <input type="range" id="lot" min="0.01" max="1" step="0.01" value="0.01" oninput="lotv.textContent=(+this.value).toFixed(2)" onchange="save()">
  <div class="lbl" style="margin-top:14px">TRADES TO OPEN</div><div class="chips" id="chips"></div>
  <div class="warn">Start with 0.01 lot on a demo account. Daily loss limit (3%) and news blackout are always on.</div>
  <button class="cta" style="margin-top:14px" onclick="resetTok()">CHANGE TOKEN</button></div></section>
</div>

<nav id="nav"><button class="on" onclick="show('home')">HOME</button><button onclick="show('sig')">SIGNAL</button><button onclick="show('logs')">LOG</button><button onclick="show('set')">SETTINGS</button></nav>

<script>
const $=id=>document.getElementById(id);
let TOK=''; try{TOK=localStorage.getItem('tok')||''}catch(e){}
let running=false;
function show(id){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id===id));
  const m={home:0,sig:1,logs:2,set:3};document.querySelectorAll('nav button').forEach((b,i)=>b.classList.toggle('on',i===m[id]))}
async function api(p,method='GET',body){
  const r=await fetch('/api/'+p,{method,headers:{'Authorization':'Bearer '+TOK,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  if(r.status===401){login();throw new Error('auth')} if(!r.ok){alert((await r.json()).detail||'Error');throw new Error('api')} return r.json()}
function login(){$('app').style.display='none';$('nav').style.display='none';$('login').style.display='block'}
function saveTok(){TOK=$('tok').value.trim();try{localStorage.setItem('tok',TOK)}catch(e){}$('login').style.display='none';boot()}
function resetTok(){try{localStorage.removeItem('tok')}catch(e){}login()}
const money=n=>(n<0?'-':'')+'$'+Math.abs(n).toFixed(2);
function render(s){
  running=s.running;$('bStartT').textContent=running?'STOP':'START';$('bStart').classList.toggle('on',running);
  $('mode').textContent=s.broker==='PAPER'?'PAPER (DEMO) MODE':'LIVE BROKER LINK';$('mode').classList.toggle('live',s.broker!=='PAPER');
  const se=s.session;$('ring').textContent=se.score;$('sname').textContent=se.name;$('sleft').textContent=se.remaining+' remaining';
  $('liq').textContent=se.liquidity;$('vol').textContent=se.volatility;
  $('prime').innerHTML=s.blackout?`<b>⚠ NEWS WINDOW</b> ${s.blackout}`:se.prime?`<b>PRIME WINDOW</b> ${se.name} — best conditions of the day.`:se.tradeable?`<b>ACTIVE</b> Conditions acceptable.`:`<b>QUIET</b> Low liquidity — bot waits.`;
  $('bal').textContent=money(s.balance);$('flt').textContent=money(s.floating);$('day').textContent=money(s.day_pnl);$('open').textContent=s.open+' / '+s.max_trades;
  $('quotes').textContent=s.bid?`Bid ${s.bid.toFixed(2)}  ·  Ask ${s.ask.toFixed(2)}`:'Quote unavailable';
  const g=s.signal||{};const a=g.action||'HOLD';$('act').textContent=a;$('act').className='big '+a.toLowerCase();
  $('conf').style.width=(g.confidence||0)+'%';$('tech').textContent=g.tech??0;$('fund').textContent=g.fund??0;
  $('sl').textContent=g.sl?g.sl.toFixed(2):'—';$('tp').textContent=g.tp?g.tp.toFixed(2):'—';
  $('why').innerHTML=(g.reasons||[]).map(r=>`<li>${r}</li>`).join('')||'<li>Waiting for first analysis…</li>';
  $('log').innerHTML=s.log.map(l=>`<div><span>${l.t}</span>${l.msg}</div>`).join('')||'<div>No activity yet.</div>';
  if(document.activeElement!==$('lot')){$('lot').value=s.lot;$('lotv').textContent=s.lot.toFixed(2)}
  $('chips').innerHTML=[1,2,3,4,5].map(n=>`<button class="${n===s.max_trades?'on':''}" onclick="save(${n})">${n}</button>`).join('');
}
async function refresh(){try{render(await api('status'))}catch(e){}}
async function toggle(){render(await api(running?'stop':'start','POST'))}
async function remove(){if(confirm('Stop the bot and close ALL open trades?'))render(await api('remove','POST'))}
async function save(n){const b={lot_size:+$('lot').value};if(n)b.max_open_trades=n;render(await api('settings','POST',b))}
function boot(){if(!TOK)return login();$('app').style.display='block';$('nav').style.display='flex';refresh();clearInterval(window.t);window.t=setInterval(refresh,5000)}
boot();

</script></body></html>
'''

"""API + serves the phone app. Run: uvicorn server:app --host 0.0.0.0 --port 8000"""
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

TOKEN = os.getenv("BOT_TOKEN", "")
engine = Engine()


@asynccontextmanager
async def lifespan(app):
    engine.ensure_loop()   # analysis runs always; orders only when STARTED
    yield


app = FastAPI(lifespan=lifespan)


def auth(authorization: str = Header(default="")):
    if not TOKEN or authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "Bad or missing token (set BOT_TOKEN on the server)")


class Settings(BaseModel):
    lot_size: float | None = None
    max_open_trades: int | None = None


@app.get("/api/status", dependencies=[Depends(auth)])
def status(): return engine.status()

@app.post("/api/start", dependencies=[Depends(auth)])
def start(): engine.start(); return engine.status()

@app.post("/api/stop", dependencies=[Depends(auth)])
def stop(): engine.stop(); return engine.status()

@app.post("/api/remove", dependencies=[Depends(auth)])
def remove():
    engine.stop(); engine.broker.close_all(); engine.say("All trades closed"); return engine.status()

@app.post("/api/settings", dependencies=[Depends(auth)])
def settings(s: Settings):
    if s.lot_size is not None:
        if not 0.01 <= s.lot_size <= 1.0: raise HTTPException(400, "lot_size must be 0.01-1.00")
        engine.cfg.lot_size = s.lot_size
    if s.max_open_trades is not None:
        if not 1 <= s.max_open_trades <= 10: raise HTTPException(400, "trades must be 1-10")
        engine.cfg.max_open_trades = s.max_open_trades
    return engine.status()

@app.get("/", response_class=HTMLResponse)
def index(): return INDEX_HTML