# -*- coding: utf-8 -*-
"""
بوت فلترة أسهم التقسيم العكسي للشورت  —  مبني على طريقة @Naifo9x
------------------------------------------------------------------
التشغيل:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run split_short_bot.py

مصادر البيانات:
    - الأسعار والتقسيمات: Yahoo Finance (yfinance)
    - رسوم الاقتراض والأسهم المتاحة للشورت: ملف Interactive Brokers العام
      ftp3.interactivebrokers.com/usa.txt  (نفس مصدر chartexchange - يتحدث كل ~15 دقيقة)

تنبيه: أداة تعليمية للفلترة، مو توصية شراء أو بيع.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ======================================================================
# ===== CORE (تحليل صافي بدون إنترنت — قابل للاختبار) ===================
# ======================================================================


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def macd(s: pd.Series, f: int = 12, sl: int = 26, sig: int = 9):
    line = ema(s, f) - ema(s, sl)
    signal = ema(line, sig)
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    for n in (20, 30, 50):
        df[f"EMA{n}"] = ema(c, n)
    df["RSI"] = rsi(c)
    df["MACD"], df["SIGNAL"], df["HIST"] = macd(c)
    df["ATR"] = atr(df)
    return df


def to_4h(h1: pd.DataFrame) -> pd.DataFrame:
    """تحويل فريم الساعة إلى 4 ساعات (شمعتين باليوم: 9:30 و 13:30 بتوقيت نيويورك)."""
    if h1 is None or h1.empty:
        return pd.DataFrame()
    g = h1.resample("4h", offset="1h30min")
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
        "Close": g["Close"].last(), "Volume": g["Volume"].sum(),
    }).dropna(subset=["Close"])
    return out


def pivots(df: pd.DataFrame, k: int = 3):
    """قمم وقيعان محلية (Swing highs / lows)."""
    hi, lo = df["High"].values, df["Low"].values
    highs, lows = [], []
    for i in range(k, len(df) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            highs.append(hi[i])
        if lo[i] == lo[i - k:i + k + 1].min():
            lows.append(lo[i])
    return highs, lows


def cluster_levels(levels, tol=0.03):
    """دمج المستويات القريبة (±3%) — يرجع [(السعر، عدد اللمسات)]."""
    out = []
    for p in sorted(levels):
        if out and abs(p - out[-1][0]) / out[-1][0] <= tol:
            lv, n = out[-1]
            out[-1] = ((lv * n + p) / (n + 1), n + 1)
        else:
            out.append((p, 1))
    return out


def support_resistance(daily: pd.DataFrame, h4: pd.DataFrame, price: float):
    frames = [daily.tail(120)]
    if h4 is not None and not h4.empty:
        frames.append(h4.tail(150))
    hs, ls = [], []
    for f in frames:
        h, l = pivots(f)
        hs += h
        ls += l
    lv = cluster_levels(hs + ls)
    sup = sorted([x for x in lv if x[0] < price * 0.995], key=lambda x: -x[0])
    res = sorted([x for x in lv if x[0] > price * 1.005], key=lambda x: x[0])
    return sup[:3], res[:3]


def reverse_splits(splits: pd.Series) -> list[dict]:
    """splits من yfinance: القيمة < 1 تعني تقسيم عكسي (مثلاً 0.1 = 1:10)."""
    out = []
    if splits is None or len(splits) == 0:
        return out
    for d, r in splits.items():
        if r and r < 1:
            out.append({"date": pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d),
                        "ratio": f"1:{round(1 / r):g}"})
    return sorted(out, key=lambda x: x["date"], reverse=True)


def pump_events(daily: pd.DataFrame, thr: float = 0.40, look: int = 252):
    """أيام انفجار (+40% من إغلاق أمس إلى أعلى سعر) وكم رجع السهم بعدها خلال 10 أيام."""
    d = daily.tail(look)
    pc = d["Close"].shift()
    spike = (d["High"] / pc - 1) >= thr
    ev = []
    idx = list(d.index)
    for t in d.index[spike.fillna(False)]:
        i = idx.index(t)
        after = d["Close"].iloc[i:i + 11]
        peak = d["High"].iloc[i]
        fade = (after.min() / peak - 1) if len(after) else np.nan
        ev.append({"date": t, "gain": peak / pc.loc[t] - 1, "fade10": fade})
    return ev


@dataclass
class Result:
    ticker: str
    price: float = math.nan
    score: int = 0
    verdict: str = ""
    checks: list = field(default_factory=list)   # (اسم البند، حالة True/False/None، شرح)
    flags: list = field(default_factory=list)    # تحذيرات
    levels: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    error: str = ""


def analyze(ticker: str, daily: pd.DataFrame, h1: pd.DataFrame | None, splits: pd.Series | None,
            borrow: dict | None, float_shares: float | None = None, today: pd.Timestamp | None = None) -> Result:
    R = Result(ticker)
    if daily is None or len(daily) < 30:
        R.error = "بيانات الأسعار غير كافية"
        return R
    daily = daily.copy()
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    D = add_indicators(daily)
    H = add_indicators(to_4h(h1)) if h1 is not None and len(h1) > 40 else pd.DataFrame()
    last = D.iloc[-1]
    price = float(last["Close"])
    R.price = price
    today = today or D.index[-1]
    score = 0
    m = R.metrics

    # ---------- 1) التقسيم العكسي ----------
    rs = reverse_splits(splits)
    if rs:
        days = (today - rs[0]["date"]).days
        m["آخر تقسيم عكسي"] = f'{rs[0]["ratio"]} — {rs[0]["date"]:%Y-%m-%d} (قبل {days} يوم)'
        recent3y = [x for x in rs if (today - x["date"]).days <= 365 * 3]
        m["عدد التقسيمات العكسية (3 سنوات)"] = len(recent3y)
        pre = D[D.index < rs[0]["date"]]
        if len(pre):
            chg = price / float(pre["Close"].iloc[-1]) - 1
            m["التغير منذ التقسيم"] = f"{chg:+.1%}"
        ok = days <= 365
        score += 20 if days <= 180 else (10 if ok else 0)
        if len(recent3y) >= 2:
            score += 5
            R.flags.append(f"سهم متكرر التقسيم العكسي ({len(recent3y)} مرات خلال 3 سنوات) — شركة تعتمد على الطرح والتخفيف.")
        R.checks.append(("التقسيم العكسي", ok, m["آخر تقسيم عكسي"]))
    else:
        R.checks.append(("التقسيم العكسي", False, "ما فيه تقسيم عكسي مسجل"))

    # ---------- 2) الشورت (IBKR) ----------
    can_short = None
    if borrow:
        avail, fee = borrow.get("available"), borrow.get("fee")
        m["متاح للشورت (IBKR)"] = f"{avail:,}" if avail is not None else "—"
        m["رسوم الاقتراض السنوية"] = f"{fee:.2f}%" if fee is not None else "—"
        if fee is not None:
            m["تكلفة يومية تقريبية"] = f"{fee / 360:.2f}% يومياً"
        can_short = bool(avail and avail > 0)
        if not can_short:
            R.flags.append("ما فيه أسهم متاحة للاقتراض في IBKR — الشورت غير ممكن حالياً.")
        else:
            score += 15 if fee < 30 else (10 if fee <= 100 else 5)
            if fee > 100:
                R.flags.append(f"رسوم الاقتراض {fee:.0f}% سنوياً ≈ {fee/360:.2f}% كل يوم — لا تطوّل الصفقة.")
            if avail < 10000:
                R.flags.append(f"الأسهم المتاحة قليلة ({avail:,}) — احتمال استدعاء الأسهم (Buy-in) وارتفاع الرسوم.")
        R.checks.append(("الشورت متاح", can_short, f'{m["متاح للشورت (IBKR)"]} سهم، رسوم {m["رسوم الاقتراض السنوية"]}'))
    else:
        R.checks.append(("الشورت متاح", None, "ما قدرت أجيب بيانات IBKR لهذا الرمز"))

    # ---------- 3) المؤشرات ----------
    bear_d = price < last["EMA20"] < last["EMA50"]
    m["يومي: EMA20/30/50"] = f'{last["EMA20"]:.2f} / {last["EMA30"]:.2f} / {last["EMA50"]:.2f}'
    m["يومي: RSI"] = round(float(last["RSI"]), 1)
    m["يومي: MACD / هستوجرام"] = f'{last["MACD"]:.3f} / {last["HIST"]:.3f}'
    if bear_d:
        score += 15
    if last["MACD"] < 0:
        score += 5
    if last["RSI"] < 30:
        score -= 10
        R.flags.append(f'RSI اليومي {last["RSI"]:.0f} (تشبّع بيع) — الدخول شورت هنا متأخر، انتظر ارتداد.')
    txt = "اتجاه هابط (السعر تحت EMA20 تحت EMA50)" if bear_d else "الاتجاه مو هابط بشكل واضح"
    if not H.empty:
        h = H.iloc[-1]
        hp = H.iloc[-2]
        m["4س: RSI"] = round(float(h["RSI"]), 1)
        m["4س: MACD / هستوجرام"] = f'{h["MACD"]:.3f} / {h["HIST"]:.3f}'
        m["4س: EMA20"] = round(float(h["EMA20"]), 3)
        weakening = h["HIST"] < hp["HIST"] and h["HIST"] > 0
        if h["RSI"] >= 55 or weakening:
            score += 10
            txt += " | 4س: ارتداد بدأ يضعف — منطقة دخول محتملة"
        elif h["RSI"] < 35:
            txt += " | 4س: السهم في قاع قصير — لا تطارده"
    R.checks.append(("المؤشرات (RSI, MACD)", bool(bear_d and last["MACD"] < 0), txt))

    # ---------- 4) سلوك السهم ----------
    y = D.tail(252)
    hi52 = float(y["High"].max())
    dd = price / hi52 - 1
    m["من أعلى 52 أسبوع"] = f"{dd:.1%}"
    ev = pump_events(D)
    m["عدد الانفجارات (+40%) بالسنة"] = len(ev)
    fades = [e["fade10"] for e in ev if not np.isnan(e["fade10"])]
    pump_fade = len(ev) > 0 and np.mean(fades) < -0.35 if fades else False
    if dd < -0.70:
        score += 10
    if pump_fade:
        score += 10
        m["متوسط التصريف بعد الانفجار (10 أيام)"] = f"{np.mean(fades):.0%}"
    beh = f"نازل {abs(dd):.1%} من القمة السنوية" + (" | نمط (ينفخ ثم يصرّف)" if pump_fade else "")
    R.checks.append(("سلوك السهم", bool(dd < -0.5 or pump_fade), beh))

    # ---------- 5 و 6) هل هبط؟ هل ارتفع؟ ----------
    r5 = price / float(D["Close"].iloc[-6]) - 1
    r20 = price / float(D["Close"].iloc[-21]) - 1
    low20, high20 = float(D["Low"].tail(20).min()), float(D["High"].tail(20).max())
    m["أداء 5 أيام / 20 يوم"] = f"{r5:+.1%} / {r20:+.1%}"
    R.checks.append(("هل هبط السهم؟", r20 < 0, f"20 يوم: {r20:+.1%} — أقل سعر {low20:.2f}"))
    bounce = price / low20 - 1
    R.checks.append(("هل ارتفع السهم؟", bounce > 0.10, f"ارتد {bounce:+.1%} من قاع 20 يوم"))
    recent_ev = [e for e in ev if (D.index[-1] - e["date"]).days <= 30]
    if recent_ev:
        score -= 10
        R.flags.append(f'انفجار +{recent_ev[-1]["gain"]:.0%} خلال آخر 30 يوم ({recent_ev[-1]["date"]:%m-%d}) — خطر Squeeze عالي.')

    # ---------- 7) الدعم والمقاومة + خطة ----------
    sup, res = support_resistance(D, H if not H.empty else None, price)
    a = float(last["ATR"])
    S1 = sup[0][0] if sup else low20
    strong = [x for x in res if x[1] >= 2]
    R1 = strong[0][0] if strong else (res[0][0] if res else high20)
    above_sup = float(D["Close"].tail(3).min()) >= S1 * 0.98
    R.checks.append(("ثبات على الدعم", bool(above_sup),
                     f"الدعم {S1:.2f} " + ("— ثابت فوقه (لا تشورت قبل كسره)" if above_sup else "— مكسور ✔ تأكيد للهبوط")))
    entry_lo = max(price, min(R1 * 0.97, float(H.iloc[-1]["EMA20"]) if not H.empty else R1 * 0.97))
    entry_hi = R1
    stop = R1 + 0.5 * a
    mid = (entry_lo + entry_hi) / 2
    cands = sorted({round(float(p), 4) for p, _ in sup} | {round(low20, 4)}, reverse=True)
    tg = [p for p in cands if p < mid - 0.5 * a]          # أهداف تبعد نص ATR على الأقل
    t1 = tg[0] if tg else min(cands[-1], mid - a)
    t2 = tg[1] if len(tg) > 1 else max(t1 - a, t1 * 0.75)
    risk = stop - mid
    reward = mid - t1
    rr = reward / risk if risk > 0 else np.nan
    entry_lo, entry_hi, stop, t1, t2 = map(float, (entry_lo, entry_hi, stop, t1, t2))
    nd = 4 if price < 1 else 3
    R.levels = {
        "الدعوم": [(round(float(p), nd), n) for p, n in sup],
        "المقاومات": [(round(float(p), nd), n) for p, n in res],
        "منطقة دخول الشورت": (round(entry_lo, nd), round(entry_hi, nd)),
        "وقف الخسارة": round(stop, nd),
        "الهدف 1": round(t1, nd),
        "الهدف 2": round(t2, nd),
        "عائد/مخاطرة": None if np.isnan(rr) else round(rr, 2),
        "ATR يومي": round(a, nd),
    }
    dist_res = R1 / price - 1
    if dist_res > 0.15:
        R.flags.append(f"السعر بعيد عن المقاومة ({dist_res:.0%}) — الأفضل تنتظر ارتداد لمنطقة الدخول بدل البيع الحين.")
    if not np.isnan(rr) and rr < 1.5:
        R.flags.append(f"العائد/المخاطرة {rr:.2f} أقل من 1.5 — الهدف قريب مقارنة بالوقف، الصفقة ما تستاهل.")
    if float_shares:
        m["الأسهم الحرة (Float)"] = f"{float_shares/1e6:.2f}M"
        if float_shares < 5e6:
            score -= 5
            R.flags.append(f"فلوت صغير ({float_shares/1e6:.1f}M) — حركات عنيفة وخطر Squeeze.")

    R.score = int(max(0, min(100, score)))
    if can_short is False:
        R.verdict = "⛔ غير قابل للشورت حالياً"
    elif R.score >= 60 and dist_res <= 0.15 and last["RSI"] >= 30 and rr >= 1.5:
        R.verdict = "🟢 مرشح شورت — قريب من منطقة الدخول"
    elif R.score >= 45:
        R.verdict = "🟡 مراقبة — انتظر ارتداد للمقاومة/EMA20"
    else:
        R.verdict = "🔴 مو مناسب حالياً"
    return R


# ======================================================================
# ===== DATA (يحتاج إنترنت) ============================================
# ======================================================================

def fetch_ibkr_borrow() -> dict:
    """يقرأ ملف IBKR العام للشورت: {SYMBOL: {'fee':%, 'available':int, 'rebate':%}}"""
    from ftplib import FTP
    lines = []
    with FTP("ftp3.interactivebrokers.com", timeout=30) as ftp:
        ftp.login("shortstock", "")
        ftp.retrlines("RETR usa.txt", lines.append)
    out = {}
    for ln in lines:
        if ln.startswith("#") or "|" not in ln:
            continue
        p = ln.split("|")
        try:
            sym = p[0].strip().upper()
            av = p[7].strip()
            avail = 10_000_001 if av.startswith(">") else int(float(av or 0))
            fee = float(p[6]) if p[6].strip() not in ("", "NA") else None
            reb = float(p[5]) if p[5].strip() not in ("", "NA") else None
            out[sym] = {"fee": fee, "available": avail, "rebate": reb}
        except (ValueError, IndexError):
            continue
    return out


def fetch_borrow_chartexchange(ticker: str) -> dict | None:
    """بديل لو ملف IBKR ما اشتغل (بعض الاستضافات تمنع FTP): نفس بيانات IBKR من chartexchange."""
    import re
    import requests
    for ex in ("nasdaq", "nyseamerican", "nyse"):
        try:
            r = requests.get(f"https://chartexchange.com/symbol/{ex}-{ticker.lower()}/borrow-fee/",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            txt = re.sub(r"<[^>]+>", "", r.text)
            m = re.search(r"there were ([\d,]+) shares available with a fee of ([\d.]+)%", txt)
            if m:
                return {"available": int(m.group(1).replace(",", "")), "fee": float(m.group(2))}
        except Exception:
            continue
    return None


def fetch_prices(ticker: str):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    daily = tk.history(period="2y", interval="1d", auto_adjust=False)
    h1 = tk.history(period="60d", interval="60m", auto_adjust=False, prepost=False)
    if not h1.empty:
        h1.index = h1.index.tz_convert("America/New_York").tz_localize(None)
    splits = tk.splits
    if splits is not None and len(splits):
        splits.index = splits.index.tz_localize(None) if splits.index.tz is not None else splits.index
    flt = None
    try:
        flt = tk.get_info().get("floatShares")
    except Exception:
        pass
    return daily, h1, splits, flt


# ======================================================================
# ===== واجهة Streamlit ================================================
# ======================================================================

def chart(daily: pd.DataFrame, r: Result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    d = add_indicators(daily).tail(120)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="السعر"), 1, 1)
    for n, c in ((20, "#2a9df4"), (30, "#f4a62a"), (50, "#a24bd6")):
        fig.add_trace(go.Scatter(x=d.index, y=d[f"EMA{n}"], name=f"EMA{n}", line=dict(width=1.2, color=c)), 1, 1)
    for p, _ in r.levels.get("الدعوم", []):
        fig.add_hline(y=p, line=dict(color="#2e9e5b", dash="dot"), row=1, col=1)
    for p, _ in r.levels.get("المقاومات", []):
        fig.add_hline(y=p, line=dict(color="#d64545", dash="dot"), row=1, col=1)
    lo, hi = r.levels.get("منطقة دخول الشورت", (None, None))
    if lo:
        fig.add_hrect(y0=lo, y1=hi, fillcolor="#d64545", opacity=0.12, line_width=0, row=1, col=1)
        fig.add_hline(y=r.levels["وقف الخسارة"], line=dict(color="#d64545", width=2), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI", line=dict(color="#555")), 2, 1)
    fig.add_hline(y=30, line=dict(dash="dot", color="#999"), row=2, col=1)
    fig.add_hline(y=70, line=dict(dash="dot", color="#999"), row=2, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d.HIST, name="Hist",
                         marker_color=np.where(d.HIST >= 0, "#2e9e5b", "#d64545")), 3, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD, name="MACD", line=dict(color="#2a9df4", width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.SIGNAL, name="Signal", line=dict(color="#f4a62a", width=1)), 3, 1)
    fig.update_layout(height=640, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.02))
    return fig


def main():
    import streamlit as st
    st.set_page_config(page_title="بوت شورت التقسيم العكسي", page_icon="📉", layout="wide")
    st.markdown("""<style>
    html, body, [class*="css"], .stMarkdown, .stTextArea, .stDataFrame {direction: rtl; text-align: right;}
    .js-plotly-plot {direction: ltr;}
    .flag {background:#fff4e5;border-right:4px solid #f0a020;padding:6px 10px;margin:4px 0;border-radius:4px;color:#222}
    </style>""", unsafe_allow_html=True)
    st.title("📉 بوت فلترة أسهم التقسيم العكسي للشورت")
    st.caption("طريقة @Naifo9x: التقسيم العكسي ← الشورت ← المؤشرات ← سلوك السهم ← هبط/ارتفع ← الدعم — "
               "أداة تعليمية، مو توصية.")

    with st.sidebar:
        st.header("الرموز")
        raw = st.text_area("اكتب الرموز (مسافة أو فاصلة أو سطر)", "ELAB GPUS", height=120)
        go_btn = st.button("🔍 حلّل", type="primary", use_container_width=True)
        st.markdown("مصادر قوائم التقسيم العكسي: "
                    "[TipRanks](https://www.tipranks.com/calendars/stock-splits/upcoming) · "
                    "[ChartExchange](https://chartexchange.com)")

    if not go_btn:
        st.info("اكتب الرموز في الجانب واضغط **حلّل**.")
        return

    tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]

    @st.cache_data(ttl=900, show_spinner=False)
    def _borrow():
        return fetch_ibkr_borrow()

    @st.cache_data(ttl=900, show_spinner=False)
    def _prices(t):
        return fetch_prices(t)

    @st.cache_data(ttl=900, show_spinner=False)
    def _borrow_ce(t):
        return fetch_borrow_chartexchange(t)

    try:
        with st.spinner("أجيب بيانات الشورت من IBKR..."):
            borrow_map = _borrow()
    except Exception:
        borrow_map = {}
        st.info("ملف IBKR ما اشتغل هنا — بجيب بيانات الشورت من chartexchange بدلاً منه.")

    results, dailies = [], {}
    bar = st.progress(0.0)
    for i, t in enumerate(tickers):
        try:
            daily, h1, splits, flt = _prices(t)
            bw = borrow_map.get(t) or _borrow_ce(t)
            r = analyze(t, daily, h1, splits, bw, flt)
            dailies[t] = daily
        except Exception as e:
            r = Result(t, error=str(e))
        results.append(r)
        bar.progress((i + 1) / len(tickers))
    bar.empty()

    rows = [{"الرمز": r.ticker, "السعر": None if math.isnan(r.price) else round(r.price, 3), "النقاط": r.score,
             "الحكم": r.verdict or r.error,
             "رسوم الاقتراض": r.metrics.get("رسوم الاقتراض السنوية", "—"),
             "متاح": r.metrics.get("متاح للشورت (IBKR)", "—"),
             "RSI يومي": str(r.metrics.get("يومي: RSI", "—")),
             "تحذيرات": len(r.flags)} for r in results]
    st.subheader("الملخص")
    st.dataframe(pd.DataFrame(rows).sort_values("النقاط", ascending=False), use_container_width=True, hide_index=True)

    for r in sorted(results, key=lambda x: -x.score):
        with st.expander(f"{r.ticker} — {r.verdict or r.error} — {r.score}/100", expanded=len(results) == 1):
            if r.error:
                st.error(r.error)
                continue
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**قائمة الفحص**")
                for name, ok, note in r.checks:
                    icon = "✅" if ok else ("➖" if ok is None else "❌")
                    st.markdown(f"{icon} **{name}** — {note}")
                st.markdown("**لاحقاً تحديد:**")
                L = r.levels
                st.markdown(
                    f"- الدعوم: {', '.join(f'{p} ({n} لمسات)' for p, n in L['الدعوم']) or '—'}\n"
                    f"- المقاومات: {', '.join(f'{p} ({n} لمسات)' for p, n in L['المقاومات']) or '—'}\n"
                    f"- منطقة دخول الشورت: **{L['منطقة دخول الشورت'][0]} – {L['منطقة دخول الشورت'][1]}**\n"
                    f"- الأهداف: {L['الهدف 1']} ثم {L['الهدف 2']}\n"
                    f"- وقف الخسارة: **{L['وقف الخسارة']}**\n"
                    f"- عائد/مخاطرة: {L['عائد/مخاطرة']}")
            with c2:
                st.markdown("**الأرقام**")
                st.table(pd.DataFrame({"القيمة": {k: str(v) for k, v in r.metrics.items()}}))
                if r.flags:
                    st.markdown("**⚠️ تحذيرات**")
                    for f in r.flags:
                        st.markdown(f"<div class='flag'>{f}</div>", unsafe_allow_html=True)
            st.plotly_chart(chart(dailies[r.ticker], r), use_container_width=True, key=f"chart_{r.ticker}")


if __name__ == "__main__":
    main()
