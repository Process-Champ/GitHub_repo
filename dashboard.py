"""
Trading P&L Dashboard — Streamlit
Reads signal history from Google Sheets → shows live equity curve,
signal table, win rate, drawdown. Deploy to Streamlit Cloud.
"""

import os
import json
import datetime
import pytz
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import plotly.graph_objects as go
import plotly.express as px
from google.oauth2.service_account import Credentials

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST = pytz.timezone("Asia/Kolkata")

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
  .signal-buy  { color: #22c55e; font-weight: 700; }
  .signal-sell { color: #ef4444; font-weight: 700; }
  .signal-hold { color: #94a3b8; }
  .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── GOOGLE SHEETS AUTH ───────────────────────────────────────────────────────

@st.cache_resource(ttl=300)
def get_sheet():
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    # In Streamlit Cloud, store this in st.secrets as [google] section
    try:
        creds_dict = dict(st.secrets["google"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    except Exception:
        # Fallback: env var (local dev)
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON", "{}")
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)
    spreadsheet = client.open("Trading data")
    return spreadsheet.worksheet("Signals")


@st.cache_data(ttl=60)  # Refresh every 60 seconds
def load_signals() -> pd.DataFrame:
    sheet = get_sheet()
    data = sheet.get_all_records()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()

    # Type conversions
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTP"] = pd.to_numeric(df["LTP"], errors="coerce")
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
    df["MACD"] = pd.to_numeric(df["MACD"], errors="coerce")
    df["EMA9"] = pd.to_numeric(df["EMA9"], errors="coerce")
    df["EMA21"] = pd.to_numeric(df["EMA21"], errors="coerce")
    df["Vol_Ratio"] = pd.to_numeric(df["Vol_Ratio"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df.sort_values(["Date", "Time"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── P&L SIMULATION ───────────────────────────────────────────────────────────

def simulate_pnl(df: pd.DataFrame, capital_per_trade: float = 10000) -> pd.DataFrame:
    """
    Paper trade simulation:
    - BUY signal → enter at LTP
    - SELL signal on same stock → exit, book P&L
    - HOLD → no action
    Returns a trade log with P&L per trade.
    """
    trades = []
    open_positions = {}  # symbol → entry price

    for _, row in df.iterrows():
        sym = row["Symbol"]
        sig = row["Signal"]
        ltp = row["LTP"]
        date = row["Date"]
        time_ = row["Time"]

        if sig == "BUY" and sym not in open_positions:
            open_positions[sym] = {
                "entry_price": ltp,
                "entry_date": date,
                "entry_time": time_,
                "qty": int(capital_per_trade / ltp) if ltp > 0 else 0,
            }

        elif sig == "SELL" and sym in open_positions:
            entry = open_positions.pop(sym)
            qty = entry["qty"]
            pnl = round((ltp - entry["entry_price"]) * qty, 2)
            pnl_pct = round(((ltp - entry["entry_price"]) / entry["entry_price"]) * 100, 2) if entry["entry_price"] > 0 else 0
            trades.append({
                "Symbol":      sym,
                "Entry_Date":  entry["entry_date"],
                "Entry_Time":  entry["entry_time"],
                "Exit_Date":   date,
                "Exit_Time":   time_,
                "Entry_Price": entry["entry_price"],
                "Exit_Price":  ltp,
                "Qty":         qty,
                "PnL":         pnl,
                "PnL_Pct":     pnl_pct,
                "Result":      "WIN" if pnl >= 0 else "LOSS",
            })

    return pd.DataFrame(trades)


def build_equity_curve(trades_df: pd.DataFrame, starting_capital: float = 100000) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"Date": [], "Equity": []})

    trades_df = trades_df.sort_values("Exit_Date")
    equity = starting_capital
    curve = []
    for _, row in trades_df.iterrows():
        equity += row["PnL"]
        curve.append({"Date": row["Exit_Date"], "Equity": round(equity, 2), "Trade_PnL": row["PnL"]})

    return pd.DataFrame(curve)


def max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return round(drawdown.min() * 100, 2)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    capital_per_trade = st.number_input("Capital per trade (₹)", value=10000, step=1000, min_value=1000)
    starting_capital  = st.number_input("Starting capital (₹)", value=100000, step=10000, min_value=10000)

    st.divider()
    st.subheader("Filter")
    date_filter = st.date_input(
        "Show signals from",
        value=datetime.date.today() - datetime.timedelta(days=15),
    )
    sig_filter = st.multiselect(
        "Signal type",
        options=["BUY", "SELL", "HOLD"],
        default=["BUY", "SELL", "HOLD"],
    )
    conf_filter = st.multiselect(
        "Confidence",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    now_ist = datetime.datetime.now(IST)
    st.caption(f"Last refresh: {now_ist.strftime('%H:%M:%S IST')}")
    st.caption("Auto-refreshes every 60 seconds")

# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────

st.title("📈 Trading Agent — Live P&L Dashboard")
st.caption("Paper trading mode · Nifty 50 Top 10 · 15-day observation window")

# Load data
try:
    df = load_signals()
except Exception as e:
    st.error(f"Could not load data from Google Sheets: {e}")
    st.info("Make sure your `st.secrets` has the [google] service account credentials.")
    st.stop()

if df.empty:
    st.warning("No signals yet. The agent hasn't run or the sheet is empty.")
    st.stop()

# Apply filters
df_filtered = df[
    (df["Date"].dt.date >= date_filter) &
    (df["Signal"].isin(sig_filter)) &
    (df["Confidence"].isin(conf_filter))
]

trades_df   = simulate_pnl(df, capital_per_trade)
equity_df   = build_equity_curve(trades_df, starting_capital)

# ─── KPI METRICS ──────────────────────────────────────────────────────────────

total_trades  = len(trades_df)
wins          = len(trades_df[trades_df["Result"] == "WIN"]) if not trades_df.empty else 0
losses        = total_trades - wins
win_rate      = round((wins / total_trades) * 100, 1) if total_trades > 0 else 0
total_pnl     = round(trades_df["PnL"].sum(), 2) if not trades_df.empty else 0
current_eq    = starting_capital + total_pnl
max_dd        = max_drawdown(equity_df["Equity"]) if not equity_df.empty else 0
buy_signals   = len(df[df["Signal"] == "BUY"])
sell_signals  = len(df[df["Signal"] == "SELL"])

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Portfolio Value", f"₹{current_eq:,.0f}",
              delta=f"₹{total_pnl:+,.2f}")
with col2:
    st.metric("Win Rate", f"{win_rate}%",
              delta=f"{wins}W / {losses}L")
with col3:
    st.metric("Total Trades", total_trades)
with col4:
    st.metric("Max Drawdown", f"{max_dd}%")
with col5:
    st.metric("BUY Signals", buy_signals)
with col6:
    st.metric("SELL Signals", sell_signals)

st.divider()

# ─── EQUITY CURVE ─────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Equity Curve")
    if not equity_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["Date"],
            y=equity_df["Equity"],
            mode="lines+markers",
            name="Portfolio Value",
            line=dict(color="#22c55e" if total_pnl >= 0 else "#ef4444", width=2),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)" if total_pnl >= 0 else "rgba(239,68,68,0.08)",
            hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(y=starting_capital, line_dash="dash", line_color="gray",
                      annotation_text=f"Starting ₹{starting_capital:,}", annotation_position="right")
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title=None, yaxis_title="Portfolio (₹)",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
            xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Equity curve will appear after the first closed trade (BUY then SELL).")

with col_right:
    st.subheader("Signal Distribution")
    sig_counts = df["Signal"].value_counts().reset_index()
    sig_counts.columns = ["Signal", "Count"]
    colors = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#94a3b8"}
    fig2 = px.pie(
        sig_counts, values="Count", names="Signal",
        color="Signal",
        color_discrete_map=colors,
        hole=0.5,
    )
    fig2.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── TRADE LOG ────────────────────────────────────────────────────────────────

col_trades, col_open = st.columns([3, 2])

with col_trades:
    st.subheader("Closed Trade Log")
    if not trades_df.empty:
        display_trades = trades_df.copy()
        display_trades["PnL"] = display_trades["PnL"].map(lambda x: f"₹{x:+,.2f}")
        display_trades["PnL_Pct"] = display_trades["PnL_Pct"].map(lambda x: f"{x:+.2f}%")
        display_trades["Entry_Price"] = display_trades["Entry_Price"].map(lambda x: f"₹{x:,.2f}")
        display_trades["Exit_Price"]  = display_trades["Exit_Price"].map(lambda x: f"₹{x:,.2f}")

        def color_result(val):
            if val == "WIN":
                return "background-color: rgba(34,197,94,0.12); color: #16a34a; font-weight:600"
            elif val == "LOSS":
                return "background-color: rgba(239,68,68,0.12); color: #dc2626; font-weight:600"
            return ""

        styled = display_trades[["Symbol","Entry_Date","Entry_Price","Exit_Date","Exit_Price","Qty","PnL","PnL_Pct","Result"]].style.applymap(
            color_result, subset=["Result"]
        )
        st.dataframe(styled, use_container_width=True, height=280)
    else:
        st.info("No closed trades yet. A trade closes when a SELL signal follows a BUY on the same stock.")

with col_open:
    st.subheader("Latest Signals")
    latest = df_filtered.sort_values(["Date","Time"], ascending=False).head(20)
    if not latest.empty:
        display_sig = latest[["Time","Symbol","Signal","LTP","RSI","Confidence","Notes"]].copy()
        display_sig["LTP"] = display_sig["LTP"].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
        display_sig["RSI"] = display_sig["RSI"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

        def color_signal(val):
            if val == "BUY":  return "color: #22c55e; font-weight:700"
            if val == "SELL": return "color: #ef4444; font-weight:700"
            return "color: #94a3b8"

        styled_sig = display_sig.style.applymap(color_signal, subset=["Signal"])
        st.dataframe(styled_sig, use_container_width=True, height=280)

st.divider()

# ─── PER-STOCK BREAKDOWN ──────────────────────────────────────────────────────

st.subheader("Per-Stock Signal Heatmap (last 15 days)")

pivot_data = df[df["Date"].dt.date >= date_filter].copy()
if not pivot_data.empty:
    sig_map   = {"BUY": 1, "HOLD": 0, "SELL": -1}
    pivot_data["sig_val"] = pivot_data["Signal"].map(sig_map)
    pivot_data["DateStr"] = pivot_data["Date"].dt.strftime("%m/%d")

    # Latest signal per stock per day
    latest_per_day = pivot_data.sort_values("Time").groupby(["Symbol", "DateStr"]).last().reset_index()
    pivot = latest_per_day.pivot(index="Symbol", columns="DateStr", values="sig_val").fillna(0)

    fig3 = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0.0, "#ef4444"], [0.5, "#1e293b"], [1.0, "#22c55e"]],
        zmin=-1, zmax=1,
        text=[[{1: "BUY", 0: "HOLD", -1: "SELL"}.get(int(v), "") for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b> on %{x}<br>Signal: %{text}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickvals=[-1, 0, 1],
            ticktext=["SELL", "HOLD", "BUY"],
            thickness=12,
        ),
    ))
    fig3.update_layout(
        height=350, margin=dict(l=0, r=60, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None, yaxis_title=None,
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No data in selected date range.")

# ─── RAW DATA EXPANDER ────────────────────────────────────────────────────────

with st.expander("📋 Raw signal data (all records)"):
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=f"signals_{datetime.date.today()}.csv",
        mime="text/csv",
    )

# ─── AUTO REFRESH ─────────────────────────────────────────────────────────────

st.markdown("""
<script>
setTimeout(function(){ window.location.reload(); }, 60000);
</script>
""", unsafe_allow_html=True)
