import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
from datetime import datetime
import pytz

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)

# Ticker mapping
ticker_mapping = {
    "nifty50": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN"
}

# Timeframe and period mapping
timeframes = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
}
periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]

# Initialize session state
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
if 'zones_list' not in st.session_state:
    st.session_state.zones_list = {}
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# Function to fetch and clean data
def fetch_and_process_data(tickers, periods_intervals):
    all_data = {}
    for ticker in tickers:
        all_data[ticker] = {}
        for period, intervals in periods_intervals.items():
            for interval in intervals:
                try:
                    data = yf.download(ticker, period=period, interval=interval)
                    st.session_state.trade_log.append(f"Fetched data for {ticker} ({period}, {interval}): Length={len(data)}")
                    if data.empty:
                        st.session_state.trade_log.append(f"No data fetched for {ticker} ({period}, {interval})")
                        continue
                    data.reset_index(inplace=True)
                    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                    data.set_index('Date', inplace=True)
                    if len(data) >= 10000:
                        st.session_state.trade_log.append(f"Data exceeds 10000 candles for {ticker} ({period}, {interval}), truncated")
                        data = data.iloc[-10000:]
                    all_data[ticker][(period, interval)] = data
                except Exception as e:
                    st.session_state.trade_log.append(f"Error fetching {ticker} ({period}, {interval}): {str(e)}")
    return all_data

# Zone identification
def identify_zones(data):
    zones = []
    if len(data) < 2:
        st.session_state.trade_log.append("Insufficient data for zone identification")
        return zones
    for i in range(1, len(data) - 1):
        body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
        total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
        if body_size_base > 0.5 * total_height_base or total_height_base == 0:
            continue
        body_size_follow = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_follow = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        if body_size_follow < 0.75 * total_height_follow or total_height_follow == 0:
            continue
        date = data.index[i].to_pydatetime()
        ist = pytz.timezone('Asia/Kolkata')
        index_tz = data.index.tz
        if index_tz:
            zone_date = date.replace(tzinfo=index_tz) if date.tzinfo is None else date.astimezone(index_tz)
        else:
            zone_date = ist.localize(date) if date.tzinfo is None else date.astimezone(ist)
        if data['close'].iloc[i+1] > data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'demand', 'level': data['low'].iloc[i]})
        elif data['close'].iloc[i+1] < data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'supply', 'level': data['high'].iloc[i]})
    st.session_state.trade_log.append(f"Identified {len(zones)} zones for data length {len(data)}")
    return zones

# Plotting function
def plot_chart(df, zones, symbol, timeframe, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices):
    if df is None or zones is None:
        st.session_state.trade_log.append(f"plot_chart failed for {symbol} (Timeframe: {timeframe}, Period: {period}): df or zones is None")
        return None
    fig, ax = mpf.plot(df, type='candle', style='charles', returnfig=True, figsize=(10, 5))
    ax[0].set_title(f'{symbol} Candlestick (Timeframe: {timeframe}, Period: {period})')
    ax[0].set_ylabel('Price')

    for zone in zones:
        limit_price = zone['level']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        if side == 'BUY' and not show_buy_zones:
            continue
        if side == 'SELL' and not show_sell_zones:
            continue
        color = 'blue' if side == 'BUY' else 'red'
        ax[0].axhline(y=limit_price, color=color, linestyle='--', alpha=0.5)

    return fig

# Streamlit UI
st.set_page_config(page_title="Indian Stocks Trading App", layout="wide")
st.title("Indian Stocks Trading App")

# Guidelines
st.markdown("""
**Guidelines**: 
- Enter valid tickers (e.g., `nifty50, banknifty`) separated by commas.
- Use consistent timeframes and periods (e.g., `1d` with `1mo`, `1wk` with `6mo`).
- Uncheck `NSE stocks` for indices like `nifty50` (`^NSEI`); check for stocks (e.g., `RELIANCE` → `RELIANCE.NS`).
""")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    tickers = st.text_input("Enter stock symbols (e.g., nifty50, banknifty)", "nifty50")
    append_ns = st.checkbox("NSE stocks", value=True)
    
    st.subheader("Timeframe 1")
    timeframe_1 = st.selectbox("Timeframe 1", list(timeframes.keys()), index=6, key="tf1")  # Default to 1d
    period_1 = st.selectbox("Period 1", periods, index=4, key="p1")
    
    st.subheader("Timeframe 2")
    timeframe_2 = st.selectbox("Timeframe 2", list(timeframes.keys()), index=7, key="tf2")  # Default to 1wk
    period_2 = st.selectbox("Period 2", periods, index=2, key="p2")
    
    show_buy_zones = st.checkbox("Show Demand Zones", value=True)
    show_sell_zones = st.checkbox("Show Supply Zones", value=True)
    show_limit_lines = st.checkbox("Show Limit Lines", value=True)
    show_prices = st.checkbox("Show Zone Prices", value=False)
    plot_button = st.button("Plot Chart")

# Process tickers
st.session_state.trade_log.append(f"Raw ticker input: {tickers}")
tickers_clean = tickers.strip()
if not tickers_clean:
    tickers_clean = "nifty50"
    st.session_state.trade_log.append("No valid tickers provided, defaulting to 'nifty50'")

entered_tickers = [ticker.strip().lower() for ticker in tickers_clean.split(",") if ticker.strip()]
if not entered_tickers:
    entered_tickers = ["nifty50"]
    st.session_state.trade_log.append("No valid tickers parsed, defaulting to ['nifty50']")

mapped_tickers = [ticker_mapping.get(ticker, ticker) for ticker in entered_tickers]
final_ticker_list = [symbol.upper() + ".NS" if append_ns and not symbol.startswith('^') else symbol.upper() for symbol in mapped_tickers if symbol]
st.session_state.trade_log.append(f"Final ticker list: {final_ticker_list}")

# Validate timeframes
timeframes_list = [timeframe_1, timeframe_2]
if isinstance(timeframes_list, tuple):
    timeframes_list = list(timeframes_list)
    st.session_state.trade_log.append("Converted timeframes_list from tuple to list")
st.session_state.trade_log.append(f"Timeframes list: {timeframes_list}")

# Define periods_intervals
periods_list = [period_1, period_2]
periods_intervals = {periods_list[i]: [timeframes_list[i]] for i in range(2)}
st.session_state.trade_log.append(f"Periods intervals: {periods_intervals}")

# Chart Tab
if plot_button:
    st.session_state.dfs = {ticker: [None] * 2 for ticker in final_ticker_list}
    st.session_state.zones_list = {ticker: [None] * 2 for ticker in final_ticker_list}
    
    all_data = fetch_and_process_data(final_ticker_list, periods_intervals)
    for ticker in final_ticker_list:
        st.session_state.trade_log.append(f"Processing ticker: {ticker}")
        data_fetched = False
        if ticker in all_data:
            for idx, (tf, period) in enumerate(zip(timeframes_list, periods_list)):
                if (period, tf) in all_data[ticker]:
                    df = all_data[ticker][(period, tf)]
                    st.session_state.dfs[ticker][idx] = df
                    st.session_state.zones_list[ticker][idx] = identify_zones(df)
                    data_fetched = True
                    st.session_state.trade_log.append(f"Found {len(st.session_state.zones_list[ticker][idx])} zones for {ticker} (Timeframe: {tf}, Period: {period})")
                else:
                    st.session_state.trade_log.append(f"No data found for {ticker} (Timeframe: {tf}, Period: {period})")
        
        if not data_fetched:
            st.error(f"No valid data fetched for {ticker}. Please check ticker or timeframe settings.")
        else:
            if final_ticker_list:
                cols = st.columns(min(4, len(final_ticker_list)))
                idx = 0
                for ticker in final_ticker_list:
                    for tf, period, df, zones in zip(timeframes_list, periods_list, st.session_state.dfs[ticker], st.session_state.zones_list[ticker]):
                        if df is not None and zones is not None:
                            with cols[idx % 4]:
                                fig = plot_chart(df, zones, ticker, tf, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices)
                                if fig:
                                    st.pyplot(fig)
                                    st.session_state.trade_log.append(f"Chart plotted successfully for {ticker} (Timeframe: {tf})!")
                                    plt.close(fig)
                            idx += 1

# Trade Log
st.header("Trade Log")
st.text_area("Log", "\n".join(st.session_state.trade_log[-20:]), height=300, disabled=True)
