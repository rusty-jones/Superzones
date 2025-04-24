import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import matplotlib.patches as patches
import warnings
import traceback

warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)

# Ticker mapping (same as PyQt5)
ticker_mapping = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN",
    "^NSEI": "^NSEI",
    "^NSEBANK": "^NSEBANK",
    "^BSESN": "^BSESN"
}

# Display mapping (subset for brevity; add more as needed)
display_mapping = {
    "^NSEI": "Nifty",
    "^NSEBANK": "Banknifty",
    "^BSESN": "Sensex",
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "HDFCBANK": "HDFCBANK",
    "ICICIBANK": "ICICIBANK",
    "INFY": "INFY",
    "SBIN": "SBIN",
    "BHARTIARTL": "BHARTIARTL"
}

# Fetch and process data
def fetch_and_process_data(ticker, period, interval, use_ema, ema_period, use_rsi, trade_log):
    try:
        mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
        if not mapped_ticker.startswith('^'):
            mapped_ticker = f"{ticker.upper()}.NS"
        trade_log.append(f"Fetching data for {ticker} ({mapped_ticker}, Period: {period}, Interval: {interval})")
        data = yf.download(mapped_ticker, period=period, interval=interval, progress=False)
        if data.empty:
            trade_log.append(f"No data returned for {ticker} ({mapped_ticker})")
            return None
        data.reset_index(inplace=True)
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        data.set_index('Date', inplace=True)
        data.index = data.index.tz_localize(None)
        trade_log.append(f"Fetched {len(data)} data points for {ticker}. First: {data.index[0]}, Last: {data.index[-1]}")
        if use_ema:
            data['ema'] = data['close'].ewm(span=ema_period, adjust=False).mean()
        if use_rsi:
            data['rsi'] = compute_rsi(data['close'], 14)
        data.dropna(inplace=True)
        trade_log.append(f"After processing, {len(data)} data points remain for {ticker}")
        return data
    except Exception as e:
        trade_log.append(f"Error fetching data for {ticker}: {str(e)}")
        return None

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Identify base and following candles
def identify_base_and_following_candles(data, trade_log):
    base_rally_candles = []
    base_drop_candles = []
    try:
        for i in range(1, len(data) - 2):
            body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
            total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
            body_size_rally = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
            total_height_rally = data['high'].iloc[i+1] - data['low'].iloc[i+1]
            body_size_drop = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
            total_height_drop = data['high'].iloc[i+1] - data['low'].iloc[i+1]
            if (body_size_base <= 0.5 * total_height_base) and (body_size_rally >= 0.71 * total_height_rally) and (data['close'].iloc[i+1] > data['open'].iloc[i+1]):
                base_rally_candles.append(data.index[i])
            if (body_size_base <= 0.5 * total_height_base) and (body_size_drop >= 0.71 * total_height_drop) and (data['close'].iloc[i+1] < data['open'].iloc[i+1]):
                base_drop_candles.append(data.index[i])
        trade_log.append(f"Identified {len(base_rally_candles)} base rally candles and {len(base_drop_candles)} base drop candles")
        if not base_rally_candles and not base_drop_candles:
            trade_log.append("No base candles detected. Try a shorter interval (e.g., 5m) or different period.")
        return base_rally_candles, base_drop_candles
    except Exception as e:
        trade_log.append(f"Error identifying base candles: {str(e)}")
        return [], []

# Plot candlestick chart with base candles
def plot_candlestick_chart(df, base_rally_candles, base_drop_candles, title):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = df.copy()
        df_plot.index.name = 'Date'
        mpf.plot(df_plot, type='candle', style='charles', ylabel='Price', volume=False, ax=ax)
        ax.set_title(title, fontsize=10)
        
        for date in base_rally_candles:
            idx = df.index.get_loc(date)
            open_price = df.loc[date, 'open']
            close_price = df.loc[date, 'close']
            high_body = max(open_price, close_price)
            low_candle = df.loc[date, 'low']
            rect = patches.Rectangle((idx, low_candle), len(df) - idx, high_body - low_candle, 
                                    linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
            ax.add_patch(rect)
            ax.text(len(df) - 1, high_body, f'{high_body:.2f}', color='blue', va='center', ha='left', fontsize=8)

        for date in base_drop_candles:
            idx = df.index.get_loc(date)
            open_price = df.loc[date, 'open']
            close_price = df.loc[date, 'close']
            high_candle = df.loc[date, 'high']
            low_body = min(open_price, close_price)
            rect = patches.Rectangle((idx, low_body), len(df) - idx, high_candle - low_body, 
                                    linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(rect)
            ax.text(len(df) - 1, low_body, f'{low_body:.2f}', color='red', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting Base Candles chart: {str(e)}")
        st.write(traceback.format_exc())
        return None
    finally:
        plt.close('all')

# Identify super zones
def identify_super_zones(ticker, period, interval, trade_log):
    try:
        data = fetch_and_process_data(ticker, period, interval, False, 20, False, trade_log)
        if data is None:
            return []
        zones = []
        for i in range(1, len(data) - 1):
            if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                zones.append({'date': data.index[i], 'type': 'demand', 'level': data['low'].iloc[i]})
            if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                zones.append({'date': data.index[i], 'type': 'supply', 'level': data['high'].iloc[i]})
        trade_log.append(f"Identified {len(zones)} super zones for {ticker} ({period}/{interval})")
        return zones
    except Exception as e:
        trade_log.append(f"Error identifying super zones for {ticker}: {str(e)}")
        return []

# Plot super zones chart
def plot_super_zones_chart(df, zones, ticker, period, interval, use_ema, use_rsi):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = df.copy()
        df_plot.index.name = 'Date'
        addplots = []
        if use_ema and 'ema' in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot['ema'], color='orange', label='EMA'))
        if use_rsi and 'rsi' in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot['rsi'], panel=1, color='purple', label='RSI', ylabel='RSI'))
        mpf.plot(df_plot, type='candle', style='classic', ylabel='Price', volume=False, ax=ax, addplot=addplots)
        ax.set_title(f"{ticker} - {period}/{interval} (Super Zones)", fontsize=10)
        
        for zone in zones:
            limit_price = zone['level']
            side = 'BUY' if zone['type'] == 'demand' else 'SELL'
            color = 'blue' if side == 'BUY' else 'red'
            ax.axhline(y=limit_price, color=color, linestyle='--', alpha=0.5)
            ax.text(len(df) - 1, limit_price, f'{limit_price:.2f}', color=color, va='center', ha='right', fontsize=8)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting Super Zones chart: {str(e)}")
        st.write(traceback.format_exc())
        return None
    finally:
        plt.close('all')

# Streamlit app
def main():
    st.set_page_config(page_title="Stock Charting Tool", layout="wide")
    st.title("Stock Charting Tool")

    # Initialize trade log
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []

    # Sidebar
    st.sidebar.header("Chart Settings")
    
    # Stock list
    stock_list = sorted(display_mapping.values())
    selected_stock = st.sidebar.selectbox("Select Stock", stock_list, index=stock_list.index("RELIANCE") if "RELIANCE" in stock_list else 0)
    ticker = [k for k, v in display_mapping.items() if v == selected_stock][0]
    
    # Ticker input
    ticker_input = st.sidebar.text_input("Ticker (e.g., RELIANCE, Nifty)", ticker)
    if ticker_input:
        ticker = ticker_input

    # Period and interval
    period = st.sidebar.selectbox("Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=2)
    interval = st.sidebar.selectbox("Interval", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], index=5)
    
    # Toggles
    use_ema = st.sidebar.checkbox("EMA", value=True)
    ema_period = st.sidebar.text_input("EMA Period", "20")
    use_rsi = st.sidebar.checkbox("RSI", value=True)
    
    # Plot button
    plot_button = st.sidebar.button("Plot Charts")

    # Main content
    if plot_button and ticker:
        with st.spinner("Fetching data..."):
            try:
                ema_period_int = int(ema_period) if ema_period.isdigit() else 20
                df = fetch_and_process_data(ticker, period, interval, use_ema, ema_period_int, use_rsi, st.session_state.trade_log)
                if df is not None:
                    # Super Zones Chart
                    st.subheader("Super Zones Chart")
                    zones = identify_super_zones(ticker, period, interval, st.session_state.trade_log)
                    fig1 = plot_super_zones_chart(df, zones, ticker, period, interval, use_ema, use_rsi)
                    if fig1:
                        st.pyplot(fig1)
                        plt.close(fig1)
                    else:
                        st.error("Failed to render Super Zones chart.")
                    
                    # Base Candles Chart
                    st.subheader("Base Candles Chart")
                    base_rally_candles, base_drop_candles = identify_base_and_following_candles(df, st.session_state.trade_log)
                    fig2 = plot_candlestick_chart(df, base_rally_candles, base_drop_candles, 
                                                 f"{ticker} - {period}/{interval} (Base Candles)")
                    if fig2:
                        st.pyplot(fig2)
                        plt.close(fig2)
                    else:
                        st.error("Failed to render Base Candles chart.")
            except Exception as e:
                st.session_state.trade_log.append(f"Error plotting charts: {str(e)}")
                st.error(f"Error plotting charts: {str(e)}")
                st.write(traceback.format_exc())

    elif plot_button:
        st.error("Please enter a valid ticker.")
        st.session_state.trade_log.append("No ticker specified")

    # Trade log
    st.subheader("Trade Log")
    trade_log_text = "\n".join(st.session_state.trade_log[-50:])  # Limit to last 50 entries
    st.text_area("Logs", trade_log_text, height=150, disabled=True)

if __name__ == "__main__":
    main()
