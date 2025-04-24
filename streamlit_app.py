import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import matplotlib.patches as patches
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)

# Fetch and process data
def fetch_and_process_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            st.warning(f"No data returned for {ticker}")
            return None
        data.reset_index(inplace=True)
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        data.set_index('Date', inplace=True)
        data.index = data.index.tz_localize(None)
        st.write(f"Fetched {len(data)} data points for {ticker} ({period}/{interval})")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Identify base and following candles
def identify_base_and_following_candles(data):
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
        st.write(f"Identified {len(base_rally_candles)} base rally candles and {len(base_drop_candles)} base drop candles")
        if not base_rally_candles and not base_drop_candles:
            st.warning("No base candles detected. Try a shorter interval (e.g., 5m) or different period.")
        return base_rally_candles, base_drop_candles
    except Exception as e:
        st.error(f"Error identifying base candles: {str(e)}")
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
def identify_super_zones(ticker, period, interval):
    try:
        data = fetch_and_process_data(ticker, period, interval)
        if data is None:
            return []
        zones = []
        for i in range(1, len(data) - 1):
            if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                zones.append({'date': data.index[i], 'type': 'demand', 'level': data['low'].iloc[i]})
            if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                zones.append({'date': data.index[i], 'type': 'supply', 'level': data['high'].iloc[i]})
        st.write(f"Identified {len(zones)} super zones")
        return zones
    except Exception as e:
        st.error(f"Error identifying super zones: {str(e)}")
        return []

# Plot super zones chart
def plot_super_zones_chart(df, zones, ticker, period, interval):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = df.copy()
        df_plot.index.name = 'Date'
        mpf.plot(df_plot, type='candle', style='classic', ylabel='Price', volume=False, ax=ax)
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

# Send email notification
def send_email(subject, body, to_email, smtp_server, smtp_port, sender_email, sender_password):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        st.success("Email notification sent!")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")

# Streamlit app
def main():
    st.set_page_config(page_title="Stock Charting Tool", layout="wide")
    st.title("Stock Charting Tool")

    # Sidebar for inputs
    st.sidebar.header("Chart Settings")
    ticker = st.sidebar.text_input("Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
    period = st.sidebar.selectbox("Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y'], index=2)
    interval = st.sidebar.selectbox("Interval", ['1m', '5m', '15m', '30m', '1h', '1d'], index=5)
    email = st.sidebar.text_input("Email for Notifications")
    smtp_server = st.sidebar.text_input("SMTP Server", "smtp.gmail.com")
    smtp_port = st.sidebar.number_input("SMTP Port", value=587)
    sender_email = st.sidebar.text_input("Sender Email")
    sender_password = st.sidebar.text_input("Sender Password", type="password")
    plot_button = st.sidebar.button("Plot Charts")

    # Main content
    if plot_button and ticker:
        with st.spinner("Fetching data..."):
            # Fetch data
            df = fetch_and_process_data(ticker, period, interval)
            if df is not None:
                # Super Zones Chart
                st.subheader("Super Zones Chart")
                zones = identify_super_zones(ticker, period, interval)
                fig1 = plot_super_zones_chart(df, zones, ticker, period, interval)
                if fig1:
                    st.pyplot(fig1)
                    plt.close(fig1)
                else:
                    st.error("Failed to render Super Zones chart.")
                
                # Base Candles Chart
                st.subheader("Base Candles Chart")
                base_rally_candles, base_drop_candles = identify_base_and_following_candles(df)
                fig2 = plot_candlestick_chart(df, base_rally_candles, base_drop_candles, 
                                             f"{ticker} - {period}/{interval} (Base Candles)")
                if fig2:
                    st.pyplot(fig2)
                    plt.close(fig2)
                else:
                    st.error("Failed to render Base Candles chart.")

                # Notifications
                if zones and email and sender_email and sender_password:
                    signals = []
                    last_candle = df.iloc[-1]
                    for zone in zones:
                        limit_price = zone['level']
                        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
                        if side == 'BUY' and last_candle['low'] >= limit_price * 0.99 and last_candle['low'] <= limit_price * 1.01:
                            signals.append(f"{side} signal at {limit_price:.2f}")
                        elif side == 'SELL' and last_candle['high'] >= limit_price * 0.99 and last_candle['high'] <= limit_price * 1.01:
                            signals.append(f"{side} signal at {limit_price:.2f}")
                    if signals:
                        body = f"Signals for {ticker} ({period}/{interval}):\n" + "\n".join(signals)
                        send_email(f"Stock Signals for {ticker}", body, email, smtp_server, smtp_port, 
                                  sender_email, sender_password)

    elif plot_button:
        st.error("Please enter a valid ticker.")

if __name__ == "__main__":
    main()
