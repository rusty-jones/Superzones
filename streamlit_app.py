<!-- ... (Previous code remains unchanged until plot_chart) ... -->

# Plotting function
def plot_chart(df, zones, symbol, timeframe, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, aligned_zones):
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
        color = 'navy' if side == 'BUY' else 'darkred'  # Darker colors
        linewidth = 1
        alpha = 0.5

        for az in aligned_zones.get(symbol, []):
            if az and isinstance(az, dict) and 'level' in az and 'type' in az:
                if abs(az['level'] - limit_price) / limit_price < 0.01 and az['type'] == zone['type']:
                    linewidth = 2
                    alpha = 0.8
                    break

        if show_limit_lines:
            ax[0].axhline(y=limit_price, color=color, linestyle='--', alpha=alpha, linewidth=linewidth)
        if show_prices:
            ax[0].text(len(df) - 1, limit_price, f'{limit_price:.2f}', ha='right', va='center', fontsize=10, color=color)

    return fig

<!-- ... (Rest of the code remains unchanged) ... -->
