import yfinance as yf
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas_ta as ta
import matplotlib
import requests
NEWS_API_KEY = '8bfe91331abd470fb6fadb46ab8321c2'

# Set matplotlib to use 'Agg' backend to avoid GUI errors on macOS
matplotlib.use('Agg')

app = Flask(__name__)

# Function to fetch real-time 5-minute interval data and plot
def plot_realtime(ticker, interval="5m"):
    stock = yf.Ticker(ticker)
    data = stock.history(interval=interval, period="1d")

    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_title(f"{ticker} - 5 Min Interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()

    # Save the plot as an image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    plt.close()  # Close the plot to avoid overlapping
    return graph_url

# Function to fetch monthly data and plot with a trendline
def plot_monthly_trend(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(interval="1mo", period="5y")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Monthly Close Price')

    # Trendline
    z = np.polyfit(range(len(data['Close'])), data['Close'], 1)
    p = np.poly1d(z)
    ax.plot(data.index, p(range(len(data['Close']))), linestyle='--', color='red', label='Trendline')

    ax.set_title(f"{ticker} - Monthly Trendline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()

    # Save the plot as an image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    plt.close()  # Close the plot to avoid overlapping
    return graph_url

# Function to calculate RSI, MACD, and provide recommendation
def calculate_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")

    close_prices = data['Close']

    # Calculate RSI, MACD, ADX, Bollinger Bands, EMA using pandas_ta
    rsi = ta.rsi(close_prices, length=14)
    macd = ta.macd(close_prices, fast=12, slow=26, signal=9)
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
    bollinger = ta.bbands(close_prices, length=20)
    ema_50 = ta.ema(close_prices, length=50)
    ema_200 = ta.ema(close_prices, length=200)

    latest_rsi = rsi.iloc[-1] if rsi is not None and not rsi.empty else None
    latest_macd = macd['MACD_12_26_9'].iloc[-1] if macd is not None and not macd.empty else None
    latest_macd_signal = macd['MACDs_12_26_9'].iloc[-1] if macd is not None and not macd.empty else None
    macd_diff = latest_macd - latest_macd_signal if latest_macd is not None and latest_macd_signal is not None else None
    latest_adx = adx['ADX_14'].iloc[-1] if adx is not None and not adx.empty else None
    latest_upper_bb = bollinger['BBU_20_2.0'].iloc[-1] if bollinger is not None and not bollinger.empty else None
    latest_lower_bb = bollinger['BBL_20_2.0'].iloc[-1] if bollinger is not None and not bollinger.empty else None
    latest_ema_50 = ema_50.iloc[-1] if ema_50 is not None and not ema_50.empty else None
    latest_ema_200 = ema_200.iloc[-1] if ema_200 is not None and not ema_200.empty else None

    # Support and Resistance calculation
    pivot = (data['High'].iloc[-1] + data['Low'].iloc[-1] + data['Close'].iloc[-1]) / 3
    resistance_1 = 2 * pivot - data['Low'].iloc[-1]
    support_1 = 2 * pivot - data['High'].iloc[-1]

    # Provide recommendation based on a scoring system
    score = 0

    # RSI Score
    if latest_rsi is not None:
        if latest_rsi < 20:
            score += 2
        elif latest_rsi < 30:
            score += 1
        elif latest_rsi > 80:
            score -= 2
        elif latest_rsi > 70:
            score -= 1

    # MACD Score
    if macd_diff is not None:
        if macd_diff > 0.5:
            score += 2
        elif macd_diff > 0:
            score += 1
        elif macd_diff < -0.5:
            score -= 2
        elif macd_diff < 0:
            score -= 1

    # ADX Score
    if latest_adx is not None and latest_adx > 25:
        score += 1

    # Bollinger Bands Score
    if latest_upper_bb is not None and latest_lower_bb is not None:
        if close_prices.iloc[-1] > latest_upper_bb:
            score -= 1  # Overbought
        elif close_prices.iloc[-1] < latest_lower_bb:
            score += 1  # Oversold

    # EMA Crossover Score
    if latest_ema_50 is not None and latest_ema_200 is not None:
        if latest_ema_50 > latest_ema_200:
            score += 1  # Bullish signal
        elif latest_ema_50 < latest_ema_200:
            score -= 1  # Bearish signal

    # Provide recommendation based on score
    if score >= 5:
        recommendation = "Strong Buy"
    elif score >= 3:
        recommendation = "Buy"
    elif score >= 0:
        recommendation = "Neutral"
    elif score >= -2:
        recommendation = "Sell"
    else:
        recommendation = "Strong Sell"

    return {
        'latest_rsi': latest_rsi,
        'latest_macd': latest_macd,
        'latest_macd_signal': latest_macd_signal,
        'latest_adx': latest_adx,
        'latest_upper_bb': latest_upper_bb,
        'latest_lower_bb': latest_lower_bb,
        'latest_ema_50': latest_ema_50,
        'latest_ema_200': latest_ema_200,
        'resistance_level': resistance_1,
        'support_level': support_1,
        'recommendation': recommendation,
        'score': score
    }

# Function to calculate and plot RSI with SMA overlay
def plot_rsi_with_sma(ticker):
    # Fetch stock data
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")
    
    # Calculate RSI
    close_prices = data['Close']
    rsi = ta.rsi(close_prices, length=14)
    
    # Calculate SMA on RSI (SMA 10 in this case, you can adjust as needed)
    sma_rsi = rsi.rolling(window=10).mean() if rsi is not None and not rsi.empty else None

    # Plot RSI
    fig, ax = plt.subplots(figsize=(10, 6))
    if rsi is not None and not rsi.empty:
        ax.plot(rsi.index, rsi, label='RSI', color='purple')
    if sma_rsi is not None and not sma_rsi.empty:
        ax.plot(sma_rsi.index, sma_rsi, label='SMA of RSI', color='orange')

    # Add horizontal lines for overbought and oversold levels
    ax.axhline(70, linestyle='--', alpha=0.5, color='gray')
    ax.axhline(30, linestyle='--', alpha=0.5, color='gray')

    # Add labels and title
    ax.set_title(f'{ticker} - RSI with SMA Overlay')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()

    # Save the plot to memory to display in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    plt.close()  # Close the plot to free up resources
    return graph_url

def fetch_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
    try:
        response = requests.get(url)
        news_data = response.json()
        if news_data.get("status") == "ok":
            articles = news_data.get("articles")
            news = []
            for article in articles[:3]:  # Limit to 5 articles
                news.append({
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'published_at': article['publishedAt']
                })
            return news
        else:
            return []
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_info = None
    technicals = None
    news = []
    graph_realtime_url = None
    graph_monthly_url = None
    rsi_graph = None
    
    if request.method == 'POST':
        # use try catch block to handle errors
        try:
            ticker = request.form['ticker']
            stock = yf.Ticker(ticker)
            info = stock.info
        
            stock_info = {
                'symbol': ticker,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE'),
                'div_yield': info.get('dividendYield'),
                'open_price': info.get('open'),
                'high_price': info.get('dayHigh'),
                'low_price': info.get('dayLow'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'current_price': info.get('regularMarketPrice', info.get('currentPrice')),
                'volume': info.get('volume'),
                'eps': info.get('trailingEps'),
                'employees': info.get('fullTimeEmployees'),
                'sector': info.get('sector'),
                'description': info.get('longBusinessSummary')

            }
            # Fetch News related to the stock
            news = fetch_news(ticker)

            # Calculate RSI, MACD, ADX, Bollinger Bands, EMA, and recommendation
            technicals = calculate_technical_indicators(ticker)

            # Generate real-time and monthly trendline graphs
            graph_realtime_url = plot_realtime(ticker)
            graph_monthly_url = plot_monthly_trend(ticker)
            rsi_graph = plot_rsi_with_sma(ticker)
        except Exception as e:
            return render_template('index.html')

    return render_template('index.html', news=news, rsi_graph=rsi_graph, stock_info=stock_info, technicals=technicals, graph_realtime_url=graph_realtime_url, graph_monthly_url=graph_monthly_url)



if __name__ == '__main__':
    app.run(debug=True)