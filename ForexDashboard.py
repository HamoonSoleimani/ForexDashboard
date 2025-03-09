
#!/usr/bin/env python3
"""
A complete robust trading robot that gathers data from multiple sources:
  - OANDA Order Book and Position Book (using oandapyV20)
  - Retail sentiment and Twitter sentiment (via scraping forexbenchmark.com)
  - Myfxbook sentiment (via scraping myfxbook.com)

With an enhanced graphical interface that visualizes trading decisions.
"""

import requests
import json
import time
import re
import logging
import sys
import threading
import traceback
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import io  # Import the io module

# Import OANDA API client
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
from oandapyV20.exceptions import V20Error

# -------------------------
# CONFIGURATION
# -------------------------
# Replace these with your valid credentials
API_KEY = ""  # Replace with your actual API key
ACCOUNT_ID = ""  # Replace with your actual account ID
ENVIRONMENT = "practice"  # "practice" or "live"

# -------------------------
# LOGGING CONFIGURATION
# -------------------------
class LogHandler(logging.Handler):
    """Custom logging handler to display logs in a Tkinter text widget."""
    def __init__(self, text_widget: scrolledtext.ScrolledText):
        super().__init__()
        self.text_widget = text_widget
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    def emit(self, record: logging.LogRecord):
        msg = self.formatter.format(record)
        def append():
            self.text_widget.configure(state='normal')
            if record.levelno >= logging.ERROR:
                self.text_widget.tag_config('error', foreground='red')
                self.text_widget.insert(tk.END, msg + '\n', 'error')
            elif record.levelno >= logging.WARNING:
                self.text_widget.tag_config('warning', foreground='orange')
                self.text_widget.insert(tk.END, msg + '\n', 'warning')
            else:
                self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)
        self.text_widget.after(0, append)

# -------------------------
# DATA STRUCTURES
# -------------------------
class TradingSignal:
    """Class to represent a trading signal with its components and history."""
    def __init__(self, instrument: str):
        self.instrument = instrument
        self.timestamp = datetime.now()
        self.total_score = 0.0
        self.order_book_score = 0
        self.position_book_score = 0
        self.pair_sentiment_score = 0
        self.currency_sentiment_score = 0
        self.retail_profit_score = 0
        self.decision = "No Trade"
        self.price = 0.0
    
    def __str__(self) -> str:
        return f"{self.instrument} {self.decision} (Score: {self.total_score:.2f})"
    
    @property
    def is_long(self) -> bool:
        return self.decision in ["Bullish", "Strong Bullish"]
    
    @property
    def is_short(self) -> bool:
        return self.decision in ["Bearish", "Strong Bearish"]
    
    @property
    def is_actionable(self) -> bool:
        return self.is_long or self.is_short

class SignalHistory:
    """Class to manage signal history for instruments."""
    def __init__(self, max_history: int = 50):
        self.signals: Dict[str, List[TradingSignal]] = {}
        self.max_history = max_history
    
    def add_signal(self, signal: TradingSignal) -> None:
        if signal.instrument not in self.signals:
            self.signals[signal.instrument] = []
        
        self.signals[signal.instrument].append(signal)
        
        # Keep only the latest max_history signals
        if len(self.signals[signal.instrument]) > self.max_history:
            self.signals[signal.instrument] = self.signals[signal.instrument][-self.max_history:]
    
    def get_signals(self, instrument: str, count: int = None) -> List[TradingSignal]:
        if instrument not in self.signals:
            return []
        
        if count is None:
            return self.signals[instrument]
        
        return self.signals[instrument][-count:]
    
    def get_all_instruments(self) -> List[str]:
        return sorted(list(self.signals.keys()))
    
    def get_latest_signal(self, instrument: str) -> Optional[TradingSignal]:
        signals = self.get_signals(instrument)
        if not signals:
            return None
        return signals[-1]
    
    def get_signal_history_as_df(self, instrument: str) -> pd.DataFrame:
        signals = self.get_signals(instrument)
        if not signals:
            return pd.DataFrame()
        
        data = []
        for signal in signals:
            data.append({
                'timestamp': signal.timestamp,
                'total_score': signal.total_score,
                'order_book_score': signal.order_book_score,
                'position_book_score': signal.position_book_score,
                'pair_sentiment_score': signal.pair_sentiment_score,
                'currency_sentiment_score': signal.currency_sentiment_score,
                'retail_profit_score': signal.retail_profit_score,
                'decision': signal.decision,
                'price': signal.price
            })
        
        return pd.DataFrame(data)

# -------------------------
# OANDA DATA FETCHER
# -------------------------
class OandaDataFetcher:
    def __init__(self, api_key: str = API_KEY, environment: str = ENVIRONMENT, account_id: str = ACCOUNT_ID):
        self.api = API(access_token=api_key, environment=environment)
        self.account_id = account_id

    def fetch_orderbook(self, instrument: str) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}  # Explicitly type params
        req = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
        try:
            response = self.api.request(req)
            print(f"Orderbook response for {instrument}: {response}")  # Debug print  <-- ADD THIS
            orderbook = response.get("orderBook", {})
            buckets = orderbook.get("buckets", [])
            time_str = orderbook.get("time", "N/A")
            price = orderbook.get("price", "N/A")
            bucket_width = orderbook.get("bucketWidth", "N/A")
            return {"buckets": buckets, "time": time_str, "price": price, "bucket_width": bucket_width}
        except V20Error as e:
            logging.error("Error fetching orderbook for %s: %s", instrument, e)
            return None
        except Exception as e:
            logging.error("Unexpected error fetching orderbook for %s: %s", instrument, e)
            logging.debug(traceback.format_exc())
            return None

    def fetch_positionbook(self, instrument: str) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}  # Explicitly type params
        req = instruments.InstrumentsPositionBook(instrument=instrument, params=params)
        try:
            response = self.api.request(req)
            print(f"Positionbook response for {instrument}: {response}")  # Debug print <-- ADD THIS
            positionbook = response.get("positionBook", {})
            buckets = positionbook.get("buckets", [])
            time_str = positionbook.get("time", "N/A")
            price = positionbook.get("price", "N/A")
            bucket_width = positionbook.get("bucketWidth", "N/A")
            return {"buckets": buckets, "time": time_str, "price": price, "bucket_width": bucket_width}
        except V20Error as e:
            logging.error("Error fetching positionbook for %s: %s", instrument, e)
            return None
        except Exception as e:
            logging.error("Unexpected error fetching positionbook for %s: %s", instrument, e)
            logging.debug(traceback.format_exc())
            return None

    def fetch_price(self, instrument: str) -> Optional[float]:
        params = {"instruments": instrument}
        req = pricing.PricingInfo(accountID=self.account_id, params=params)
        try:
            response = self.api.request(req)
            print(f"Price response for {instrument}: {response}")  # Add this
            prices = response.get("prices", [])
            if prices and len(prices) > 0:
                price_data = prices[0]
                asks = price_data.get("asks", [])
                bids = price_data.get("bids", [])
                if asks and bids:
                    ask_price = float(asks[0].get("price", 0))
                    bid_price = float(bids[0].get("price", 0))
                    return (ask_price + bid_price) / 2
            return None
        except V20Error as e:
            logging.error("Error fetching price for %s: %s", instrument, e)
            return None
        except Exception as e:
            logging.error("Unexpected error fetching price for %s: %s", instrument, e)
            logging.debug(traceback.format_exc())
            return None

    def parse_buckets(self, buckets: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
        prices = []
        short_counts = []
        long_counts = []
        for bucket in buckets:
            try:
                price = float(bucket.get("price", 0))
                short_pct = float(bucket.get("shortCountPercent", 0))
                long_pct = float(bucket.get("longCountPercent", 0))
                prices.append(price)
                short_counts.append(short_pct)
                long_counts.append(long_pct)
            except Exception as e:
                logging.error("Error parsing bucket data: %s", e)
        return prices, short_counts, long_counts

    def filter_buckets(self, prices: List[float], short_counts: List[float], long_counts: List[float], 
                       current_price: float, threshold: float = 0.03) -> Tuple[List[float], List[float], List[float]]:
        lower_bound = current_price * (1 - threshold)
        upper_bound = current_price * (1 + threshold)
        filtered = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                    if lower_bound <= p <= upper_bound]
        if filtered:
            fp, fs, fl = zip(*filtered)
            return list(fp), list(fs), list(fl)
        return [], [], []

    def fetch_account_summary(self) -> Dict[str, Any]:
        """Fetch account summary information from OANDA."""
        req = accounts.AccountSummary(accountID=self.account_id)
        try:
            response = self.api.request(req)
            print(f"Account summary response: {response}")  # Add this
            return response.get("account", {})
        except V20Error as e:
            logging.error("Error fetching account summary: %s", e)
            return {}
        except Exception as e:
            logging.error("Unexpected error fetching account summary: %s", e)
            logging.debug(traceback.format_exc())
            return {}

# -------------------------
# SENTIMENT DATA FETCHER
# -------------------------
class SentimentDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        # Add cache to minimize redundant requests
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes

    def fetch_data(self, url: str, retries: int = 3, timeout: int = 15) -> Optional[str]:
        # Check cache first
        now = time.time()
        if url in self.cache and self.cache_expiry.get(url, 0) > now:
            return self.cache[url]
        
        for i in range(retries):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                
                # Store in cache
                self.cache[url] = response.text
                self.cache_expiry[url] = now + self.cache_duration
                
                return response.text
            except requests.exceptions.RequestException as e:
                logging.error("Error fetching %s (attempt %d/%d): %s", url, i+1, retries, e)
                time.sleep(2)
        
        return None

    def fetch_retail_positions_data(self) -> Optional[str]:
        url = "https://forexbenchmark.com/quant/retail_positions/"
        return self.fetch_data(url)

    def extract_pie_chart_data(self, html_content: Optional[str]) -> Dict[str, str]:
        if not html_content:
            return {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tags = soup.find_all('script')
        pie_chart_data = {}
        
        for script in script_tags:
            script_text = script.string
            if not script_text:
                continue
            
            if "mpld3.draw_figure" in script_text and "fig_el" in script_text and "paths" in script_text and "texts" in script_text:
                try:
                    json_match = re.search(r'mpld3\.draw_figure\("fig_el[^"]*",\s*({.*?})\);', script_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        data = json.loads(json_str)
                        if 'axes' in data and data['axes']:
                            axes_data = data['axes'][0]
                            if 'texts' in axes_data:
                                texts = axes_data['texts']
                                for idx, text_obj in enumerate(texts):
                                    if 'text' in text_obj and '%' in text_obj['text']:
                                        percentage = text_obj['text']
                                        if idx > 0 and 'text' in texts[idx-1]:
                                            currency = texts[idx-1]['text']
                                            pie_chart_data[currency] = percentage
                                return pie_chart_data
                except Exception as e:
                    logging.error("Error extracting pie chart data: %s", e)
                    logging.debug(traceback.format_exc())
        
        return pie_chart_data

    def fetch_twitter_sentiment_data(self) -> Optional[str]:
        url = "https://forexbenchmark.com/quant/twitter/"
        return self.fetch_data(url)

    def extract_twitter_chart_data(self, html_content: Optional[str]) -> Dict[str, str]:
        if not html_content:
            return {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tags = soup.find_all('script')
        chart_data = {}
        
        for script in script_tags:
            script_text = script.string
            if not script_text:
                continue
            
            if "mpld3.draw_figure" in script_text and "fig_el" in script_text and "lines" in script_text and "texts" in script_text:
                try:
                    json_match = re.search(r'mpld3\.draw_figure\("fig_el[^"]*",\s*({.*?})\);', script_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        data = json.loads(json_str)
                        if 'axes' in data and data['axes']:
                            axes_data = data['axes'][0]
                            if 'texts' in axes_data:
                                texts = axes_data['texts']
                                currencies = []
                                for text_obj in texts:
                                    if ('text' in text_obj and 'position' in text_obj and
                                        0.4 < text_obj['position'][1] < 1.0 and
                                        len(text_obj['text']) == 6):
                                        currencies.append(text_obj['text'])
                                if 'lines' in axes_data:
                                    lines = axes_data['lines']
                                    for i, line_obj in enumerate(lines[:len(currencies)]):
                                        if 'color' in line_obj:
                                            if i < len(currencies):
                                                currency = currencies[i]
                                                chart_data[currency] = line_obj['color']
                                    return chart_data
                except Exception as e:
                    logging.error("Error extracting Twitter sentiment chart data: %s", e)
                    logging.debug(traceback.format_exc())
        
        return chart_data

    def fetch_myfxbook_sentiment_data(self) -> Optional[str]:
        url = "https://www.myfxbook.com/community/outlook"
        return self.fetch_data(url)

    def extract_myfxbook_table_data(self, html_content: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        if not html_content:
            return None

        try:
            # Use StringIO to wrap the HTML string
            dfs = pd.read_html(io.StringIO(html_content), attrs={'id': 'outlookSymbolsTable'})
            if dfs:
                df = dfs[0]
                df.dropna(axis=1, how='all', inplace=True)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df.columns = [str(col).strip() for col in df.columns]
                return df.to_dict(orient='records')
        except Exception as e:
            logging.error("Error extracting Myfxbook sentiment table: %s", e)
            logging.debug(traceback.format_exc())

        return None

# -------------------------
# TRADING STRATEGY (RULE-BASED DECISION ENGINE)
# -------------------------
class TradingStrategy:
    def __init__(self, oanda_fetcher: OandaDataFetcher, sentiment_fetcher: SentimentDataFetcher):
        self.oanda_fetcher = oanda_fetcher
        self.sentiment_fetcher = sentiment_fetcher
        # Hardcoded valid instruments list
        self.instruments = ['EUR_USD', 'NZD_USD', 'USD_CHF', 'AUD_USD', 'GBP_CHF', 
                           'EUR_JPY', 'USD_JPY', 'EUR_CHF', 'GBP_USD', 'GBP_JPY', 
                           'EUR_GBP', 'AUD_JPY', 'USD_CAD', 'EUR_AUD']
        logging.info(f"Using hardcoded valid instruments: {self.instruments}")

        # Weights per procedure:
        self.weights = {
            'order_book': 0.30,  # Changed to 30%
            'position_book': 0.225, # Changed to 22.5%
            'pair_sentiment': 0.225, # Changed to 22.5%
            'currency_sentiment': 0.15, # Changed to 15%
            'retail_profitability': 0.10 # Changed to 10%
        }
        
        # Signal thresholds
        self.thresholds = {
            'strong_bullish': 1.0,
            'bullish': 0.7, 
            'bearish': -0.7,
            'strong_bearish': -1.0
        }

    def pips_to_price_diff(self, instrument: str, pips: float) -> float:
        """Converts pips to a price difference."""
        if instrument.endswith("_JPY") or instrument == 'XAU_USD':  # JPY pairs and gold
            pip_value = 0.01
        else:  # Most others
            pip_value = 0.0001
        return pips * pip_value

    def analyze_orderbook(self, instrument: str) -> int:
        orderbook_data = self.oanda_fetcher.fetch_orderbook(instrument)
        if not orderbook_data or 'buckets' not in orderbook_data:
            return 0
        
        buckets = orderbook_data['buckets']
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)

        try:
            current_price = float(orderbook_data['price'])
        except (ValueError, TypeError):
            logging.warning(f"Invalid price in orderbook for {instrument}: {orderbook_data.get('price', 'N/A')}")
            return 0  # Can't proceed without current price

        price_diff_50_pips = self.pips_to_price_diff(instrument, 50)
        lower_bound = current_price - price_diff_50_pips
        upper_bound = current_price + price_diff_50_pips

        # Filter buckets within 50 pips
        filtered_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                           if lower_bound <= p <= upper_bound]
        if not filtered_buckets:
            return 0

        fp, fs, fl = zip(*filtered_buckets)
        total_long = sum(fl)
        total_short = sum(fs)

        # Calculate percentage difference
        if total_short > 0:  # Avoid division by zero
            long_short_diff_pct = ((total_long - total_short) / total_short) * 100
        else:
            long_short_diff_pct = 0 if total_long == 0 else float('inf')  # If no shorts, long dominance is infinite

        if total_long > 0:
            short_long_diff_pct = ((total_short - total_long) / total_long) * 100
        else:
            short_long_diff_pct = 0 if total_short == 0 else float('inf')

        score = 0

        # 1- Sell orders below market (stop orders) = Potential Stop Loss Level for Long Positions/ Potential Entry for Sell Stop orders
        # 2- Buy orders above market (buy orders) = Potential Stop Loss Level for Short Positions/ Potential Entry for Buy Stop orders
        # These are not directly actionable signals, but informational

        # 3- Sell Orders above market = Spotting Potential Resistance with Sell Limit Orders
        # 4- Buy Orders below market = Spotting Potential Support with Buy Limit Orders

        if long_short_diff_pct >= 20:
            score = 1  # Bullish: Long orders 20% higher than short orders within 50 pips
        elif short_long_diff_pct >= 20:
            score = -1 # Bearish: Short orders 20% higher than long orders within 50 pips

        return score

    def analyze_positionbook(self, instrument: str) -> int:
        positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)
        if not positionbook_data or 'buckets' not in positionbook_data:
            return 0
        
        buckets = positionbook_data['buckets']
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)

        try:
            current_price = float(positionbook_data['price'])
        except (ValueError, TypeError):
            logging.warning(f"Invalid price in positionbook for {instrument}: {positionbook_data.get('price', 'N/A')}")
            return 0

        price_diff_50_pips = self.pips_to_price_diff(instrument, 50)
        lower_bound = current_price - price_diff_50_pips
        upper_bound = current_price + price_diff_50_pips

        # Filter buckets within 50 pips
        filtered_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                             if lower_bound <= p <= upper_bound]
        if not filtered_buckets:
            return 0

        fp, fs, fl = zip(*filtered_buckets)
        total_long = sum(fl)
        total_short = sum(fs)
        total_positions = total_long + total_short

        score = 0

        if total_positions > 0:  # Avoid division by zero
            long_ratio = (total_long / total_positions) * 100
            short_ratio = (total_short / total_positions) * 100

            # 1- Identifying a Potential Top with a Large Number of Long Positions
            #    (Contrarian: If many traders are long, it's a potential reversal of long/bearish signal)
            if long_ratio >= 65:
                score = -1

            # 2- Spotting Potential Support with a Large Number of Short Positions
            #    (Contrarian: If many traders are short, it's a potential reversal of short/bullish signal)
            elif short_ratio >= 65:
                score = 1

        return score

    def analyze_pair_specific_sentiment(self, instrument: str) -> int:
        score = 0
        pair = instrument.replace('_', '').upper()  # e.g., EUR_USD -> EURUSD
        # Use Myfxbook sentiment data for pair-specific analysis
        html = self.sentiment_fetcher.fetch_myfxbook_sentiment_data()
        records = self.sentiment_fetcher.extract_myfxbook_table_data(html)
        
        if records:
            for record in records:
                if "Symbol" in record and record["Symbol"].upper() == pair:
                    try:
                        long_pct_str = record.get("Long %", "0%")
                        if isinstance(long_pct_str, str):
                            long_pct_str = long_pct_str.strip('%')
                        long_pct = float(long_pct_str)
                        
                        if long_pct < 40.0:
                            score = 2    # strong bullish contrarian signal
                        elif long_pct > 60.0:
                            score = -2   # bearish contrarian signal
                        break
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Error parsing sentiment data for {instrument}: {e}")
        
        return score

    def analyze_currency_sentiment(self) -> int:
        score = 0
        html = self.sentiment_fetcher.fetch_retail_positions_data()
        pie_data = self.sentiment_fetcher.extract_pie_chart_data(html)
        
        if pie_data:
            sentiments = []
            for currency, perc in pie_data.items():
                try:
                    if isinstance(perc, str):
                        perc = perc.strip('%')
                    val = float(perc)
                    sentiments.append(val)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error parsing currency sentiment: {e}")
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                # If overall long % is above 50 then assign a bullish signal, otherwise bearish.
                if avg_sentiment > 50:
                    score = 1
                elif avg_sentiment < 50:
                    score = -1
        
        return score

    def analyze_retail_profitability(self) -> int:
        score = 0
        # Use Twitter sentiment as a proxy for retail profit-taking signals
        html = self.sentiment_fetcher.fetch_twitter_sentiment_data()
        twitter_chart = self.sentiment_fetcher.extract_twitter_chart_data(html)
        
        if twitter_chart:
            # Map certain color codes to bullish or bearish signals.
            positive_colors = ["#91DB57", "#57DB80", "#57D3DB", "#5770DB", "#A157DB"]
            negative_colors = ["#DB5F57", "#DBC257"]
            bullish = 0
            bearish = 0
            
            for color in twitter_chart.values():
                if color in positive_colors:
                    bullish += 1
                elif color in negative_colors:
                    bearish += 1
            
            if bullish > bearish:
                score = 1
            elif bearish > bullish:
                score = -1
        
        return score

    def compute_total_score(self, instrument: str) -> Tuple[float, Dict[str, float]]:
        order_score = self.analyze_orderbook(instrument)
        position_score = self.analyze_positionbook(instrument)
        pair_sentiment_score = self.analyze_pair_specific_sentiment(instrument)
        currency_sentiment_score = self.analyze_currency_sentiment()
        retail_profit_score = self.analyze_retail_profitability()

        weighted_total = (order_score * self.weights['order_book'] +
                          position_score * self.weights['position_book'] +
                          pair_sentiment_score * self.weights['pair_sentiment'] +
                          currency_sentiment_score * self.weights['currency_sentiment'] +
                          retail_profit_score * self.weights['retail_profitability'])
        
        details = {
            "order_score": order_score,
            "position_score": position_score,
            "pair_sentiment_score": pair_sentiment_score,
            "currency_sentiment_score": currency_sentiment_score,
            "retail_profit_score": retail_profit_score
        }
        
        return weighted_total, details

    def decide_trade(self) -> Dict[str, Dict[str, Any]]:
        decisions = {}
        
        for instrument in self.instruments:
            try:
                total_score, details = self.compute_total_score(instrument)
                current_price = self.oanda_fetcher.fetch_price(instrument)
                
                # Create a trading signal
                signal = TradingSignal(instrument)
                signal.total_score = total_score
                signal.order_book_score = details['order_score']
                signal.position_book_score = details['position_score']
                signal.pair_sentiment_score = details['pair_sentiment_score']
                signal.currency_sentiment_score = details['currency_sentiment_score']
                signal.retail_profit_score = details['retail_profit_score']
                
                if current_price:
                    signal.price = current_price
                
                # Determine decision based on thresholds
                if total_score > self.thresholds['strong_bullish']:
                    signal.decision = "Strong Bullish"
                elif total_score > self.thresholds['bullish']:
                    signal.decision = "Bullish"
                elif total_score < self.thresholds['strong_bearish']:
                    signal.decision = "Strong Bearish"
                elif total_score < self.thresholds['bearish']:
                    signal.decision = "Bearish"
                else:
                    signal.decision = "No Trade"
                
                decisions[instrument] = {
                    "instrument": instrument,
                    "decision": signal.decision,
                    "total_score": total_score,
                    "details": details,
                    "price": signal.price,
                    "signal": signal
                }
                
            except Exception as e:
                logging.error(f"Error deciding trade for {instrument}: {e}")
                logging.debug(traceback.format_exc())
                # Add a default/error entry
                decisions[instrument] = {
                    "instrument": instrument,
                    "decision": "Error",
                    "total_score": 0.0,
                    "details": {
                        "order_score": 0,
                        "position_score": 0,
                        "pair_sentiment_score": 0,
                        "currency_sentiment_score": 0,
                        "retail_profit_score": 0
                    },
                    "price": 0.0,
                    "signal": None
                }
        
        return decisions

# -------------------------
# GRAPHICAL USER INTERFACE
# -------------------------
class TradingRobotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Forex Trading Robot Dashboard")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 768)
        self.root.configure(bg="#2E3B4E")

        # Set the window icon and style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure ttk styles
        self.style.configure('TFrame', background='#2E3B4E')
        self.style.configure('TLabel', background='#2E3B4E', foreground='white', font=('Segoe UI', 10))
        self.style.configure('TButton', background='#4A6491', foreground='white', font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#FFFFFF')
        self.style.configure('Status.TLabel', font=('Segoe UI', 10), foreground='#AAAAAA')
        self.style.configure('Treeview', background='#374B61', foreground='white', fieldbackground='#374B61', font=('Segoe UI', 9))
        self.style.configure('Treeview.Heading', background='#2E3B4E', foreground='white', font=('Segoe UI', 10, 'bold'))
        self.style.map('Treeview', background=[('selected', '#4A6491')])
        
        # Add notebook style
        self.style.configure('TNotebook', background='#2E3B4E', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#374B61', foreground='white', padding=[10, 2])
        self.style.map('TNotebook.Tab', background=[('selected', '#4A6491')])

        # Create header frame *FIRST*
        self.create_header_frame()

        # Initialize trading components
        self.initialize_trading_components()

        # Initialize data history for charts
        self.history = SignalHistory(max_history=100)
        
        # Selected instrument for detailed analysis - MOVED BEFORE create_main_frame()
        self.selected_instrument = tk.StringVar()
        self.selected_instrument.set(self.trading_strategy.instruments[0])
        self.selected_instrument.trace_add("write", self.on_instrument_selected)

        # Create main frames
        self.create_main_frame()
        self.create_footer_frame()

        # Start the trading thread
        self.start_trading_thread()

        # *** ADD THIS LINE TO INITIALLY LOAD DATA ***
        self.refresh_data()
        
    def create_header_frame(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Logo and title
        title_label = ttk.Label(header_frame, text="Forex Trading Robot", style='Header.TLabel')
        title_label.pack(side=tk.LEFT, padx=5)

        # Account info
        self.account_label = ttk.Label(header_frame, text="Account: Loading...", style='Status.TLabel')
        self.account_label.pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = ttk.Label(header_frame, text="Initializing...", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Add a separator
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)

    def create_main_frame(self):
        # Create a notebook (tabbed interface) for multiple windows
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create 6 tabs as requested
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.detailed_analysis_tab = ttk.Frame(self.notebook)
        self.signal_history_tab = ttk.Frame(self.notebook)
        self.order_book_tab = ttk.Frame(self.notebook)
        self.position_book_tab = ttk.Frame(self.notebook)
        self.sentiment_analysis_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.detailed_analysis_tab, text="Detailed Analysis")
        self.notebook.add(self.signal_history_tab, text="Signal History")
        self.notebook.add(self.order_book_tab, text="Order Book")
        self.notebook.add(self.position_book_tab, text="Position Book")
        self.notebook.add(self.sentiment_analysis_tab, text="Sentiment Analysis")
        
        # Create content for each tab
        self.create_dashboard_tab()
        self.create_detailed_analysis_tab()
        self.create_signal_history_tab()
        self.create_order_book_tab()
        self.create_position_book_tab()
        self.create_sentiment_analysis_tab()

    def create_dashboard_tab(self):
        # Split into left and right frames
        left_frame = ttk.Frame(self.dashboard_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(self.dashboard_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create decisions table
        decisions_frame = ttk.LabelFrame(left_frame, text="Trade Decisions", padding=10)
        decisions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Create treeview for displaying trade decisions
        columns = ('Instrument', 'Decision', 'Score', 'Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit')
        self.decisions_tree = ttk.Treeview(decisions_frame, columns=columns, show='headings', height=10)

        # Configure columns
        for col in columns:
            self.decisions_tree.heading(col, text=col)
            width = 100 if col == 'Instrument' or col == 'Decision' else 70
            self.decisions_tree.column(col, width=width, anchor=tk.CENTER)

        # Add scrollbars
        scrollbar_y = ttk.Scrollbar(decisions_frame, orient=tk.VERTICAL, command=self.decisions_tree.yview)
        scrollbar_x = ttk.Scrollbar(decisions_frame, orient=tk.HORIZONTAL, command=self.decisions_tree.xview)
        self.decisions_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Pack treeview and scrollbars
        self.decisions_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind selection event
        self.decisions_tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        # Create charts frame
        charts_frame = ttk.LabelFrame(left_frame, text="Trading Signals History", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Create matplotlib figure for charts
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create details frame
        details_frame = ttk.LabelFrame(right_frame, text="Trade Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Create text widget for details
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, background='#374B61', foreground='white')
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.configure(state='disabled')

        # Create logs frame
        logs_frame = ttk.LabelFrame(right_frame, text="System Logs", padding=10)
        logs_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Create text widget for logs
        self.logs_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, background='#374B61', foreground='white')
        self.logs_text.pack(fill=tk.BOTH, expand=True)
        self.logs_text.configure(state='disabled')

        # Configure logging handler to display in logs text widget
        log_handler = LogHandler(self.logs_text)
        logger = logging.getLogger()
        logger.addHandler(log_handler)
        logger.setLevel(logging.INFO)

    def create_detailed_analysis_tab(self):
        # Create a frame for instrument selection
        control_frame = ttk.Frame(self.detailed_analysis_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for instrument selection
        self.instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument, 
                                             values=self.trading_strategy.instruments, state="readonly")
        self.instrument_combo.pack(side=tk.LEFT, padx=5)
        
        # Create refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh Analysis", command=self.refresh_detailed_analysis)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # Create a frame for the analysis charts
        charts_frame = ttk.Frame(self.detailed_analysis_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split into left and right frames for charts
        left_charts = ttk.Frame(charts_frame)
        left_charts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_charts = ttk.Frame(charts_frame)
        right_charts.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create frames for individual charts
        signal_history_frame = ttk.LabelFrame(left_charts, text="Signal History", padding=10)
        signal_history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        component_analysis_frame = ttk.LabelFrame(left_charts, text="Component Analysis", padding=10)
        component_analysis_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        orderbook_frame = ttk.LabelFrame(right_charts, text="Order Book Analysis", padding=10)
        orderbook_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        positionbook_frame = ttk.LabelFrame(right_charts, text="Position Book Analysis", padding=10)
        positionbook_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Create matplotlib figures for each chart
        self.signal_history_fig = Figure(figsize=(5, 3), dpi=100)
        self.signal_history_canvas = FigureCanvasTkAgg(self.signal_history_fig, master=signal_history_frame)
        self.signal_history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.component_analysis_fig = Figure(figsize=(5, 3), dpi=100)
        self.component_analysis_canvas = FigureCanvasTkAgg(self.component_analysis_fig, master=component_analysis_frame)
        self.component_analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.orderbook_fig = Figure(figsize=(5, 3), dpi=100)
        self.orderbook_canvas = FigureCanvasTkAgg(self.orderbook_fig, master=orderbook_frame)
        self.orderbook_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.positionbook_fig = Figure(figsize=(5, 3), dpi=100)
        self.positionbook_canvas = FigureCanvasTkAgg(self.positionbook_fig, master=positionbook_frame)
        self.positionbook_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_signal_history_tab(self):
        # Create a frame for controls
        control_frame = ttk.Frame(self.signal_history_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for instrument selection (reusing the same StringVar)
        signal_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument, 
                                              values=self.trading_strategy.instruments, state="readonly")
        signal_instrument_combo.pack(side=tk.LEFT, padx=5)
        
        # Add history length selector
        ttk.Label(control_frame, text="History Length:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.history_length_var = tk.StringVar(value="20")
        history_length_combo = ttk.Combobox(control_frame, textvariable=self.history_length_var,
                                           values=["10", "20", "50", "100"], width=5, state="readonly")
        history_length_combo.pack(side=tk.LEFT, padx=5)
        
        # Create refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh History", command=self.refresh_signal_history)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # Create a main frame for the history view
        main_frame = ttk.Frame(self.signal_history_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create left frame for chart
        chart_frame = ttk.LabelFrame(main_frame, text="Signal History Chart", padding=10)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create right frame for table
        table_frame = ttk.LabelFrame(main_frame, text="Signal History Table", padding=10)
        table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create matplotlib figure for history chart
        self.history_fig = Figure(figsize=(6, 4), dpi=100)
        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=chart_frame)
        self.history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for history table
        columns = ('Time', 'Decision', 'Total Score', 'Price', 'Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit')
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Configure columns
        for col in columns:
            self.history_tree.heading(col, text=col)
            width = 150 if col == 'Time' else 100 if col == 'Decision' else 70
            self.history_tree.column(col, width=width, anchor=tk.CENTER)
        
        # Add scrollbars
        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Pack treeview and scrollbars
        self.history_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    def create_order_book_tab(self):
        # Create a frame for controls
        control_frame = ttk.Frame(self.order_book_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for instrument selection (reusing the same StringVar)
        orderbook_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument, 
                                                 values=self.trading_strategy.instruments, state="readonly")
        orderbook_instrument_combo.pack(side=tk.LEFT, padx=5)
        
        # Create refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh Order Book", command=self.refresh_order_book)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # Create main frame for the order book
        main_frame = ttk.Frame(self.order_book_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create left frame for chart
        chart_frame = ttk.LabelFrame(main_frame, text="Order Book Visualization", padding=10)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create right frame for analysis
        analysis_frame = ttk.LabelFrame(main_frame, text="Order Book Analysis", padding=10)
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0, 5))
        
        # Create info frame for data
        info_frame = ttk.LabelFrame(main_frame, text="Raw Order Book Data", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(5, 0))
        
        # Create matplotlib figure for order book visualization
        self.order_book_fig = Figure(figsize=(6, 4), dpi=100)
        self.order_book_canvas = FigureCanvasTkAgg(self.order_book_fig, master=chart_frame)
        self.order_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for order book analysis
        self.order_book_analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD, 
                                                                 background='#374B61', foreground='white')
        self.order_book_analysis_text.pack(fill=tk.BOTH, expand=True)
        self.order_book_analysis_text.configure(state='disabled')
        
        # Create text widget for raw order book data
        self.order_book_raw_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, 
                                                           background='#374B61', foreground='white',
                                                           font=('Courier New', 9))
        self.order_book_raw_text.pack(fill=tk.BOTH, expand=True)
        self.order_book_raw_text.configure(state='disabled')

    def create_position_book_tab(self):
        # Create a frame for controls
        control_frame = ttk.Frame(self.position_book_tab)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for instrument selection (reusing the same StringVar)
        positionbook_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument, 
                                                    values=self.trading_strategy.instruments, state="readonly")
        positionbook_instrument_combo.pack(side=tk.LEFT, padx=5)
        
        # Create refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh Position Book", command=self.refresh_position_book)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # Create main frame for the position book
        main_frame = ttk.Frame(self.position_book_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create left frame for chart
        chart_frame = ttk.LabelFrame(main_frame, text="Position Book Visualization", padding=10)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create right frame for analysis
        analysis_frame = ttk.LabelFrame(main_frame, text="Position Book Analysis", padding=10)
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0, 5))
        
        # Create info frame for data
        info_frame = ttk.LabelFrame(main_frame, text="Raw Position Book Data", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(5, 0))
        
        # Create matplotlib figure for position book visualization
        self.position_book_fig = Figure(figsize=(6, 4), dpi=100)
        self.position_book_canvas = FigureCanvasTkAgg(self.position_book_fig, master=chart_frame)
        self.position_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for position book analysis
        self.position_book_analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD, 
                                                                    background='#374B61', foreground='white')
        self.position_book_analysis_text.pack(fill=tk.BOTH, expand=True)
        self.position_book_analysis_text.configure(state='disabled')
        
        # Create text widget for raw position book data
        self.position_book_raw_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, 
                                                              background='#374B61', foreground='white',
                                                              font=('Courier New', 9))
        self.position_book_raw_text.pack(fill=tk.BOTH, expand=True)
        self.position_book_raw_text.configure(state='disabled')

    def create_sentiment_analysis_tab(self):
        # Create main frame for sentiment analysis
        main_frame = ttk.Frame(self.sentiment_analysis_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Create refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh Sentiment Data", 
                                   command=self.refresh_sentiment_analysis)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # Split into three columns
        left_column = ttk.Frame(main_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        middle_column = ttk.Frame(main_frame)
        middle_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_column = ttk.Frame(main_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create frames for each sentiment source
        retail_frame = ttk.LabelFrame(left_column, text="Retail Positions (ForexBenchmark)", padding=10)
        retail_frame.pack(fill=tk.BOTH, expand=True)
        
        twitter_frame = ttk.LabelFrame(middle_column, text="Twitter Sentiment", padding=10)
        twitter_frame.pack(fill=tk.BOTH, expand=True)
        
        myfxbook_frame = ttk.LabelFrame(right_column, text="MyFxBook Sentiment", padding=10)
        myfxbook_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figures for visualizations
        self.retail_fig = Figure(figsize=(4, 6), dpi=100)
        self.retail_canvas = FigureCanvasTkAgg(self.retail_fig, master=retail_frame)
        self.retail_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.twitter_fig = Figure(figsize=(4, 6), dpi=100)
        self.twitter_canvas = FigureCanvasTkAgg(self.twitter_fig, master=twitter_frame)
        self.twitter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for MyFxBook data
        columns = ('Pair', 'Long %', 'Short %', 'Long Lots', 'Short Lots')
        self.myfxbook_tree = ttk.Treeview(myfxbook_frame, columns=columns, show='headings')
        
        # Configure columns
        for col in columns:
            self.myfxbook_tree.heading(col, text=col)
            self.myfxbook_tree.column(col, width=80, anchor=tk.CENTER)
        
        # Add scrollbars
        scrollbar_y = ttk.Scrollbar(myfxbook_frame, orient=tk.VERTICAL, command=self.myfxbook_tree.yview)
        scrollbar_x = ttk.Scrollbar(myfxbook_frame, orient=tk.HORIZONTAL, command=self.myfxbook_tree.xview)
        self.myfxbook_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Pack treeview and scrollbars
        self.myfxbook_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    def create_footer_frame(self):
        # Add a separator
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)

        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, padx=10, pady=10)

        # Add control buttons
        refresh_button = ttk.Button(footer_frame, text="Refresh Now", command=self.refresh_data)
        refresh_button.pack(side=tk.LEFT, padx=5)

        # Add update interval selector
        ttk.Label(footer_frame, text="Update Interval:").pack(side=tk.LEFT, padx=(20, 5))

        self.interval_var = tk.StringVar(value="60")
        interval_combobox = ttk.Combobox(footer_frame, textvariable=self.interval_var,
                                          values=["30", "60", "120", "300", "600"], width=5, state="readonly")
        interval_combobox.pack(side=tk.LEFT, padx=5)
        ttk.Label(footer_frame, text="seconds").pack(side=tk.LEFT, padx=5)

        # Add last update time
        self.last_update_label = ttk.Label(footer_frame, text="Last Update: Never", style='Status.TLabel')
        self.last_update_label.pack(side=tk.RIGHT, padx=5)

    def initialize_trading_components(self):
        try:
            self.oanda_fetcher = OandaDataFetcher()
            self.sentiment_fetcher = SentimentDataFetcher()
            self.trading_strategy = TradingStrategy(self.oanda_fetcher, self.sentiment_fetcher)
            
            # Fetch and display account info
            self.update_account_info()

            # Set initial status
            self.update_status("Ready to start trading analysis")
            
        except Exception as e:
            logging.error(f"Failed to initialize trading components: {e}")
            logging.debug(traceback.format_exc())
            messagebox.showerror("Initialization Error", f"Failed to initialize trading components: {e}")

    def update_account_info(self):
        try:
            account_summary = self.oanda_fetcher.fetch_account_summary()
            
            if account_summary:
                balance = account_summary.get("balance", "Unknown")
                currency = account_summary.get("currency", "Unknown")
                margin_rate = account_summary.get("marginRate", "Unknown")
                
                account_info = f"Account: {ACCOUNT_ID} | Balance: {balance} {currency} | Margin Rate: {margin_rate}"
                self.account_label.config(text=account_info)
            else:
                self.account_label.config(text="Account: Failed to fetch account info")
                
        except Exception as e:
            logging.error(f"Failed to update account info: {e}")
            self.account_label.config(text="Account: Error fetching account info")

    def update_status(self, message: str):
        self.status_label.config(text=message)

    def update_last_update_time(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_update_label.config(text=f"Last Update: {now}")

    def start_trading_thread(self):
        self.running = True
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

    def trading_loop(self):
        while self.running:
            try:
                self.root.after(0, lambda: self.update_status("Fetching trading data..."))
                self.refresh_data()
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                logging.debug(traceback.format_exc())
                self.root.after(0, lambda err=e: self.update_status(f"Error: {str(err)}"))  # Pass 'e' as 'err'

            try:
                # Wait for the specified interval
                interval = int(self.interval_var.get())
                time.sleep(interval)
            except ValueError:
                # Default to 60 seconds if interval is invalid
                time.sleep(60)
            except Exception as e:
                logging.error(f"Error in trading loop sleep: {e}")
                time.sleep(60)  # Default fallback

    def refresh_data(self):
            import random
            try:
                # Update status
                self.update_status("Analyzing trading data...")

                # Get trade decisions
                all_decisions = self.trading_strategy.decide_trade()

                # Add signals to history
                for instrument, data in all_decisions.items():
                    if "signal" in data and data["signal"]:
                        self.history.add_signal(data["signal"])

                # Update UI with new decisions
                self.root.after(0, lambda: self.update_decisions_table(all_decisions))

                # Update trade details
                self.root.after(0, lambda: self.update_details_text(all_decisions))

                # Update chart
                self.root.after(0, lambda: self.update_chart(all_decisions))

                # Update the currently selected instrument details if applicable
                selected = self.selected_instrument.get()
                if selected:
                    self.root.after(0, self.refresh_detailed_analysis)
                    self.root.after(0, self.refresh_signal_history)
                    self.root.after(0, self.refresh_order_book)
                    self.root.after(0, self.refresh_position_book)

                # Refresh sentiment analysis periodically (less frequently)
                if random.random() < 0.2:  # ~20% chance each refresh
                    self.root.after(0, self.refresh_sentiment_analysis)

                # Update account info periodically
                if random.random() < 0.1:  # ~10% chance each refresh
                    self.root.after(0, self.update_account_info)

                # Update last update time
                self.root.after(0, self.update_last_update_time)

                # Update status
                self.root.after(0, lambda: self.update_status("Trading analysis completed"))

            except Exception as e:
                logging.error(f"Error refreshing data: {e}")
                logging.debug(traceback.format_exc())
                self.root.after(0, lambda err=e: self.update_status(f"Error: {str(err)}"))  # Pass 'e' as 'err'

    def update_decisions_table(self, decisions: Dict[str, Dict[str, Any]]):
        # Clear existing data
        for item in self.decisions_tree.get_children():
            self.decisions_tree.delete(item)

        # Add new data
        for instrument, data in decisions.items():
            decision = data["decision"]
            score = data["total_score"]
            details = data["details"]

            # Determine row tag based on decision for coloring
            row_tag = "neutral"
            if decision == "Strong Bullish":
                row_tag = "strong_long"
            elif decision == "Bullish":
                row_tag = "long"
            elif decision == "Strong Bearish":
                row_tag = "strong_short"
            elif decision == "Bearish":
                row_tag = "short"
            elif decision == "Error":
                row_tag = "error"

            # Insert data into table
            values = (
                instrument,
                decision,
                f"{score:.2f}",
                f"{details['order_score']:.1f}",
                f"{details['position_score']:.1f}",
                f"{details['pair_sentiment_score']:.1f}",
                f"{details['currency_sentiment_score']:.1f}",
                f"{details['retail_profit_score']:.1f}"
            )

            self.decisions_tree.insert("", tk.END, values=values, tags=(row_tag,))

        # Configure the tags for coloring
        self.decisions_tree.tag_configure("strong_long", background="#008000")  # Dark Green
        self.decisions_tree.tag_configure("long", background="#4CAF50")        # Green
        self.decisions_tree.tag_configure("strong_short", background="#8B0000") # Dark Red
        self.decisions_tree.tag_configure("short", background="#F44336")       # Red
        self.decisions_tree.tag_configure("neutral", background="#607D8B")      # Blue-grey
        self.decisions_tree.tag_configure("error", background="#424242")       # Dark grey

    def update_details_text(self, decisions: Dict[str, Dict[str, Any]]):  # Correct indentation
        self.details_text.configure(state='normal')
        self.details_text.delete(1.0, tk.END)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.details_text.insert(tk.END, f"Trading Analysis at {now}\n", "header")
        self.details_text.insert(tk.END, "=" * 50 + "\n\n")

        # Add the weighting information
        self.details_text.insert(tk.END, "Signal Weighting:\n", "subheader")
        self.details_text.insert(tk.END, f"• Order Book: {self.trading_strategy.weights['order_book']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Position Book: {self.trading_strategy.weights['position_book']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Pair-Specific Sentiment: {self.trading_strategy.weights['pair_sentiment']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Currency-Level Sentiment: {self.trading_strategy.weights['currency_sentiment']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Retail Profitability: {self.trading_strategy.weights['retail_profitability']*100:.0f}%\n\n")

        self.details_text.insert(tk.END, "Decision Thresholds:\n", "subheader")
        self.details_text.insert(tk.END, f"• Total score > +{self.trading_strategy.thresholds['strong_bullish']}: Strong Bullish\n")
        self.details_text.insert(tk.END, f"• Total score > +{self.trading_strategy.thresholds['bullish']}: Bullish\n")
        self.details_text.insert(tk.END, f"• Total score < -{self.trading_strategy.thresholds['strong_bearish']}: Strong Bearish\n")
        self.details_text.insert(tk.END, f"• Total score < -{self.trading_strategy.thresholds['bearish']}: Bearish\n")
        self.details_text.insert(tk.END, "• Otherwise: No Trade\n\n")

        # Count decision types
        decision_counts = {"Strong Bullish": 0, "Bullish": 0, "No Trade": 0, "Bearish": 0, "Strong Bearish": 0, "Error": 0}
        for data in decisions.values():
            decision = data["decision"]
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # Display decision counts
        self.details_text.insert(tk.END, "Current Signals Summary:\n", "subheader")
        for decision, count in decision_counts.items():
            if count > 0:
                tag = "neutral"
                if decision == "Strong Bullish":
                    tag = "strong_long"
                elif decision == "Bullish":
                    tag = "long"
                elif decision == "Strong Bearish":
                    tag = "strong_short"
                elif decision == "Bearish":
                    tag = "short"
                elif decision == "Error":
                    tag = "error"
                
                self.details_text.insert(tk.END, f"• {decision}: {count}\n", tag)
        
        self.details_text.insert(tk.END, "\n" + "-" * 50 + "\n\n")

        # Add detailed information for each instrument with tradable signals
        tradable_signals = {k: v for k, v in decisions.items() 
                           if v["decision"] in ["Strong Bullish", "Bullish", "Strong Bearish", "Bearish"]}
        
        if tradable_signals:
            self.details_text.insert(tk.END, "Tradable Signals:\n", "subheader")
            
            for instrument, data in tradable_signals.items():
                decision = data["decision"]
                score = data["total_score"]
                details = data["details"]
                price = data.get("price", 0.0)

                # Insert header with instrument name
                self.details_text.insert(tk.END, f"{instrument} ", "instrument_header")
                
                # Add current price if available
                if price:
                    self.details_text.insert(tk.END, f"@ {price:.5f}\n", "price")
                else:
                    self.details_text.insert(tk.END, "\n", "price")

                # Set text color based on decision
                tag = "neutral"
                if decision == "Strong Bullish":
                    tag = "strong_long"
                elif decision == "Bullish":
                    tag = "long"
                elif decision == "Strong Bearish":
                    tag = "strong_short"
                elif decision == "Bearish":
                    tag = "short"

                self.details_text.insert(tk.END, f"Decision: {decision}\n", tag)
                self.details_text.insert(tk.END, f"Total Score: {score:.2f}\n", tag)
                
                # Add component scores
                self.details_text.insert(tk.END, "Components: ")
                self.details_text.insert(tk.END, f"OB: {details['order_score']} | ")
                self.details_text.insert(tk.END, f"PB: {details['position_score']} | ")
                self.details_text.insert(tk.END, f"PS: {details['pair_sentiment_score']} | ")
                self.details_text.insert(tk.END, f"CS: {details['currency_sentiment_score']} | ")
                self.details_text.insert(tk.END, f"RP: {details['retail_profit_score']}\n")

                self.details_text.insert(tk.END, "\n")
        else:
            self.details_text.insert(tk.END, "No tradable signals at this time.\n", "neutral")

        # Configure text tags
        self.details_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.details_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.details_text.tag_configure("instrument_header", font=("Segoe UI", 11, "bold"))
        self.details_text.tag_configure("price", font=("Segoe UI", 10, "italic"))
        self.details_text.tag_configure("strong_long", foreground="#008000")  # Dark Green
        self.details_text.tag_configure("long", foreground="#4CAF50")        # Green
        self.details_text.tag_configure("strong_short", foreground="#8B0000") # Dark Red
        self.details_text.tag_configure("short", foreground="#F44336")       # Red
        self.details_text.tag_configure("neutral", foreground="#607D8B")      # Blue-grey
        self.details_text.tag_configure("error", foreground="#424242")       # Dark grey

        self.details_text.configure(state='disabled')

    def update_chart(self, decisions: Dict[str, Dict[str, Any]]):
        # Clear the figure
        self.fig.clear()

        # Create subplots: one for all instruments and one for details
        ax_all = self.fig.add_subplot(211)  # Top plot for all instruments
        ax_detail = self.fig.add_subplot(212)  # Bottom plot for key components
        
        # Get all instruments with history
        instruments_with_history = []
        for instrument in self.trading_strategy.instruments:
            signals = self.history.get_signals(instrument, count=20)
            if len(signals) >= 2:
                instruments_with_history.append(instrument)
        
        # Plot all instruments in top chart
        for instrument in instruments_with_history:
            signals = self.history.get_signals(instrument, count=20)
            timestamps = [s.timestamp for s in signals]
            scores = [s.total_score for s in signals]
            
            # Determine color based on most recent decision
            color = 'gray'
            if signals and signals[-1].decision == "Strong Bullish":
                color = '#008000'  # Dark green
            elif signals and signals[-1].decision == "Bullish":
                color = '#4CAF50'  # Green
            elif signals and signals[-1].decision == "Strong Bearish":
                color = '#8B0000'  # Dark red
            elif signals and signals[-1].decision == "Bearish":
                color = '#F44336'  # Red
                
            ax_all.plot(timestamps, scores, marker='o', linestyle='-', linewidth=1.5, 
                      markersize=4, label=instrument, color=color, alpha=0.7)
        
        # Add decision threshold lines to top chart
        ax_all.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000', 
                     linestyle='--', alpha=0.7, label='Strong Bullish')
        ax_all.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50', 
                     linestyle='--', alpha=0.7, label='Bullish')
        ax_all.axhline(y=-self.trading_strategy.thresholds['bearish'], color='#F44336', 
                     linestyle='--', alpha=0.7, label='Bearish')
        ax_all.axhline(y=-self.trading_strategy.thresholds['strong_bearish'], color='#8B0000', 
                     linestyle='--', alpha=0.7, label='Strong Bearish')
        ax_all.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Format top chart
        ax_all.set_title('All Instruments Trading Signals')
        ax_all.set_ylabel('Signal Score')
        ax_all.set_ylim(-2, 2)
        ax_all.grid(True, alpha=0.3)
        
        # Format x-axis dates nicely
        ax_all.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Get instruments with tradable signals for the detail chart
        tradable_instruments = [k for k, v in decisions.items() 
                              if v["decision"] in ["Strong Bullish", "Bullish", "Strong Bearish", "Bearish"]]
        
        if tradable_instruments:
            # Pick up to 5 instruments for the detail chart
            detail_instruments = tradable_instruments[:5]
            
            # Create bar positions
            components = ['Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit']
            x = np.arange(len(components))
            width = 0.15  # Width of bars
            
            # Plot bars for each instrument's components
            for i, instrument in enumerate(detail_instruments):
                data = decisions[instrument]
                details = data["details"]
                
                component_scores = [
                    details['order_score'] * self.trading_strategy.weights['order_book'],
                    details['position_score'] * self.trading_strategy.weights['position_book'],
                    details['pair_sentiment_score'] * self.trading_strategy.weights['pair_sentiment'],
                    details['currency_sentiment_score'] * self.trading_strategy.weights['currency_sentiment'],
                    details['retail_profit_score'] * self.trading_strategy.weights['retail_profitability']
                ]
                
                # Determine color based on decision
                color = 'gray'
                if data["decision"] == "Strong Bullish":
                    color = '#008000'  # Dark green
                elif data["decision"] == "Bullish":
                    color = '#4CAF50'  # Green
                elif data["decision"] == "Strong Bearish":
                    color = '#8B0000'  # Dark red
                elif data["decision"] == "Bearish":
                    color = '#F44336'  # Red
                
                offset = width * (i - len(detail_instruments)/2 + 0.5)
                ax_detail.bar(x + offset, component_scores, width, label=instrument, color=color, alpha=0.7)
            
            # Format detail chart
            ax_detail.set_title('Signal Components of Tradable Instruments')
            ax_detail.set_ylabel('Weighted Score')
            ax_detail.set_xticks(x)
            ax_detail.set_xticklabels(components, rotation=45, ha='right')
            ax_detail.legend(loc='upper right')
            ax_detail.grid(True, axis='y', alpha=0.3)
            
        else:
            # If no tradable signals, show a message
            ax_detail.text(0.5, 0.5, 'No tradable signals at this time', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax_detail.transAxes, fontsize=12)
            ax_detail.set_xticks([])
            ax_detail.set_yticks([])
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Redraw the canvas
        self.canvas.draw()

    def on_tree_select(self, event):
        """Handle selection in the decisions tree."""
        selected_items = self.decisions_tree.selection()
        if not selected_items:
            return
            
        # Get the selected instrument
        item_id = selected_items[0]
        item_values = self.decisions_tree.item(item_id, 'values')
        instrument = item_values[0]
        
        # Update the selected instrument
        self.selected_instrument.set(instrument)
        
        # Switch to the detailed analysis tab
        self.notebook.select(self.detailed_analysis_tab)

    def on_instrument_selected(self, *args):
        """Handle instrument selection change."""
        # This method will be called whenever the selected_instrument StringVar changes
        # We'll refresh all relevant data for the new instrument
        instrument = self.selected_instrument.get()
        if instrument:
            # Update UI areas that depend on the selected instrument
            self.refresh_detailed_analysis()
            self.refresh_signal_history()
            self.refresh_order_book()
            self.refresh_position_book()

    def refresh_detailed_analysis(self):
        """Refresh the detailed analysis for the selected instrument."""
        instrument = self.selected_instrument.get()
        if not instrument:
            return
            
        try:
            # Get latest signal
            latest_signal = self.history.get_latest_signal(instrument)
            if not latest_signal:
                return
                
            # Update signal history chart
            self.update_signal_history_chart(instrument)
            
            # Update component analysis chart
            self.update_component_analysis_chart(latest_signal)
            
            # Update order book chart
            self.update_detailed_orderbook_chart(instrument)
            
            # Update position book chart
            self.update_detailed_positionbook_chart(instrument)
            
        except Exception as e:
            logging.error(f"Error refreshing detailed analysis: {e}")
            logging.debug(traceback.format_exc())

    def update_signal_history_chart(self, instrument: str):
        """Update the signal history chart in the detailed analysis tab."""
        signals = self.history.get_signals(instrument)
        if not signals:
            self.signal_history_fig.clear()
            ax = self.signal_history_fig.add_subplot(111)
            ax.text(0.5, 0.5, f'No signal history for {instrument}', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.signal_history_canvas.draw()
            return
            
        # Clear the figure
        self.signal_history_fig.clear()
        ax = self.signal_history_fig.add_subplot(111)
        
        # Get data
        timestamps = [s.timestamp for s in signals]
        total_scores = [s.total_score for s in signals]
        
        # Plot the total score
        ax.plot(timestamps, total_scores, marker='o', linestyle='-', linewidth=2, 
              label='Total Score', color='blue', zorder=5)
        
        # Plot decision threshold lines
        ax.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000', 
                 linestyle='--', alpha=0.7, label='Strong Bullish')
        ax.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50', 
                 linestyle='--', alpha=0.7, label='Bullish')
        ax.axhline(y=-self.trading_strategy.thresholds['bearish'], color='#F44336', 
                 linestyle='--', alpha=0.7, label='Bearish')
        ax.axhline(y=-self.trading_strategy.thresholds['strong_bearish'], color='#8B0000', 
                 linestyle='--', alpha=0.7, label='Strong Bearish')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Highlight the background based on signal decisions
        for i in range(len(signals) - 1):
            start = timestamps[i]
            end = timestamps[i+1]
            decision = signals[i].decision
            
            if decision == "Strong Bullish":
                ax.axvspan(start, end, alpha=0.2, color='#008000')
            elif decision == "Bullish":
                ax.axvspan(start, end, alpha=0.2, color='#4CAF50')
            elif decision == "Strong Bearish":
                ax.axvspan(start, end, alpha=0.2, color='#8B0000')
            elif decision == "Bearish":
                ax.axvspan(start, end, alpha=0.2, color='#F44336')
        
        # Format chart
        ax.set_title(f'Signal History for {instrument}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Score')
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Format x-axis dates nicely
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        self.signal_history_fig.tight_layout()
        
        # Redraw the canvas
        self.signal_history_canvas.draw()
            
    def update_component_analysis_chart(self, signal: TradingSignal):
        """Update the component analysis chart in the detailed analysis tab."""
        # Clear the figure
        self.component_analysis_fig.clear()
        ax = self.component_analysis_fig.add_subplot(111)
        
        # Define components and their weights
        components = ['Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit']
        weights = [
            self.trading_strategy.weights['order_book'],
            self.trading_strategy.weights['position_book'],
            self.trading_strategy.weights['pair_sentiment'],
            self.trading_strategy.weights['currency_sentiment'],
            self.trading_strategy.weights['retail_profitability']
        ]
        
        # Get component scores
        raw_scores = [
            signal.order_book_score,
            signal.position_book_score,
            signal.pair_sentiment_score,
            signal.currency_sentiment_score,
            signal.retail_profit_score
        ]
        
        # Calculate weighted scores
        weighted_scores = [raw * weight for raw, weight in zip(raw_scores, weights)]
        
        # Calculate bar positions
        x = np.arange(len(components))
        width = 0.35
        
        # Plot raw scores
        bars1 = ax.bar(x - width/2, raw_scores, width, label='Raw Score', color='skyblue')
        
        # Plot weighted scores
        bars2 = ax.bar(x + width/2, weighted_scores, width, label='Weighted Score', color='navy')
        
        # Add value labels on top of bars
        for i, (v1, v2) in enumerate(zip(raw_scores, weighted_scores)):
            ax.text(i - width/2, v1 + 0.1, f"{v1}", ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, v2 + 0.1, f"{v2:.2f}", ha='center', va='bottom', fontsize=8)
        
        # Format chart
        ax.set_title(f'Component Analysis for {signal.instrument}')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set y-axis limits with some padding
        max_score = max(max(raw_scores, default=0), max(weighted_scores, default=0))
        min_score = min(min(raw_scores, default=0), min(weighted_scores, default=0))
        padding = 0.5
        ax.set_ylim(min(min_score - padding, -2), max(max_score + padding, 2))
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Add total score text
        ax.text(0.02, 0.95, f"Total Score: {signal.total_score:.2f}", 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust layout
        self.component_analysis_fig.tight_layout()
        
        # Redraw the canvas
        self.component_analysis_canvas.draw()

    def update_detailed_orderbook_chart(self, instrument: str):
        """Update the order book chart in the detailed analysis tab."""
        # Clear the figure
        self.orderbook_fig.clear()
        ax = self.orderbook_fig.add_subplot(111)

        # Fetch order book data
        orderbook_data = self.oanda_fetcher.fetch_orderbook(instrument)

        if not orderbook_data or 'buckets' not in orderbook_data:
            ax.text(0.5, 0.5, f'No order book data available for {instrument}',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.orderbook_canvas.draw()
            return

        # Extract data
        buckets = orderbook_data['buckets']
        current_price = float(orderbook_data.get('price', 0))
        bucket_width = float(orderbook_data.get('bucketWidth', 0))  # Keep for possible fallback
        time_str = orderbook_data.get('time', 'N/A')

        # Parse buckets
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)

        # --- FILTERING LOGIC ---
        if current_price > 0:  # Ensure current_price is valid
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_counts, long_counts, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []
        # --- END FILTERING LOGIC ---

        if not filtered_prices:
            ax.text(0.5, 0.5, f'No order book data within price range for {instrument}',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.orderbook_canvas.draw()
            return

        # --- Percentage Plot (Corrected) ---
        # Calculate a dynamic bar width
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8  # 80% of the price difference
        else:
            bar_width = bucket_width  # Fallback if only one price point

        # Plot the bars
        ax.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7, label='Long %')
        ax.bar(filtered_prices, [-s for s in filtered_shorts], width=bar_width, color='red', alpha=0.7,
              label='Short %', bottom=[-(l + s) for l, s in zip(filtered_longs, filtered_shorts)])


        # --- Axis Limits and Formatting ---
        if current_price > 0:
            ax.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')

        # Set x-axis limits (zoom in)
        padding = (max(filtered_prices) - min(filtered_prices)) * 0.1  # 10% padding
        ax.set_xlim(min(filtered_prices) - padding, max(filtered_prices) + padding)

        # Set y-axis limits
        min_y = min(0, -max(filtered_shorts) - max(filtered_longs)) if filtered_shorts and filtered_longs else -1 # Handle potential empty lists
        max_y = max(filtered_longs) if filtered_longs else 1  # Ensure we have a default
        ax.set_ylim(min_y, max_y)


        # Format chart
        ax.set_title(f'Order Book for {instrument} at {time_str}')
        ax.set_xlabel('Price')
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format the x-axis to avoid scientific notation
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')

        # Adjust layout
        self.orderbook_fig.tight_layout()

        # Redraw the canvas
        self.orderbook_canvas.draw()

    def update_detailed_positionbook_chart(self, instrument: str):
        """Update the position book chart in the detailed analysis tab."""
        # Clear the figure
        self.positionbook_fig.clear()
        ax = self.positionbook_fig.add_subplot(111)

        # Fetch position book data
        positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)

        if not positionbook_data or 'buckets' not in positionbook_data:
            ax.text(0.5, 0.5, f'No position book data available for {instrument}',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.positionbook_canvas.draw()
            return

        # Extract data
        buckets = positionbook_data['buckets']
        current_price = float(positionbook_data.get('price', 0))
        bucket_width = float(positionbook_data.get('bucketWidth', 0))  #Keep for fallback
        time_str = positionbook_data.get('time', 'N/A')

        # Parse buckets
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)


        # --- FILTERING LOGIC ---
        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_counts, long_counts, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []
        # --- END FILTERING LOGIC ---


        if not filtered_prices:
            ax.text(0.5, 0.5, f'No position book data within price range for {instrument}',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.positionbook_canvas.draw()
            return


        # --- Percentage Plot (Corrected) ---
        #Calculate a dynamic bar width
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8  # 80% of the price difference
        else:
            bar_width = bucket_width

        # Plot the bars
        ax.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7, label='Long %')
        ax.bar(filtered_prices, [-s for s in filtered_shorts], width=bar_width, color='red', alpha=0.7,
              label='Short %', bottom=[-(l+s) for l, s in zip(filtered_longs, filtered_shorts)])



        # --- Axis Limits and Formatting ---
        if current_price > 0:
          ax.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')

        # Set x-axis limits (zoom in)
        padding = (max(filtered_prices) - min(filtered_prices)) * 0.1  # 10% padding
        ax.set_xlim(min(filtered_prices) - padding, max(filtered_prices) + padding)


        # Set y-axis limits
        min_y = min(0, -max(filtered_shorts) - max(filtered_longs)) if filtered_shorts and filtered_longs else -1 # Handle potential empty lists
        max_y = max(filtered_longs) if filtered_longs else 1
        ax.set_ylim(min_y, max_y)



        # Format chart
        ax.set_title(f'Position Book for {instrument} at {time_str}')
        ax.set_xlabel('Price')
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format the x-axis to avoid scientific notation
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')

        # Adjust layout
        self.positionbook_fig.tight_layout()

        # Redraw the canvas
        self.positionbook_canvas.draw()
    def refresh_signal_history(self):
        """Refresh the signal history tab."""
        instrument = self.selected_instrument.get()
        if not instrument:
            return
            
        try:
            # Get history length
            try:
                history_length = int(self.history_length_var.get())
            except ValueError:
                history_length = 20
            
            # Get signals
            signals = self.history.get_signals(instrument, count=history_length)
            
            # Update history chart
            self.update_history_chart(instrument, signals)
            
            # Update history table
            self.update_history_table(signals)
            
        except Exception as e:
            logging.error(f"Error refreshing signal history: {e}")
            logging.debug(traceback.format_exc())

    def update_history_chart(self, instrument: str, signals: List[TradingSignal]):
        """Update the history chart in the signal history tab."""
        # Clear the figure
        self.history_fig.clear()
        
        if not signals:
            ax = self.history_fig.add_subplot(111)
            ax.text(0.5, 0.5, f'No signal history for {instrument}', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.history_canvas.draw()
            return
        
        # Create subplots
        gs = self.history_fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = self.history_fig.add_subplot(gs[0])  # Signal scores
        ax2 = self.history_fig.add_subplot(gs[1], sharex=ax1)  # Price
        
        # Get data
        timestamps = [s.timestamp for s in signals]
        total_scores = [s.total_score for s in signals]
        order_book_scores = [s.order_book_score for s in signals]
        position_book_scores = [s.position_book_score for s in signals]
        pair_sentiment_scores = [s.pair_sentiment_score for s in signals]
        currency_sentiment_scores = [s.currency_sentiment_score for s in signals]
        retail_profit_scores = [s.retail_profit_score for s in signals]
        prices = [s.price for s in signals]
        
        # Plot signal scores
        ax1.plot(timestamps, total_scores, marker='o', linestyle='-', linewidth=2, 
               label='Total Score', color='blue', zorder=10)
        ax1.plot(timestamps, order_book_scores, marker='x', linestyle='--', linewidth=1, 
               label='Order Book', color='green', alpha=0.7)
        ax1.plot(timestamps, position_book_scores, marker='s', linestyle='--', linewidth=1, 
               label='Position Book', color='purple', alpha=0.7)
        ax1.plot(timestamps, pair_sentiment_scores, marker='+', linestyle='--', linewidth=1, 
               label='Pair Sentiment', color='orange', alpha=0.7)
        ax1.plot(timestamps, currency_sentiment_scores, marker='d', linestyle='--', linewidth=1, 
               label='Currency Sentiment', color='brown', alpha=0.7)
        ax1.plot(timestamps, retail_profit_scores, marker='*', linestyle='--', linewidth=1, 
               label='Retail Profit', color='gray', alpha=0.7)
        
        # Plot threshold lines
        ax1.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000', 
                  linestyle='--', alpha=0.7, label='Strong Bullish')
        ax1.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50', 
                  linestyle='--', alpha=0.7, label='Bullish')
        ax1.axhline(y=-self.trading_strategy.thresholds['bearish'], color='#F44336', 
                  linestyle='--', alpha=0.7, label='Bearish')
        ax1.axhline(y=-self.trading_strategy.thresholds['strong_bearish'], color='#8B0000', 
                  linestyle='--', alpha=0.7, label='Strong Bearish')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot price
        if prices and any(p > 0 for p in prices):
            ax2.plot(timestamps, prices, marker='o', linestyle='-', linewidth=1.5, 
                   color='black', label='Price')
        else:
            ax2.text(0.5, 0.5, 'No price data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax2.transAxes, fontsize=10)
        
        # Format signal score chart
        ax1.set_title(f'Signal History for {instrument}')
        ax1.set_ylabel('Signal Score')
        ax1.set_ylim(-2.5, 2.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize='small')
        
        # Format price chart
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates nicely
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        self.history_fig.tight_layout()
        
        # Redraw the canvas
        self.history_canvas.draw()

    def update_history_table(self, signals: List[TradingSignal]):
        """Update the history table in the signal history tab."""
        # Clear existing data
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        if not signals:
            return
        
        # Add signals in reverse order (newest first)
        for signal in reversed(signals):
            # Determine row tag based on decision for coloring
            row_tag = "neutral"
            if signal.decision == "Strong Bullish":
                row_tag = "strong_long"
            elif signal.decision == "Bullish":
                row_tag = "long"
            elif signal.decision == "Strong Bearish":
                row_tag = "strong_short"
            elif signal.decision == "Bearish":
                row_tag = "short"
            
            # Insert data into table
            values = (
                signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                signal.decision,
                f"{signal.total_score:.2f}",
                f"{signal.price:.5f}" if signal.price else "N/A",
                f"{signal.order_book_score}",
                f"{signal.position_book_score}",
                f"{signal.pair_sentiment_score}",
                f"{signal.currency_sentiment_score}",
                f"{signal.retail_profit_score}"
            )
            
            self.history_tree.insert("", tk.END, values=values, tags=(row_tag,))
        
        # Configure the tags for coloring
        self.history_tree.tag_configure("strong_long", background="#008000")  # Dark Green
        self.history_tree.tag_configure("long", background="#4CAF50")        # Green
        self.history_tree.tag_configure("strong_short", background="#8B0000") # Dark Red
        self.history_tree.tag_configure("short", background="#F44336")       # Red
        self.history_tree.tag_configure("neutral", background="#607D8B")      # Blue-grey

    def refresh_order_book(self):
        """Refresh the order book tab."""
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        
        try:
            # Fetch order book data
            orderbook_data = self.oanda_fetcher.fetch_orderbook(instrument)
            
            # Update order book visualization
            self.update_order_book_visualization(instrument, orderbook_data)
            
            # Update order book analysis
            self.update_order_book_analysis(instrument, orderbook_data)
            
            # Update raw order book data
            self.update_raw_order_book_data(orderbook_data)
            
        except Exception as e:
            logging.error(f"Error refreshing order book: {e}")
            logging.debug(traceback.format_exc())

    def update_order_book_visualization(self, instrument: str, orderbook_data: Optional[Dict[str, Any]]):
        """Update the order book visualization."""
        self.order_book_fig.clear()

        if not orderbook_data or 'buckets' not in orderbook_data:
            ax = self.order_book_fig.add_subplot(111)
            ax.text(0.5, 0.5, f'No order book data available for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.order_book_canvas.draw()
            return

        gs = self.order_book_fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax1 = self.order_book_fig.add_subplot(gs[0])  # Percentages
        ax2 = self.order_book_fig.add_subplot(gs[1], sharex=ax1)  # Cumulative

        buckets = orderbook_data['buckets']
        current_price = float(orderbook_data.get('price', 0))
        bucket_width = float(orderbook_data.get('bucketWidth', 0))  # Keep for cumulative chart
        time_str = orderbook_data.get('time', 'N/A')

        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)

        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_percents, long_percents, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []

        if not filtered_prices:
            ax1.text(0.5, 0.5, f'No order book data within price range for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            self.order_book_canvas.draw()
            return

        # --- Cumulative Plot (Corrected) ---
        price_sorted_data = sorted(zip(filtered_prices, filtered_shorts, filtered_longs))
        sorted_prices, sorted_shorts, sorted_longs = zip(*price_sorted_data)
        cum_shorts = np.cumsum(sorted_shorts)
        cum_longs = np.cumsum(sorted_longs)
        ax2.plot(sorted_prices, cum_longs, 'g-', label='Cum. Long %')
        ax2.plot(sorted_prices, cum_shorts, 'r-', label='Cum. Short %')
        # --- End Cumulative Plot ---

        # --- Percentage Plot (Corrected) ---
        # Calculate a dynamic bar width
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8  # 80% of the price difference
        else:
            bar_width = bucket_width  # Fallback if only one price point
        
        # Plot the bars
        ax1.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7, label='Long %')
        ax1.bar(filtered_prices, [-s for s in filtered_shorts], width=bar_width, color='red', alpha=0.7,
                label='Short %', bottom=[-(l + s) for l, s in zip(filtered_longs, filtered_shorts)])


        # --- Axis Limits and Formatting (Both Charts) ---
        if current_price > 0:
            ax1.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
            ax2.axvline(x=current_price, color='blue', linestyle='-', linewidth=2)

        # Set x-axis limits (zoom in)
        padding = (max(filtered_prices) - min(filtered_prices)) * 0.1  # 10% padding
        ax1.set_xlim(min(filtered_prices) - padding, max(filtered_prices) + padding)
        ax2.set_xlim(min(filtered_prices) - padding, max(filtered_prices) + padding)  # Same for cumulative

        #Set Y-axis limits
        min_y = min(0, -max(filtered_shorts) - max(filtered_longs)) if filtered_shorts and filtered_longs else -1 # Handle potential empty lists
        max_y = max(filtered_longs) if filtered_longs else 1  # Ensure we have a default if lists are empty

        ax1.set_ylim(min_y, max_y)

        ax1.set_title(f'Order Book for {instrument} at {time_str}')
        ax1.set_ylabel('Percentage')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Price')
        ax2.set_ylabel('Cum. %')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        ax1.ticklabel_format(useOffset=False, style='plain', axis='x')
        ax2.ticklabel_format(useOffset=False, style='plain', axis='x')
        # --- End Axis Limits and Formatting ---

        self.order_book_fig.tight_layout()
        self.order_book_canvas.draw()

    def update_order_book_analysis(self, instrument: str, orderbook_data: Optional[Dict[str, Any]]):
        """Update the order book analysis text."""
        self.order_book_analysis_text.configure(state='normal')
        self.order_book_analysis_text.delete(1.0, tk.END)
        
        if not orderbook_data or 'buckets' not in orderbook_data:
            self.order_book_analysis_text.insert(tk.END, f"No order book data available for {instrument}.")
            self.order_book_analysis_text.configure(state='disabled')
            return
        
        # Extract data
        buckets = orderbook_data['buckets']
        current_price = float(orderbook_data.get('price', 0))
        bucket_width = float(orderbook_data.get('bucketWidth', 0))
        time_str = orderbook_data.get('time', 'N/A')
        
        # Parse buckets
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        
        # Calculate some analysis metrics
        total_long = sum(long_counts)
        total_short = sum(short_counts)
        
        # Calculate price levels of interest
        price_diff_50_pips = self.trading_strategy.pips_to_price_diff(instrument, 50)
        lower_bound_50pips = current_price - price_diff_50_pips
        upper_bound_50pips = current_price + price_diff_50_pips
        
        # Filter buckets within 50 pips
        near_price_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                              if lower_bound_50pips <= p <= upper_bound_50pips]
        
        near_price_long_pct = 0
        near_price_short_pct = 0
        
        if near_price_buckets:
            _, near_shorts, near_longs = zip(*near_price_buckets)
            near_price_long_pct = sum(near_longs)
            near_price_short_pct = sum(near_shorts)
        
        # Determine support and resistance levels (simplistic method)
        support_levels = []
        resistance_levels = []
        
        # Look for price levels with high concentration of limit orders
        for i, (price, short_pct, long_pct) in enumerate(zip(prices, short_counts, long_counts)):
            # High concentration of buy limit orders below current price = potential support
            if price < current_price and long_pct > 5:  # Arbitrary threshold
                support_levels.append((price, long_pct))
            
            # High concentration of sell limit orders above current price = potential resistance
            if price > current_price and short_pct > 5:  # Arbitrary threshold
                resistance_levels.append((price, short_pct))
        
        # Sort by percentage (highest first)
        support_levels = sorted(support_levels, key=lambda x: x[1], reverse=True)[:3]  # Top 3
        resistance_levels = sorted(resistance_levels, key=lambda x: x[1], reverse=True)[:3]  # Top 3
        
        # Write analysis to text widget
        self.order_book_analysis_text.insert(tk.END, f"ORDER BOOK ANALYSIS FOR {instrument}\n", "header")
        self.order_book_analysis_text.insert(tk.END, f"Data as of: {time_str}\n\n")
        
        self.order_book_analysis_text.insert(tk.END, "CURRENT PRICE ANALYSIS:\n", "subheader")
        self.order_book_analysis_text.insert(tk.END, f"Current Price: {current_price:.5f}\n")
        
        # Long/short ratio analysis
        if total_short > 0:
            long_short_ratio = total_long / total_short
            self.order_book_analysis_text.insert(tk.END, f"Overall Long/Short Ratio: {long_short_ratio:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "Overall Long/Short Ratio: ∞ (no short orders)\n")
        
        # Near price analysis
        self.order_book_analysis_text.insert(tk.END, f"\nORDERS WITHIN 50 PIPS OF CURRENT PRICE:\n", "subheader")
        self.order_book_analysis_text.insert(tk.END, f"Long Orders: {near_price_long_pct:.2f}%\n")
        self.order_book_analysis_text.insert(tk.END, f"Short Orders: {near_price_short_pct:.2f}%\n")
        
        if near_price_short_pct > 0:
            near_long_short_ratio = near_price_long_pct / near_price_short_pct
            self.order_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: {near_long_short_ratio:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "Near Price Long/Short Ratio: ∞ (no short orders near price)\n")
        
        # Support and resistance levels
        self.order_book_analysis_text.insert(tk.END, f"\nKEY SUPPORT LEVELS (Buy Limit Orders):\n", "subheader")
        if support_levels:
            for price, pct in support_levels:
                diff_pips = abs(current_price - price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.order_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Strength: {pct:.2f}%\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "No significant support levels detected.\n")
        
        self.order_book_analysis_text.insert(tk.END, f"\nKEY RESISTANCE LEVELS (Sell Limit Orders):\n", "subheader")
        if resistance_levels:
            for price, pct in resistance_levels:
                diff_pips = abs(price - current_price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.order_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Strength: {pct:.2f}%\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "No significant resistance levels detected.\n")
        
        # Trading signals based on order book
        score = self.trading_strategy.analyze_orderbook(instrument)
        self.order_book_analysis_text.insert(tk.END, f"\nORDER BOOK TRADING SIGNAL:\n", "subheader")
        if score > 0:
            self.order_book_analysis_text.insert(tk.END, f"Bullish Signal (Score: {score})\n", "bullish")
            self.order_book_analysis_text.insert(tk.END, "Long orders significantly outnumber short orders within range of current price.\n")
        elif score < 0:
            self.order_book_analysis_text.insert(tk.END, f"Bearish Signal (Score: {score})\n", "bearish")
            self.order_book_analysis_text.insert(tk.END, "Short orders significantly outnumber long orders within range of current price.\n")
        else:
            self.order_book_analysis_text.insert(tk.END, f"Neutral Signal (Score: {score})\n", "neutral")
            self.order_book_analysis_text.insert(tk.END, "No clear bias in order distribution near current price.\n")
        
        # Configure text tags
        self.order_book_analysis_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.order_book_analysis_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.order_book_analysis_text.tag_configure("bullish", foreground="#4CAF50")
        self.order_book_analysis_text.tag_configure("bearish", foreground="#F44336")
        self.order_book_analysis_text.tag_configure("neutral", foreground="#607D8B")
        
        self.order_book_analysis_text.configure(state='disabled')

    def update_raw_order_book_data(self, orderbook_data: Optional[Dict[str, Any]]):
        """Update the raw order book data text."""
        self.order_book_raw_text.configure(state='normal')
        self.order_book_raw_text.delete(1.0, tk.END)
        
        if not orderbook_data:
            self.order_book_raw_text.insert(tk.END, "No order book data available.")
            self.order_book_raw_text.configure(state='disabled')
            return
        
        # Format the JSON data for display
        formatted_json = json.dumps(orderbook_data, indent=2)
        self.order_book_raw_text.insert(tk.END, formatted_json)
        
        self.order_book_raw_text.configure(state='disabled')

    def refresh_position_book(self):
        """Refresh the position book tab."""
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        
        try:
            # Fetch position book data
            positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)
            
            # Update position book visualization
            self.update_position_book_visualization(instrument, positionbook_data)
            
            # Update position book analysis
            self.update_position_book_analysis(instrument, positionbook_data)
            
            # Update raw position book data
            self.update_raw_position_book_data(positionbook_data)
            
        except Exception as e:
            logging.error(f"Error refreshing position book: {e}")
            logging.debug(traceback.format_exc())

    def update_position_book_visualization(self, instrument: str, positionbook_data: Optional[Dict[str, Any]]):
        """Update the position book visualization."""
        self.position_book_fig.clear()

        if not positionbook_data or 'buckets' not in positionbook_data:
            ax = self.position_book_fig.add_subplot(111)
            ax.text(0.5, 0.5, f'No position book data available for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.position_book_canvas.draw()
            return

        gs = self.position_book_fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax1 = self.position_book_fig.add_subplot(gs[0])  # Percentages
        ax2 = self.position_book_fig.add_subplot(gs[1], sharex=ax1)  # Cumulative


        buckets = positionbook_data['buckets']
        current_price = float(positionbook_data.get('price', 0))
        bucket_width = float(positionbook_data.get('bucketWidth', 0))  # Keep for cumulative
        time_str = positionbook_data.get('time', 'N/A')

        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)

        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_percents, long_percents, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []

        if not filtered_prices:
            ax1.text(0.5, 0.5, f'No position book data within price range for {instrument}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax1.transAxes, fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            self.position_book_canvas.draw()
            return

        # --- Cumulative Plot (Corrected) ---
        price_sorted_data = sorted(zip(filtered_prices, filtered_shorts, filtered_longs))
        sorted_prices, sorted_shorts, sorted_longs = zip(*price_sorted_data)
        cum_shorts = np.cumsum(sorted_shorts)
        cum_longs = np.cumsum(sorted_longs)
        ax2.plot(sorted_prices, cum_longs, 'g-', label='Cum. Long %')
        ax2.plot(sorted_prices, cum_shorts, 'r-', label='Cum. Short %')
        # --- End Cumulative Plot ---


        # --- Percentage Plot (Corrected) ---
        # Calculate a dynamic bar width
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8  # 80% of the price difference
        else:
            bar_width = bucket_width # Fallback

        # Plot the bars
        ax1.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7, label='Long %')
        ax1.bar(filtered_prices, [-s for s in filtered_shorts], width=bar_width, color='red', alpha=0.7,
                label='Short %', bottom=[-(l+s) for l, s in zip(filtered_longs, filtered_shorts)])

        # --- Axis Limits and Formatting (Both Charts) ---
        if current_price > 0:
            ax1.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
            ax2.axvline(x=current_price, color='blue', linestyle='-', linewidth=2)

        # Set x-axis limits (zoom in)
        padding = (max(filtered_prices) - min(filtered_prices)) * 0.1  # 10% padding
        ax1.set_xlim(min(filtered_prices) - padding, max(filtered_prices) + padding)
        ax2.set_xlim(min(filtered_prices) - padding, max(filtered_prices) + padding)

        #Set Y-axis limits
        min_y = min(0, -max(filtered_shorts) - max(filtered_longs)) if filtered_shorts and filtered_longs else -1 # Handle potential empty lists
        max_y = max(filtered_longs) if filtered_longs else 1

        ax1.set_ylim(min_y, max_y)


        ax1.set_title(f'Position Book for {instrument} at {time_str}')
        ax1.set_ylabel('Percentage')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Price')
        ax2.set_ylabel('Cum. %')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        ax1.ticklabel_format(useOffset=False, style='plain', axis='x')
        ax2.ticklabel_format(useOffset=False, style='plain', axis='x')

        # --- End Axis Limits and Formatting ---

        self.position_book_fig.tight_layout()
        self.position_book_canvas.draw()

    def update_position_book_analysis(self, instrument: str, positionbook_data: Optional[Dict[str, Any]]):
        """Update the position book analysis text."""
        self.position_book_analysis_text.configure(state='normal')
        self.position_book_analysis_text.delete(1.0, tk.END)
        
        if not positionbook_data or 'buckets' not in positionbook_data:
            self.position_book_analysis_text.insert(tk.END, f"No position book data available for {instrument}.")
            self.position_book_analysis_text.configure(state='disabled')
            return
        
        # Extract data
        buckets = positionbook_data['buckets']
        current_price = float(positionbook_data.get('price', 0))
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        time_str = positionbook_data.get('time', 'N/A')
        
        # Parse buckets
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        
        # Calculate some analysis metrics
        total_long = sum(long_counts)
        total_short = sum(short_counts)
        
        # Calculate price levels of interest
        price_diff_50_pips = self.trading_strategy.pips_to_price_diff(instrument, 50)
        lower_bound_50pips = current_price - price_diff_50_pips
        upper_bound_50pips = current_price + price_diff_50_pips
        
        # Filter buckets within 50 pips
        near_price_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                              if lower_bound_50pips <= p <= upper_bound_50pips]
        
        near_price_long_pct = 0
        near_price_short_pct = 0
        
        if near_price_buckets:
            _, near_shorts, near_longs = zip(*near_price_buckets)
            near_price_long_pct = sum(near_longs)
            near_price_short_pct = sum(near_shorts)
        
        # Determine areas of high position concentration
        long_concentration = []
        short_concentration = []
        
        # Look for price levels with high concentration of positions
        for i, (price, short_pct, long_pct) in enumerate(zip(prices, short_counts, long_counts)):
            if long_pct > 5:  # Arbitrary threshold
                long_concentration.append((price, long_pct))
            
            if short_pct > 5:  # Arbitrary threshold
                short_concentration.append((price, short_pct))
        
        # Sort by percentage (highest first)
        long_concentration = sorted(long_concentration, key=lambda x: x[1], reverse=True)[:3]  # Top 3
        short_concentration = sorted(short_concentration, key=lambda x: x[1], reverse=True)[:3]  # Top 3
        
        # Write analysis to text widget
        self.position_book_analysis_text.insert(tk.END, f"POSITION BOOK ANALYSIS FOR {instrument}\n", "header")
        self.position_book_analysis_text.insert(tk.END, f"Data as of: {time_str}\n\n")
        
        self.position_book_analysis_text.insert(tk.END, "OVERALL POSITION ANALYSIS:\n", "subheader")
        self.position_book_analysis_text.insert(tk.END, f"Current Price: {current_price:.5f}\n")
        
        # Long/short ratio analysis
        if total_short > 0:
            long_short_ratio = total_long / total_short
            self.position_book_analysis_text.insert(tk.END, f"Overall Long/Short Ratio: {long_short_ratio:.2f}\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "Overall Long/Short Ratio: ∞ (no short positions)\n")
        
        # Calculate percentage
        if total_long + total_short > 0:
            long_percentage = (total_long / (total_long + total_short)) * 100
            short_percentage = (total_short / (total_long + total_short)) * 100
            self.position_book_analysis_text.insert(tk.END, f"Long Positions: {long_percentage:.2f}%\n")
            self.position_book_analysis_text.insert(tk.END, f"Short Positions: {short_percentage:.2f}%\n")
        
        # Near price analysis
        self.position_book_analysis_text.insert(tk.END, f"\nPOSITIONS WITHIN 50 PIPS OF CURRENT PRICE:\n", "subheader")
        self.position_book_analysis_text.insert(tk.END, f"Long Positions: {near_price_long_pct:.2f}%\n")
        self.position_book_analysis_text.insert(tk.END, f"Short Positions: {near_price_short_pct:.2f}%\n")
        
        if near_price_short_pct > 0:
            near_long_short_ratio = near_price_long_pct / near_price_short_pct
            self.position_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: {near_long_short_ratio:.2f}\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "Near Price Long/Short Ratio: ∞ (no short positions near price)\n")
        
        # Areas of high position concentration
        self.position_book_analysis_text.insert(tk.END, f"\nHIGH CONCENTRATION OF LONG POSITIONS:\n", "subheader")
        if long_concentration:
            for price, pct in long_concentration:
                diff_pips = abs(current_price - price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.position_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Concentration: {pct:.2f}%\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "No significant concentration of long positions detected.\n")
        
        self.position_book_analysis_text.insert(tk.END, f"\nHIGH CONCENTRATION OF SHORT POSITIONS:\n", "subheader")
        if short_concentration:
            for price, pct in short_concentration:
                diff_pips = abs(price - current_price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.position_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Concentration: {pct:.2f}%\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "No significant concentration of short positions detected.\n")
        
        # Trading signals based on position book
        score = self.trading_strategy.analyze_positionbook(instrument)
        self.position_book_analysis_text.insert(tk.END, f"\nPOSITION BOOK TRADING SIGNAL:\n", "subheader")
        if score > 0:
            self.position_book_analysis_text.insert(tk.END, f"Bullish Signal (Score: {score})\n", "bullish")
            self.position_book_analysis_text.insert(tk.END, "Using a contrarian view: High short position concentration suggests potential bullish reversal.\n")
        elif score < 0:
            self.position_book_analysis_text.insert(tk.END, f"Bearish Signal (Score: {score})\n", "bearish")
            self.position_book_analysis_text.insert(tk.END, "Using a contrarian view: High long position concentration suggests potential bearish reversal.\n")
        else:
            self.position_book_analysis_text.insert(tk.END, f"Neutral Signal (Score: {score})\n", "neutral")
            self.position_book_analysis_text.insert(tk.END, "No clear contrarian bias in position distribution.\n")
        
        # Configure text tags
        self.position_book_analysis_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.position_book_analysis_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.position_book_analysis_text.tag_configure("bullish", foreground="#4CAF50")
        self.position_book_analysis_text.tag_configure("bearish", foreground="#F44336")
        self.position_book_analysis_text.tag_configure("neutral", foreground="#607D8B")
        
        self.position_book_analysis_text.configure(state='disabled')

    def update_raw_position_book_data(self, positionbook_data: Optional[Dict[str, Any]]):
        """Update the raw position book data text."""
        self.position_book_raw_text.configure(state='normal')
        self.position_book_raw_text.delete(1.0, tk.END)
        
        if not positionbook_data:
            self.position_book_raw_text.insert(tk.END, "No position book data available.")
            self.position_book_raw_text.configure(state='disabled')
            return
        
        # Format the JSON data for display
        formatted_json = json.dumps(positionbook_data, indent=2)
        self.position_book_raw_text.insert(tk.END, formatted_json)
        
        self.position_book_raw_text.configure(state='disabled')

    def refresh_sentiment_analysis(self):
        """Refresh the sentiment analysis tab."""
        try:
            # Update retail positions chart
            self.update_retail_positions_chart()
            
            # Update Twitter sentiment chart
            self.update_twitter_sentiment_chart()
            
            # Update MyFxBook sentiment table
            self.update_myfxbook_sentiment_table()
            
        except Exception as e:
            logging.error(f"Error refreshing sentiment analysis: {e}")
            logging.debug(traceback.format_exc())

    def update_retail_positions_chart(self):
        """Update the retail positions chart in the sentiment analysis tab."""
        # Clear the figure
        self.retail_fig.clear()
        
        # Fetch data
        html = self.sentiment_fetcher.fetch_retail_positions_data()
        pie_data = self.sentiment_fetcher.extract_pie_chart_data(html)
        
        if not pie_data:
            ax = self.retail_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No retail positions data available', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.retail_canvas.draw()
            return
        
        # Create pie chart
        ax1 = self.retail_fig.add_subplot(211)
        
        # Create bar chart
        ax2 = self.retail_fig.add_subplot(212)
        
        # Process data for charts
        labels = []
        long_values = []
        short_values = []
        
        for currency, perc_str in pie_data.items():
            if isinstance(perc_str, str):
                perc_str = perc_str.strip('%')
            try:
                perc = float(perc_str)
                labels.append(currency)
                long_values.append(perc)
                short_values.append(100 - perc)
            except (ValueError, TypeError):
                pass
        
        if not labels:
            ax1.text(0.5, 0.5, 'Could not parse retail positions data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax1.transAxes, fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            self.retail_canvas.draw()
            return
        
        # Create pie chart of average sentiment
        avg_long = sum(long_values) / len(long_values) if long_values else 0
        avg_short = 100 - avg_long
        
        pie_sizes = [avg_long, avg_short]
        pie_labels = ['Long', 'Short']
        pie_colors = ['green', 'red']
        pie_explode = (0.1, 0)  # explode the 1st slice (Long)
        
        ax1.pie(pie_sizes, explode=pie_explode, labels=pie_labels, colors=pie_colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        
        ax1.set_title('Average Retail Position Sentiment')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Create bar chart for individual currencies
        x = np.arange(len(labels))
        width = 0.35
        
        ax2.bar(x - width/2, long_values, width, label='Long %', color='green')
        ax2.bar(x + width/2, short_values, width, label='Short %', color='red')
        
        ax2.set_title('Retail Positions by Currency')
        ax2.set_ylabel('Percentage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        
        # Add a reference line at 50%
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        self.retail_fig.tight_layout()
        
        # Redraw the canvas
        self.retail_canvas.draw()

    def update_twitter_sentiment_chart(self):
        """Update the Twitter sentiment chart in the sentiment analysis tab."""
        # Clear the figure
        self.twitter_fig.clear()
        
        # Fetch data
        html = self.sentiment_fetcher.fetch_twitter_sentiment_data()
        twitter_chart = self.sentiment_fetcher.extract_twitter_chart_data(html)
        
        if not twitter_chart:
            ax = self.twitter_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No Twitter sentiment data available', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.twitter_canvas.draw()
            return
        
        # Create chart
        ax = self.twitter_fig.add_subplot(111)
        
        # Process data
        pairs = list(twitter_chart.keys())
        colors = list(twitter_chart.values())
        
        # Map colors to sentiment scores (-1 to 1)
        positive_colors = ["#91DB57", "#57DB80", "#57D3DB", "#5770DB", "#A157DB"]
        negative_colors = ["#DB5F57", "#DBC257"]
        neutral_colors = ["#C0C0C0", "#A0A0A0"]
        
        sentiment_scores = []
        bar_colors = []
        
        for color in colors:
            if color in positive_colors:
                # Map positive colors to values between 0.5 and 1.0
                idx = positive_colors.index(color)
                score = 0.5 + (idx + 1) * (0.5 / len(positive_colors))
                sentiment_scores.append(score)
                bar_colors.append('green')
            elif color in negative_colors:
                # Map negative colors to values between -1.0 and -0.5
                idx = negative_colors.index(color)
                score = -1.0 + idx * (0.5 / len(negative_colors))
                sentiment_scores.append(score)
                bar_colors.append('red')
            else:
                # Neutral colors map to values between -0.5 and 0.5
                sentiment_scores.append(0)
                bar_colors.append('gray')
        
        # Sort by sentiment score
        sorted_data = sorted(zip(pairs, sentiment_scores, bar_colors), key=lambda x: x[1])
        sorted_pairs, sorted_scores, sorted_colors = zip(*sorted_data) if sorted_data else ([], [], [])
        
        if not sorted_pairs:
            ax.text(0.5, 0.5, 'Could not parse Twitter sentiment data', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.twitter_canvas.draw()
            return
        
        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_pairs))
        
        # Create bars with individual colors
        bars = ax.barh(y_pos, sorted_scores, align='center', color=sorted_colors)
        
        # Add sentiment labels
        for i, score in enumerate(sorted_scores):
            if score >= 0.5:
                ax.text(score + 0.05, i, "Bullish", va='center', color='green')
            elif score <= -0.5:
                ax.text(score - 0.25, i, "Bearish", va='center', color='red')
            else:
                ax.text(score + 0.05, i, "Neutral", va='center', color='gray')
        
        # Set chart properties
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_pairs)
        ax.set_xlim(-1.1, 1.1)  # Set x-axis limits
        ax.set_xlabel('Sentiment Score')
        ax.set_title('Twitter Sentiment Analysis by Currency Pair')
        
        # Add reference lines
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3)
        ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.3)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Adjust layout
        self.twitter_fig.tight_layout()
        
        # Redraw the canvas
        self.twitter_canvas.draw()

    def update_myfxbook_sentiment_table(self):
        """Update the MyFxBook sentiment table in the sentiment analysis tab."""
        # Clear existing data
        for item in self.myfxbook_tree.get_children():
            self.myfxbook_tree.delete(item)
        
        # Fetch data
        html = self.sentiment_fetcher.fetch_myfxbook_sentiment_data()
        records = self.sentiment_fetcher.extract_myfxbook_table_data(html)
        
        if not records:
            return
        
        # Add data to table
        for record in records:
            if "Symbol" not in record:
                continue
            
            pair = record["Symbol"]
            
            # Extract and format data
            try:
                long_pct = record.get("Long %", "0%")
                short_pct = record.get("Short %", "0%")
                long_lots = record.get("Long Lots", "0")
                short_lots = record.get("Short Lots", "0")
                
                # Ensure percentages have % symbol
                if isinstance(long_pct, str) and not long_pct.endswith('%'):
                    long_pct = f"{long_pct}%"
                elif isinstance(long_pct, (int, float)):
                    long_pct = f"{long_pct}%"
                    
                if isinstance(short_pct, str) and not short_pct.endswith('%'):
                    short_pct = f"{short_pct}%"
                elif isinstance(short_pct, (int, float)):
                    short_pct = f"{short_pct}%"
                
                # Insert data into table
                values = (pair, long_pct, short_pct, long_lots, short_lots)
                
                # Determine row tag based on sentiment bias
                row_tag = "neutral"
                try:
                    long_val = float(long_pct.rstrip('%'))
                    if long_val >= 60:
                        row_tag = "long"
                    elif long_val <= 40:
                        row_tag = "short"
                except ValueError:
                    pass
                
                self.myfxbook_tree.insert("", tk.END, values=values, tags=(row_tag,))
                
            except Exception as e:
                logging.error(f"Error processing MyFxBook data for {pair}: {e}")
        
        # Configure the tags for coloring
        self.myfxbook_tree.tag_configure("long", background="#4CAF50")  # Green
        self.myfxbook_tree.tag_configure("short", background="#F44336")  # Red
        self.myfxbook_tree.tag_configure("neutral", background="#607D8B")  # Blue-grey

# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    # Create root window
    root = tk.Tk()
    
    # Create application
    app = TradingRobotGUI(root)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
