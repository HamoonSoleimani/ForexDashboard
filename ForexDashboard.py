import requests
import json
import time
import re
import logging
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime, timedelta
import threading
import traceback
import random
import io
import csv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

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
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
ACCOUNT_ID = "YOUR_ACCOUNT_ID"  # Replace with your actual account ID
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
        # Schedule the UI update in the main thread
        self.text_widget.after(0, lambda: self.append_log(msg, record.levelno))

    def append_log(self, msg, levelno):
        self.text_widget.configure(state='normal')
        if levelno >= logging.ERROR:
            self.text_widget.tag_config('error', foreground='red')
            self.text_widget.insert(tk.END, msg + '\n', 'error')
        elif levelno >= logging.WARNING:
            self.text_widget.tag_config('warning', foreground='orange')
            self.text_widget.insert(tk.END, msg + '\n', 'warning')
        else:
            self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)

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
        params: Dict[str, Any] = {}
        req = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
        try:
            response = self.api.request(req)
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
        params: Dict[str, Any] = {}
        req = instruments.InstrumentsPositionBook(instrument=instrument, params=params)
        try:
            response = self.api.request(req)
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
        req = accounts.AccountSummary(accountID=self.account_id)
        try:
            response = self.api.request(req)
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
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/91.0.4472.124 Safari/537.36')
        })
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes

    def fetch_data(self, url: str, retries: int = 3, timeout: int = 15) -> Optional[str]:
        now = time.time()
        if url in self.cache and self.cache_expiry.get(url, 0) > now:
            return self.cache[url]
        for i in range(retries):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                self.cache[url] = response.text
                self.cache_expiry[url] = now + self.cache_duration
                return response.text
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching {url} (attempt {i+1}/{retries}): {e}")
                time.sleep(2**(i+1))
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
            if ("mpld3.draw_figure" in script_text and "fig_el" in script_text
                and "paths" in script_text and "texts" in script_text):
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
            if ("mpld3.draw_figure" in script_text and "fig_el" in script_text
                and "lines" in script_text and "texts" in script_text):
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
                                        if 'color' in line_obj and i < len(currencies):
                                            currency = currencies[i]
                                            chart_data[currency] = line_obj['color']
                                    return chart_data
                except Exception as e:
                    logging.error("Error extracting Twitter sentiment chart data: %s", e)
                    logging.debug(traceback.format_exc())
        return chart_data

# -------------------------
# TRADING STRATEGY (RULE-BASED DECISION ENGINE)
# -------------------------
class TradingStrategy:
    def __init__(self, oanda_fetcher: OandaDataFetcher, sentiment_fetcher: SentimentDataFetcher):
        self.oanda_fetcher = oanda_fetcher
        self.sentiment_fetcher = sentiment_fetcher
        self.instruments = ['EUR_USD', 'NZD_USD', 'USD_CHF', 'AUD_USD', 'GBP_CHF',
                             'EUR_JPY', 'USD_JPY', 'EUR_CHF', 'GBP_USD', 'GBP_JPY',
                             'EUR_GBP', 'AUD_JPY', 'USD_CAD', 'EUR_AUD']
        logging.info(f"Using hardcoded valid instruments: {self.instruments}")
        self.weights = {
            'order_book': 0.40,
            'position_book': 0.35,
            'currency_sentiment': 0.10,
            'retail_profitability': 0.15
        }
        self.thresholds = {
            'strong_bullish': 1.0,
            'bullish': 0.7,
            'bearish': -0.7,
            'strong_bearish': -1.0
        }

    def pips_to_price_diff(self, instrument: str, pips: float) -> float:
        if instrument.endswith("_JPY") or instrument == 'XAU_USD':
            pip_value = 0.01
        else:
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
            return 0
        price_diff_50_pips = self.pips_to_price_diff(instrument, 50)
        lower_bound = current_price - price_diff_50_pips
        upper_bound = current_price + price_diff_50_pips
        filtered_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                            if lower_bound <= p <= upper_bound]
        if not filtered_buckets:
            return 0
        fp, fs, fl = zip(*filtered_buckets)
        total_long = sum(fl)
        total_short = sum(fs)
        if total_short > 0:
            long_short_diff_pct = ((total_long - total_short) / total_short) * 100
        else:
            long_short_diff_pct = 0 if total_long == 0 else float('inf')
        score = 0
        if long_short_diff_pct >= 20:
            score = 1
        elif long_short_diff_pct <= -20:
            score = -1
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
        filtered_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                             if lower_bound <= p <= upper_bound]
        if not filtered_buckets:
            return 0
        fp, fs, fl = zip(*filtered_buckets)
        total_long = sum(fl)
        total_short = sum(fs)
        total_positions = total_long + total_short
        score = 0
        if total_positions > 0:
            long_ratio = (total_long / total_positions) * 100
            short_ratio = (total_short / total_positions) * 100
            if long_ratio >= 65:
                score = -1
            elif short_ratio >= 65:
                score = 1
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
                if avg_sentiment > 50:
                    score = 1
                elif avg_sentiment < 50:
                    score = -1
        return score

    def analyze_retail_profitability(self) -> int:
        score = 0
        html = self.sentiment_fetcher.fetch_twitter_sentiment_data()
        twitter_chart = self.sentiment_fetcher.extract_twitter_chart_data(html)
        if twitter_chart:
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
        currency_sentiment_score = self.analyze_currency_sentiment()
        retail_profit_score = self.analyze_retail_profitability()
        weighted_total = (order_score * self.weights['order_book'] +
                          position_score * self.weights['position_book'] +
                          currency_sentiment_score * self.weights['currency_sentiment'] +
                          retail_profit_score * self.weights['retail_profitability'])
        details = {
            "order_score": order_score,
            "position_score": position_score,
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
                signal = TradingSignal(instrument)
                signal.total_score = total_score
                signal.order_book_score = details['order_score']
                signal.position_book_score = details['position_score']
                signal.currency_sentiment_score = details['currency_sentiment_score']
                signal.retail_profit_score = details['retail_profit_score']
                if current_price:
                    signal.price = current_price
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
                decisions[instrument] = {
                    "instrument": instrument,
                    "decision": "Error",
                    "total_score": 0.0,
                    "details": {
                        "order_score": 0,
                        "position_score": 0,
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
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2E3B4E')
        self.style.configure('TLabel', background='#2E3B4E', foreground='white', font=('Segoe UI', 10))
        self.style.configure('TButton', background='#4A6491', foreground='white', font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#FFFFFF')
        self.style.configure('Status.TLabel', font=('Segoe UI', 10), foreground='#AAAAAA')
        self.style.configure('Treeview', background='#374B61', foreground='white', fieldbackground='#374B61', font=('Segoe UI', 9))
        self.style.configure('Treeview.Heading', background='#2E3B4E', foreground='white', font=('Segoe UI', 10, 'bold'))
        self.style.map('Treeview', background=[('selected', '#4A6491')])
        self.style.configure('TNotebook', background='#2E3B4E', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#374B61', foreground='white', padding=[10, 2])
        self.style.map('TNotebook.Tab', background=[('selected', '#4A6491')])
        # Variable for Top N (user adjustable, default: 3)
        self.top_n_var = tk.StringVar(value="3")
        self.create_header_frame()
        self.initialize_trading_components()
        self.history = SignalHistory(max_history=100)
        self.selected_instrument = tk.StringVar()
        self.selected_instrument.set(self.trading_strategy.instruments[0])
        self.selected_instrument.trace_add("write", self.on_instrument_selected)
        self.create_main_frame()
        self.create_footer_frame()
        # Start periodic update
        self.schedule_refresh()

    def create_header_frame(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        title_label = ttk.Label(header_frame, text="Forex Trading Robot", style='Header.TLabel')
        title_label.pack(side=tk.LEFT, padx=5)
        self.account_label = ttk.Label(header_frame, text="Account: Loading...", style='Status.TLabel')
        self.account_label.pack(side=tk.LEFT, padx=20)
        self.status_label = ttk.Label(header_frame, text="Initializing...", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=5)
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)

    def create_main_frame(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.detailed_analysis_tab = ttk.Frame(self.notebook)
        self.signal_history_tab = ttk.Frame(self.notebook)
        self.order_book_tab = ttk.Frame(self.notebook)
        self.position_book_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.detailed_analysis_tab, text="Detailed Analysis")
        self.notebook.add(self.signal_history_tab, text="Signal History")
        self.notebook.add(self.order_book_tab, text="Order Book")
        self.notebook.add(self.position_book_tab, text="Position Book")
        self.create_dashboard_tab()
        self.create_detailed_analysis_tab()
        self.create_signal_history_tab()
        self.create_order_book_tab()
        self.create_position_book_tab()

    def create_dashboard_tab(self):
        left_frame = ttk.Frame(self.dashboard_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        right_frame = ttk.Frame(self.dashboard_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        decisions_frame = ttk.LabelFrame(left_frame, text="Trade Decisions", padding=10)
        decisions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        columns = ('Instrument', 'Decision', 'Score', 'Order Book', 'Position Book', 'Currency Sentiment', 'Retail Profit')
        self.decisions_tree = ttk.Treeview(decisions_frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.decisions_tree.heading(col, text=col)
            width = 100 if col in ('Instrument', 'Decision') else 70
            self.decisions_tree.column(col, width=width, anchor=tk.CENTER)
        scrollbar_y = ttk.Scrollbar(decisions_frame, orient=tk.VERTICAL, command=self.decisions_tree.yview)
        scrollbar_x = ttk.Scrollbar(decisions_frame, orient=tk.HORIZONTAL, command=self.decisions_tree.xview)
        self.decisions_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.decisions_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.decisions_tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        charts_frame = ttk.LabelFrame(left_frame, text="Trading Signals History", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        details_frame = ttk.LabelFrame(right_frame, text="Trade Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, background='#374B61', foreground='white')
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.configure(state='disabled')
        logs_frame = ttk.LabelFrame(right_frame, text="System Logs", padding=10)
        logs_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.logs_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, background='#374B61', foreground='white')
        self.logs_text.pack(fill=tk.BOTH, expand=True)
        self.logs_text.configure(state='disabled')
        log_handler = LogHandler(self.logs_text)
        logger = logging.getLogger()
        logger.addHandler(log_handler)
        logger.setLevel(logging.INFO)

    def create_detailed_analysis_tab(self):
        control_frame = ttk.Frame(self.detailed_analysis_tab)
        control_frame.pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        self.instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument,
                                             values=self.trading_strategy.instruments, state="readonly")
        self.instrument_combo.pack(side=tk.LEFT, padx=5)
        refresh_button = ttk.Button(control_frame, text="Refresh Analysis", command=self.refresh_detailed_analysis)
        refresh_button.pack(side=tk.LEFT, padx=10)
        charts_frame = ttk.Frame(self.detailed_analysis_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        left_charts = ttk.Frame(charts_frame)
        left_charts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        right_charts = ttk.Frame(charts_frame)
        right_charts.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        signal_history_frame = ttk.LabelFrame(left_charts, text="Signal History", padding=10)
        signal_history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        component_analysis_frame = ttk.LabelFrame(left_charts, text="Component Analysis", padding=10)
        component_analysis_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        orderbook_frame = ttk.LabelFrame(right_charts, text="Order Book Analysis", padding=10)
        orderbook_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        positionbook_frame = ttk.LabelFrame(right_charts, text="Position Book Analysis", padding=10)
        positionbook_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.signal_history_fig = Figure(figsize=(5, 3), dpi=100)
        self.signal_history_canvas = FigureCanvasTkAgg(self.signal_history_fig, master=signal_history_frame)
        self.signal_history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.component_analysis_fig = Figure(figsize=(5, 3), dpi=100)
        self.component_analysis_canvas = FigureCanvasTkAgg(self.component_analysis_fig, master=component_analysis_frame)
        self.component_analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # --- Detailed Order Book Analysis in Detailed Analysis Tab ---
        self.detail_order_book_fig = Figure(figsize=(5, 4), dpi=100)
        self.detail_order_book_canvas = FigureCanvasTkAgg(self.detail_order_book_fig, master=orderbook_frame)
        self.detail_order_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # --- Detailed Position Book Analysis in Detailed Analysis Tab ---
        self.detail_position_book_fig = Figure(figsize=(5, 4), dpi=100)
        self.detail_position_book_canvas = FigureCanvasTkAgg(self.detail_position_book_fig, master=positionbook_frame)
        self.detail_position_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_signal_history_tab(self):
        control_frame = ttk.Frame(self.signal_history_tab)
        control_frame.pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        signal_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument,
                                               values=self.trading_strategy.instruments, state="readonly")
        signal_instrument_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="History Length:").pack(side=tk.LEFT, padx=(20, 5))
        self.history_length_var = tk.StringVar(value="20")
        history_length_combo = ttk.Combobox(control_frame, textvariable=self.history_length_var,
                                            values=["10", "20", "50", "100"], width=5, state="readonly")
        history_length_combo.pack(side=tk.LEFT, padx=5)
        refresh_button = ttk.Button(control_frame, text="Refresh History", command=self.refresh_signal_history)
        refresh_button.pack(side=tk.LEFT, padx=10)
        main_frame = ttk.Frame(self.signal_history_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        chart_frame = ttk.LabelFrame(main_frame, text="Signal History Chart", padding=10)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        table_frame = ttk.LabelFrame(main_frame, text="Signal History Table", padding=10)
        table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.history_fig = Figure(figsize=(6, 4), dpi=100)
        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=chart_frame)
        self.history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        columns = ('Time', 'Decision', 'Total Score', 'Price', 'Order Book', 'Position Book', 'Currency Sentiment', 'Retail Profit')
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        for col in columns:
            self.history_tree.heading(col, text=col)
            width = 150 if col == 'Time' else 100 if col == 'Decision' else 70
            self.history_tree.column(col, width=width, anchor=tk.CENTER)
        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.history_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    def create_order_book_tab(self):
        control_frame = ttk.Frame(self.order_book_tab)
        control_frame.pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        orderbook_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument,
                                                  values=self.trading_strategy.instruments, state="readonly")
        orderbook_instrument_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Pip Range:").pack(side=tk.LEFT, padx=5)
        self.pip_range_var = tk.StringVar(value="50")
        pip_range_combo = ttk.Combobox(control_frame, textvariable=self.pip_range_var,
                                       values=["30", "50", "100", "150"],
                                       state="readonly", width=5)
        pip_range_combo.pack(side=tk.LEFT, padx=5)
        # Adding Top N selector for Order Book
        ttk.Label(control_frame, text="Top N:").pack(side=tk.LEFT, padx=5)
        top_n_combo = ttk.Combobox(control_frame, textvariable=self.top_n_var,
                                   values=["1", "2", "3", "4", "5"],
                                   state="readonly", width=5)
        top_n_combo.pack(side=tk.LEFT, padx=5)
        refresh_button = ttk.Button(control_frame, text="Refresh Order Book", command=self.refresh_order_book)
        refresh_button.pack(side=tk.LEFT, padx=10)
        main_frame = ttk.Frame(self.order_book_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        chart_frame = ttk.LabelFrame(main_frame, text="Order Book Visualization", padding=10)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        zoom_button_frame = ttk.Frame(chart_frame)
        zoom_button_frame.pack(fill=tk.X, pady=(0, 5))
        self.zoom_in_button_ob = ttk.Button(zoom_button_frame, text="Zoom In")
        self.zoom_in_button_ob.pack(side=tk.LEFT, padx=5)
        self.zoom_out_button_ob = ttk.Button(zoom_button_frame, text="Zoom Out")
        self.zoom_out_button_ob.pack(side=tk.LEFT, padx=5)
        self.reset_zoom_button_ob = ttk.Button(zoom_button_frame, text="Reset Zoom")
        self.reset_zoom_button_ob.pack(side=tk.LEFT, padx=5)
        # Bind zoom button commands for Order Book Tab
        self.zoom_in_button_ob.config(command=self.on_zoom_in_ob)
        self.zoom_out_button_ob.config(command=self.on_zoom_out_ob)
        self.reset_zoom_button_ob.config(command=self.on_reset_zoom_ob)
        analysis_frame = ttk.LabelFrame(main_frame, text="Order Book Analysis", padding=10)
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0,5))
        self.order_book_fig = Figure(figsize=(5, 4), dpi=100)
        self.order_book_canvas = FigureCanvasTkAgg(self.order_book_fig, master=chart_frame)
        self.order_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.order_book_analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD,
                                                                  background='#374B61', foreground='white')
        self.order_book_analysis_text.pack(fill=tk.BOTH, expand=True)
        self.order_book_analysis_text.configure(state='disabled')

    def create_position_book_tab(self):
        control_frame = ttk.Frame(self.position_book_tab)
        control_frame.pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
        positionbook_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument,
                                                     values=self.trading_strategy.instruments, state="readonly")
        positionbook_instrument_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Pip Range:").pack(side=tk.LEFT, padx=5)
        pip_range_combo_pb = ttk.Combobox(control_frame, textvariable=self.pip_range_var,
                                          values=["30", "50", "100", "150"],
                                          state="readonly", width=5)
        pip_range_combo_pb.pack(side=tk.LEFT, padx=5)
        # Adding Top N selector for Position Book
        ttk.Label(control_frame, text="Top N:").pack(side=tk.LEFT, padx=5)
        top_n_combo_pb = ttk.Combobox(control_frame, textvariable=self.top_n_var,
                                      values=["1", "2", "3", "4", "5"],
                                      state="readonly", width=5)
        top_n_combo_pb.pack(side=tk.LEFT, padx=5)
        refresh_button = ttk.Button(control_frame, text="Refresh Position Book", command=self.refresh_position_book)
        refresh_button.pack(side=tk.LEFT, padx=10)
        main_frame = ttk.Frame(self.position_book_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        chart_frame = ttk.LabelFrame(main_frame, text="Position Book Visualization", padding=10)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        zoom_button_frame = ttk.Frame(chart_frame)
        zoom_button_frame.pack(fill=tk.X, pady=(0, 5))
        self.zoom_in_button_pb = ttk.Button(zoom_button_frame, text="Zoom In")
        self.zoom_in_button_pb.pack(side=tk.LEFT, padx=5)
        self.zoom_out_button_pb = ttk.Button(zoom_button_frame, text="Zoom Out")
        self.zoom_out_button_pb.pack(side=tk.LEFT, padx=5)
        self.reset_zoom_button_pb = ttk.Button(zoom_button_frame, text="Reset Zoom")
        self.reset_zoom_button_pb.pack(side=tk.LEFT, padx=5)
        # Bind zoom button commands for Position Book Tab
        self.zoom_in_button_pb.config(command=self.on_zoom_in_pb)
        self.zoom_out_button_pb.config(command=self.on_zoom_out_pb)
        self.reset_zoom_button_pb.config(command=self.on_reset_zoom_pb)
        analysis_frame = ttk.LabelFrame(main_frame, text="Position Book Analysis", padding=10)
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0,5))
        self.position_book_fig = Figure(figsize=(5, 4), dpi=100)
        self.position_book_canvas = FigureCanvasTkAgg(self.position_book_fig, master=chart_frame)
        self.position_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.position_book_analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD,
                                                                     background='#374B61', foreground='white')
        self.position_book_analysis_text.pack(fill=tk.BOTH, expand=True)
        self.position_book_analysis_text.configure(state='disabled')

    def create_footer_frame(self):
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, padx=10, pady=10)
        refresh_button = ttk.Button(footer_frame, text="Refresh Now", command=self.refresh_data)
        refresh_button.pack(side=tk.LEFT, padx=5)
        export_button = ttk.Button(footer_frame, text="Export Data", command=self.export_data)
        export_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(footer_frame, text="Update Interval:").pack(side=tk.LEFT, padx=(20, 5))
        self.interval_var = tk.StringVar(value="60")
        interval_combobox = ttk.Combobox(footer_frame, textvariable=self.interval_var,
                                          values=["30", "60", "120", "300", "600"],
                                          width=5, state="readonly")
        interval_combobox.pack(side=tk.LEFT, padx=5)
        ttk.Label(footer_frame, text="seconds").pack(side=tk.LEFT, padx=5)
        self.last_update_label = ttk.Label(footer_frame, text="Last Update: Never", style='Status.TLabel')
        self.last_update_label.pack(side=tk.RIGHT, padx=5)

    def initialize_trading_components(self):
        try:
            self.oanda_fetcher = OandaDataFetcher()
            self.sentiment_fetcher = SentimentDataFetcher()
            self.trading_strategy = TradingStrategy(self.oanda_fetcher, self.sentiment_fetcher)
            self.update_account_info()
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

    def schedule_refresh(self):
        self.refresh_data()
        try:
            interval = int(self.interval_var.get()) * 1000  # milliseconds
        except ValueError:
            interval = 60000
        self.root.after(interval, self.schedule_refresh)

    def refresh_data(self):
        try:
            self.update_status("Analyzing trading data...")
            all_decisions = self.trading_strategy.decide_trade()
            for instrument, data in all_decisions.items():
                if "signal" in data and data["signal"]:
                    self.history.add_signal(data["signal"])
            self.update_decisions_table(all_decisions)
            self.update_details_text(all_decisions)
            self.update_chart(all_decisions)
            selected = self.selected_instrument.get()
            if selected:
                self.refresh_detailed_analysis()
                self.refresh_signal_history()
                self.refresh_order_book()
                self.refresh_position_book()
            if random.random() < 0.1:
                self.update_account_info()
            self.update_last_update_time()
            self.update_status("Trading analysis completed")
        except Exception as e:
            logging.error(f"Error refreshing data: {e}")
            logging.debug(traceback.format_exc())
            self.update_status(f"Error: {e}")

    def update_decisions_table(self, decisions: Dict[str, Dict[str, Any]]):
        for item in self.decisions_tree.get_children():
            self.decisions_tree.delete(item)
        for instrument, data in decisions.items():
            decision = data["decision"]
            score = data["total_score"]
            details = data["details"]
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
            values = (
                instrument,
                decision,
                f"{score:.2f}",
                f"{details['order_score']:.1f}",
                f"{details['position_score']:.1f}",
                f"{details['currency_sentiment_score']:.1f}",
                f"{details['retail_profit_score']:.1f}"
            )
            self.decisions_tree.insert("", tk.END, values=values, tags=(row_tag,))
        self.decisions_tree.tag_configure("strong_long", background="#008000")
        self.decisions_tree.tag_configure("long", background="#4CAF50")
        self.decisions_tree.tag_configure("strong_short", background="#8B0000")
        self.decisions_tree.tag_configure("short", background="#F44336")
        self.decisions_tree.tag_configure("neutral", background="#607D8B")
        self.decisions_tree.tag_configure("error", background="#424242")

    def update_details_text(self, decisions: Dict[str, Dict[str, Any]]):
        self.details_text.configure(state='normal')
        self.details_text.delete(1.0, tk.END)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.details_text.insert(tk.END, f"Trading Analysis at {now}\n", "header")
        self.details_text.insert(tk.END, "=" * 50 + "\n\n")
        self.details_text.insert(tk.END, "Signal Weighting:\n", "subheader")
        self.details_text.insert(tk.END, f"• Order Book: {self.trading_strategy.weights['order_book']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Position Book: {self.trading_strategy.weights['position_book']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Currency-Level Sentiment: {self.trading_strategy.weights['currency_sentiment']*100:.0f}%\n")
        self.details_text.insert(tk.END, f"• Retail Profitability: {self.trading_strategy.weights['retail_profitability']*100:.0f}%\n\n")
        self.details_text.insert(tk.END, "Decision Thresholds:\n", "subheader")
        self.details_text.insert(tk.END, f"• Total score > {self.trading_strategy.thresholds['strong_bullish']}: Strong Bullish\n")
        self.details_text.insert(tk.END, f"• Total score > {self.trading_strategy.thresholds['bullish']}: Bullish\n")
        self.details_text.insert(tk.END, f"• Total score < {self.trading_strategy.thresholds['strong_bearish']}: Strong Bearish\n")
        self.details_text.insert(tk.END, f"• Total score < {self.trading_strategy.thresholds['bearish']}: Bearish\n")
        self.details_text.insert(tk.END, "• Otherwise: No Trade\n\n")
        decision_counts = {"Strong Bullish": 0, "Bullish": 0, "No Trade": 0, "Bearish": 0, "Strong Bearish": 0, "Error": 0}
        for data in decisions.values():
            decision = data["decision"]
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
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
        tradable_signals = {k: v for k, v in decisions.items()
                            if v["decision"] in ["Strong Bullish", "Bullish", "Strong Bearish", "Bearish"]}
        if tradable_signals:
            self.details_text.insert(tk.END, "Tradable Signals:\n", "subheader")
            for instrument, data in tradable_signals.items():
                decision = data["decision"]
                score = data["total_score"]
                details = data["details"]
                price = data.get("price", 0.0)
                self.details_text.insert(tk.END, f"{instrument} ", "instrument_header")
                if price:
                    self.details_text.insert(tk.END, f"@ {price:.5f}\n", "price")
                else:
                    self.details_text.insert(tk.END, "\n", "price")
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
                self.details_text.insert(tk.END, "Components: ")
                self.details_text.insert(tk.END, f"OB: {details['order_score']} | ")
                self.details_text.insert(tk.END, f"PB: {details['position_score']} | ")
                self.details_text.insert(tk.END, f"CS: {details['currency_sentiment_score']} | ")
                self.details_text.insert(tk.END, f"RP: {details['retail_profit_score']}\n")
                self.details_text.insert(tk.END, "\n")
        else:
            self.details_text.insert(tk.END, "No tradable signals at this time.\n", "neutral")
        self.details_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.details_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.details_text.tag_configure("instrument_header", font=("Segoe UI", 11, "bold"))
        self.details_text.tag_configure("price", font=("Segoe UI", 10, "italic"))
        self.details_text.tag_configure("strong_long", foreground="#008000")
        self.details_text.tag_configure("long", foreground="#4CAF50")
        self.details_text.tag_configure("strong_short", foreground="#8B0000")
        self.details_text.tag_configure("short", foreground="#F44336")
        self.details_text.tag_configure("neutral", foreground="#607D8B")
        self.details_text.tag_configure("error", foreground="#424242")
        self.details_text.configure(state='disabled')

    def update_chart(self, decisions: Dict[str, Dict[str, Any]]):
        self.fig.clear()
        ax_all = self.fig.add_subplot(211)
        ax_detail = self.fig.add_subplot(212)
        instruments_with_history = []
        for instrument in self.trading_strategy.instruments:
            signals = self.history.get_signals(instrument, count=20)
            if len(signals) >= 2:
                instruments_with_history.append(instrument)
        for instrument in instruments_with_history:
            signals = self.history.get_signals(instrument, count=20)
            timestamps = [s.timestamp for s in signals]
            scores = [s.total_score for s in signals]
            color = 'gray'
            if signals and signals[-1].decision == "Strong Bullish":
                color = '#008000'
            elif signals and signals[-1].decision == "Bullish":
                color = '#4CAF50'
            elif signals and signals[-1].decision == "Strong Bearish":
                color = '#8B0000'
            elif signals and signals[-1].decision == "Bearish":
                color = '#F44336'
            ax_all.plot(timestamps, scores, marker='o', linestyle='-', linewidth=1.5,
                        markersize=4, label=instrument, color=color, alpha=0.7)
        ax_all.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000',
                       linestyle='--', alpha=0.7, label='Strong Bullish')
        ax_all.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50',
                       linestyle='--', alpha=0.7, label='Bullish')
        ax_all.axhline(y=self.trading_strategy.thresholds['bearish'], color='#F44336',
                       linestyle='--', alpha=0.7, label='Bearish')
        ax_all.axhline(y=self.trading_strategy.thresholds['strong_bearish'], color='#8B0000',
                       linestyle='--', alpha=0.7, label='Strong Bearish')
        ax_all.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_all.set_title('All Instruments Trading Signals')
        ax_all.set_ylabel('Signal Score')
        ax_all.set_ylim(-2, 2)
        ax_all.grid(True, alpha=0.3)
        ax_all.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax_all.get_xticklabels(), rotation=45, ha='right')
        tradable_instruments = [k for k, v in decisions.items()
                                if v["decision"] in ["Strong Bullish", "Bullish", "Strong Bearish", "Bearish"]]
        if tradable_instruments:
            detail_instruments = tradable_instruments[:5]
            components = ['Order Book', 'Position Book', 'Currency Sentiment', 'Retail Profit']
            x = np.arange(len(components))
            width = 0.15
            for i, instrument in enumerate(detail_instruments):
                data = decisions[instrument]
                details = data["details"]
                component_scores = [
                    details['order_score'] * self.trading_strategy.weights['order_book'],
                    details['position_score'] * self.trading_strategy.weights['position_book'],
                    details['currency_sentiment_score'] * self.trading_strategy.weights['currency_sentiment'],
                    details['retail_profit_score'] * self.trading_strategy.weights['retail_profitability']
                ]
                color = 'gray'
                if data["decision"] == "Strong Bullish":
                    color = '#008000'
                elif data["decision"] == "Bullish":
                    color = '#4CAF50'
                elif data["decision"] == "Strong Bearish":
                    color = '#8B0000'
                elif data["decision"] == "Bearish":
                    color = '#F44336'
                offset = width * (i - len(detail_instruments)/2 + 0.5)
                ax_detail.bar(x + offset, component_scores, width, label=instrument, color=color, alpha=0.7)
            ax_detail.set_title('Signal Components of Tradable Instruments')
            ax_detail.set_ylabel('Weighted Score')
            ax_detail.set_xticks(x)
            ax_detail.set_xticklabels(components, rotation=45, ha='right')
            ax_detail.legend(loc='upper right')
            ax_detail.grid(True, axis='y', alpha=0.3)
        else:
            ax_detail.text(0.5, 0.5, 'No tradable signals at this time',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_detail.transAxes, fontsize=12)
            ax_detail.set_xticks([])
            ax_detail.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()

    def on_tree_select(self, event):
        selected_items = self.decisions_tree.selection()
        if not selected_items:
            return
        item_id = self.decisions_tree.selection()[0]
        item_values = self.decisions_tree.item(item_id, 'values')
        instrument = item_values[0]
        self.selected_instrument.set(instrument)
        self.notebook.select(self.detailed_analysis_tab)

    def on_instrument_selected(self, *args):
        instrument = self.selected_instrument.get()
        if instrument:
            self.refresh_detailed_analysis()
            self.refresh_signal_history()
            self.refresh_order_book()
            self.refresh_position_book()

    def refresh_detailed_analysis(self):
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        try:
            latest_signal = self.history.get_latest_signal(instrument)
            if not latest_signal:
                return
            self.update_signal_history_chart(instrument)
            self.update_component_analysis_chart(latest_signal)
            self.update_detailed_orderbook_chart(instrument)
            self.update_detailed_positionbook_chart(instrument)
        except Exception as e:
            logging.error(f"Error refreshing detailed analysis: {e}")
            logging.debug(traceback.format_exc())

    def update_signal_history_chart(self, instrument: str):
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
        self.signal_history_fig.clear()
        ax = self.signal_history_fig.add_subplot(111)
        timestamps = [s.timestamp for s in signals]
        total_scores = [s.total_score for s in signals]
        ax.plot(timestamps, total_scores, marker='o', linestyle='-', linewidth=2,
                label='Total Score', color='blue', zorder=5)
        ax.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000',
                   linestyle='--', alpha=0.7, label='Strong Bullish')
        ax.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50',
                   linestyle='--', alpha=0.7, label='Bullish')
        ax.axhline(y=self.trading_strategy.thresholds['bearish'], color='#F44336',
                   linestyle='--', alpha=0.7, label='Bearish')
        ax.axhline(y=self.trading_strategy.thresholds['strong_bearish'], color='#8B0000',
                   linestyle='--', alpha=0.7, label='Strong Bearish')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax.set_title(f'Signal History for {instrument}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Score')
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        self.signal_history_fig.tight_layout()
        self.signal_history_canvas.draw()

    def update_component_analysis_chart(self, signal: TradingSignal):
        self.component_analysis_fig.clear()
        ax = self.component_analysis_fig.add_subplot(111)
        components = ['Order Book', 'Position Book', 'Currency Sentiment', 'Retail Profit']
        weights = [
            self.trading_strategy.weights['order_book'],
            self.trading_strategy.weights['position_book'],
            self.trading_strategy.weights['currency_sentiment'],
            self.trading_strategy.weights['retail_profitability']
        ]
        raw_scores = [
            signal.order_book_score,
            signal.position_book_score,
            signal.currency_sentiment_score,
            signal.retail_profit_score
        ]
        weighted_scores = [raw * weight for raw, weight in zip(raw_scores, weights)]
        x = np.arange(len(components))
        width = 0.35
        ax.bar(x - width/2, raw_scores, width, label='Raw Score', color='skyblue')
        ax.bar(x + width/2, weighted_scores, width, label='Weighted Score', color='navy')
        for i, (v1, v2) in enumerate(zip(raw_scores, weighted_scores)):
            ax.text(i - width/2, v1 + 0.1, f"{v1}", ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, v2 + 0.1, f"{v2:.2f}", ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Component Analysis for {signal.instrument}')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        max_score = max(max(raw_scores, default=0), max(weighted_scores, default=0))
        min_score = min(min(raw_scores, default=0), min(weighted_scores, default=0))
        padding = 0.5
        ax.set_ylim(min(min_score - padding, -2), max(max_score + padding, 2))
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
        ax.text(0.02, 0.95, f"Total Score: {signal.total_score:.2f}",
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        self.component_analysis_fig.tight_layout()
        self.component_analysis_canvas.draw()

    def update_detailed_orderbook_chart(self, instrument: str):
        self.detail_order_book_fig.clear()
        gs = self.detail_order_book_fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        self.ax1_ob = self.detail_order_book_fig.add_subplot(gs[0])
        self.ax2_ob = self.detail_order_book_fig.add_subplot(gs[1], sharex=self.ax1_ob)
        self.ax3_ob = self.detail_order_book_fig.add_subplot(gs[2], sharex=self.ax1_ob)

        orderbook_data = self.oanda_fetcher.fetch_orderbook(instrument)
        if not orderbook_data or 'buckets' not in orderbook_data:
            self.ax1_ob.clear()
            self.ax2_ob.clear()
            self.ax3_ob.clear()
            self.ax1_ob.text(0.5, 0.5, f'No order book data available for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_ob.transAxes, fontsize=12)
            self.detail_order_book_canvas.draw()
            return
        buckets = orderbook_data['buckets']
        try:
            current_price = float(orderbook_data.get('price', 0))
        except (ValueError, TypeError):
            current_price = 0
        bucket_width = float(orderbook_data.get('bucketWidth', 0))
        time_str = orderbook_data.get('time', 'N/A')
        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)
        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_percents, long_percents, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []
        if not filtered_prices:
            self.ax1_ob.text(0.5, 0.5, f'No order book data within price range for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_ob.transAxes, fontsize=12)
            self.detail_order_book_canvas.draw()
            return
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8
        else:
            bar_width = bucket_width
        self.ax1_ob.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7,
                        label='Long %', align='edge')
        self.ax1_ob.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width, color='red', alpha=0.7,
                        label='Short %', align='edge')
        diff = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        self.ax1_ob.plot(filtered_prices, diff, color='purple', alpha=0.7, label='Long-Short Diff', linewidth=2)
        if current_price > 0:
            self.ax1_ob.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
        self.initial_xlim_ob = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                                max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        self.ax1_ob.set_xlim(self.initial_xlim_ob)
        self.ax1_ob.set_title(f'Order Book for {instrument} at {time_str}')
        self.ax1_ob.set_ylabel('Percentage')
        self.ax1_ob.legend()
        self.ax1_ob.grid(True, alpha=0.3)

        cumulative_long = np.cumsum(filtered_longs)
        cumulative_short = np.cumsum(filtered_shorts)
        self.ax2_ob.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-', marker='o', label='Cumulative Long')
        self.ax2_ob.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-', marker='o', label='Cumulative Short')
        self.ax2_ob.set_ylabel('Cumulative %')
        self.ax2_ob.legend()
        self.ax2_ob.grid(True, alpha=0.3)

        diff_array = np.array(diff)
        heatmap_data = np.tile(diff_array, (10, 1))
        if diff_array.size == 0:
            norm = mcolors.Normalize(vmin=-1, vmax=1)
        elif np.all(diff_array == diff_array[0]):
            norm = mcolors.Normalize(vmin=diff_array[0]-1, vmax=diff_array[0]+1)
        else:
            max_val = max(abs(diff_array.min()), abs(diff_array.max()))
            norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        im = self.ax3_ob.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', norm=norm,
                                extent=[min(filtered_prices), max(filtered_prices), 0, 1])
        self.ax3_ob.set_title('Heatmap: Long-Short Difference')
        self.ax3_ob.set_xlabel('Price')
        self.ax3_ob.set_yticks([])
        self.detail_order_book_canvas.draw()

    def update_detailed_positionbook_chart(self, instrument: str):
        self.detail_position_book_fig.clear()
        gs = self.detail_position_book_fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        self.ax1_pb = self.detail_position_book_fig.add_subplot(gs[0])
        self.ax2_pb = self.detail_position_book_fig.add_subplot(gs[1], sharex=self.ax1_pb)
        self.ax3_pb = self.detail_position_book_fig.add_subplot(gs[2], sharex=self.ax1_pb)
        positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)
        if not positionbook_data or 'buckets' not in positionbook_data:
            self.ax1_pb.clear()
            self.ax2_pb.clear()
            self.ax3_pb.clear()
            self.ax1_pb.text(0.5, 0.5, f'No position book data available for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_pb.transAxes, fontsize=12)
            self.detail_position_book_canvas.draw()
            return

        buckets = positionbook_data['buckets']
        try:
            current_price = float(positionbook_data.get('price', 0))
        except (ValueError, TypeError):
            current_price = 0
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        time_str = positionbook_data.get('time', 'N/A')
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_counts, long_counts, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []
        if not filtered_prices:
            self.ax1_pb.text(0.5, 0.5, f'No position book data within price range for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_pb.transAxes, fontsize=12)
            self.detail_position_book_canvas.draw()
            return
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8
        else:
            bar_width = bucket_width
        self.ax1_pb.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7,
                        label='Long %', align='edge')
        self.ax1_pb.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width, color='red', alpha=0.7,
                        label='Short %', align='edge')
        diff_pct = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        self.ax1_pb.plot(filtered_prices, diff_pct, color='purple', alpha=0.7, label='Long-Short Diff', linewidth=2)
        if current_price > 0:
            self.ax1_pb.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
        self.initial_xlim_pb = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                                max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        self.ax1_pb.set_xlim(self.initial_xlim_pb)
        self.ax1_pb.set_title(f'Position Book for {instrument} at {time_str}')
        self.ax1_pb.set_xlabel('Price')
        self.ax1_pb.set_ylabel('Percentage')
        self.ax1_pb.legend()
        self.ax1_pb.grid(True, alpha=0.3)

        cumulative_long = np.cumsum(filtered_longs)
        cumulative_short = np.cumsum(filtered_shorts)
        self.ax2_pb.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-', marker='o', label='Cumulative Long')
        self.ax2_pb.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-', marker='o', label='Cumulative Short')
        self.ax2_pb.set_ylabel('Cumulative %')
        self.ax2_pb.legend()
        self.ax2_pb.grid(True, alpha=0.3)

        diff_array = np.array(diff_pct)
        heatmap_data = np.tile(diff_array, (10, 1))
        if diff_array.size == 0:
            norm = mcolors.Normalize(vmin=-1, vmax=1)
        elif np.all(diff_array == diff_array[0]):
            norm = mcolors.Normalize(vmin=diff_array[0]-1, vmax=diff_array[0]+1)
        else:
            max_val = max(abs(diff_array.min()), abs(diff_array.max()))
            norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        im = self.ax3_pb.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', norm=norm,
                                 extent=[min(filtered_prices), max(filtered_prices), 0, 1])
        self.ax3_pb.set_title('Heatmap: Long-Short Difference')
        self.ax3_pb.set_xlabel('Price')
        self.ax3_pb.set_yticks([])
        self.detail_position_book_canvas.draw()

    def refresh_signal_history(self):
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        try:
            try:
                history_length = int(self.history_length_var.get())
            except ValueError:
                history_length = 20
            signals = self.history.get_signals(instrument, count=history_length)
            self.update_history_chart(instrument, signals)
            self.update_history_table(signals)
        except Exception as e:
            logging.error(f"Error refreshing signal history: {e}")
            logging.debug(traceback.format_exc())

    def update_history_chart(self, instrument: str, signals: List[TradingSignal]):
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
        gs = self.history_fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = self.history_fig.add_subplot(gs[0])
        ax2 = self.history_fig.add_subplot(gs[1], sharex=ax1)
        timestamps = [s.timestamp for s in signals]
        total_scores = [s.total_score for s in signals]
        order_book_scores = [s.order_book_score for s in signals]
        position_book_scores = [s.position_book_score for s in signals]
        currency_sentiment_scores = [s.currency_sentiment_score for s in signals]
        retail_profit_scores = [s.retail_profit_score for s in signals]
        prices = [s.price for s in signals]
        ax1.plot(timestamps, total_scores, marker='o', linestyle='-', linewidth=2,
                 label='Total Score', color='blue', zorder=10)
        ax1.plot(timestamps, order_book_scores, marker='x', linestyle='--', linewidth=1,
                 label='Order Book', color='green', alpha=0.7)
        ax1.plot(timestamps, position_book_scores, marker='s', linestyle='--', linewidth=1,
                 label='Position Book', color='purple', alpha=0.7)
        ax1.plot(timestamps, currency_sentiment_scores, marker='+', linestyle='--', linewidth=1,
                 label='Currency Sentiment', color='orange', alpha=0.7)
        ax1.plot(timestamps, retail_profit_scores, marker='*', linestyle='--', linewidth=1,
                 label='Retail Profit', color='gray', alpha=0.7)
        ax1.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000',
                 linestyle='--', alpha=0.7, label='Strong Bullish')
        ax1.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50',
                 linestyle='--', alpha=0.7, label='Bullish')
        ax1.axhline(y=self.trading_strategy.thresholds['bearish'], color='#F44336',
                 linestyle='--', alpha=0.7, label='Bearish')
        ax1.axhline(y=self.trading_strategy.thresholds['strong_bearish'], color='#8B0000',
                 linestyle='--', alpha=0.7, label='Strong Bearish')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax1.set_title(f'Signal History for {instrument}')
        ax1.set_ylabel('Signal Score')
        ax1.set_ylim(-2.5, 2.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize='small')
        ax2.plot(timestamps, prices, marker='o', linestyle='-', linewidth=1.5,
                 label='Price', color='purple', zorder=5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        self.history_fig.tight_layout()
        self.history_canvas.draw()

    def update_history_table(self, signals: List[TradingSignal]):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        if not signals:
            return
        for signal in reversed(signals):
            row_tag = "neutral"
            if signal.decision == "Strong Bullish":
                row_tag = "strong_long"
            elif signal.decision == "Bullish":
                row_tag = "long"
            elif signal.decision == "Strong Bearish":
                row_tag = "strong_short"
            elif signal.decision == "Bearish":
                row_tag = "short"
            values = (
                signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                signal.decision,
                f"{signal.total_score:.2f}",
                f"{signal.price:.5f}" if signal.price else "N/A",
                f"{signal.order_book_score}",
                f"{signal.position_book_score}",
                f"{signal.currency_sentiment_score}",
                f"{signal.retail_profit_score}"
            )
            self.history_tree.insert("", tk.END, values=values, tags=(row_tag,))
        self.history_tree.tag_configure("strong_long", background="#008000")
        self.history_tree.tag_configure("long", background="#4CAF50")
        self.history_tree.tag_configure("strong_short", background="#8B0000")
        self.history_tree.tag_configure("short", background="#F44336")
        self.history_tree.tag_configure("neutral", background="#607D8B")

    def refresh_order_book(self):
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        try:
            orderbook_data = self.oanda_fetcher.fetch_orderbook(instrument)
            self.update_order_book_visualization(instrument, orderbook_data)
            self.update_order_book_analysis(instrument, orderbook_data)
        except Exception as e:
            logging.error(f"Error refreshing order book: {e}")
            logging.debug(traceback.format_exc())

    def update_order_book_visualization(self, instrument: str, orderbook_data: Optional[Dict[str, Any]]):
        self.order_book_fig.clear()
        gs = self.order_book_fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        self.ax1_ob = self.order_book_fig.add_subplot(gs[0])
        self.ax2_ob = self.order_book_fig.add_subplot(gs[1], sharex=self.ax1_ob)
        self.ax3_ob = self.order_book_fig.add_subplot(gs[2], sharex=self.ax1_ob)
        if not orderbook_data or 'buckets' not in orderbook_data:
            self.ax1_ob.clear()
            self.ax2_ob.clear()
            self.ax3_ob.clear()
            self.ax1_ob.text(0.5, 0.5, f'No order book data available for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_ob.transAxes, fontsize=12)
            self.order_book_canvas.draw()
            return
        buckets = orderbook_data['buckets']
        try:
            current_price = float(orderbook_data.get('price', 0))
        except (ValueError, TypeError):
            current_price = 0
        bucket_width = float(orderbook_data.get('bucketWidth', 0))
        time_str = orderbook_data.get('time', 'N/A')
        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)
        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_percents, long_percents, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []
        if not filtered_prices:
            self.ax1_ob.text(0.5, 0.5, f'No order book data within price range for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_ob.transAxes, fontsize=12)
            self.order_book_canvas.draw()
            return
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8
        else:
            bar_width = bucket_width
        self.ax1_ob.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7,
                        label='Long %', align='edge')
        self.ax1_ob.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width, color='red', alpha=0.7,
                        label='Short %', align='edge')
        diff = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        self.ax1_ob.plot(filtered_prices, diff, color='purple', alpha=0.7, label='Long-Short Diff', linewidth=2)
        if current_price > 0:
            self.ax1_ob.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
        self.initial_xlim_ob = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                                max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        self.ax1_ob.set_xlim(self.initial_xlim_ob)
        self.ax1_ob.set_title(f'Order Book for {instrument} at {time_str}')
        self.ax1_ob.set_ylabel('Percentage')
        self.ax1_ob.legend()
        self.ax1_ob.grid(True, alpha=0.3)
        cumulative_long = np.cumsum(filtered_longs)
        cumulative_short = np.cumsum(filtered_shorts)
        self.ax2_ob.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-', marker='o', label='Cumulative Long')
        self.ax2_ob.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-', marker='o', label='Cumulative Short')
        self.ax2_ob.set_ylabel('Cumulative %')
        self.ax2_ob.legend()
        self.ax2_ob.grid(True, alpha=0.3)
        diff_array = np.array(diff)
        heatmap_data = np.tile(diff_array, (10, 1))
        if diff_array.size == 0:
            norm = mcolors.Normalize(vmin=-1, vmax=1)
        elif np.all(diff_array == diff_array[0]):
            norm = mcolors.Normalize(vmin=diff_array[0]-1, vmax=diff_array[0]+1)
        else:
            max_val = max(abs(diff_array.min()), abs(diff_array.max()))
            norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        im = self.ax3_ob.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', norm=norm,
                                extent=[min(filtered_prices), max(filtered_prices), 0, 1])
        self.ax3_ob.set_title('Heatmap: Long-Short Difference')
        self.ax3_ob.set_xlabel('Price')
        self.ax3_ob.set_yticks([])
        self.order_book_canvas.draw()

    def update_order_book_analysis(self, instrument: str, orderbook_data: Optional[Dict[str, Any]]):
        self.order_book_analysis_text.configure(state='normal')
        self.order_book_analysis_text.delete(1.0, tk.END)
        if not orderbook_data:
            self.order_book_analysis_text.insert(tk.END, f"No order book data available for {instrument}.")
            self.order_book_analysis_text.configure(state='disabled')
            return
        buckets = orderbook_data['buckets']
        try:
            current_price = float(orderbook_data.get('price', 0))
        except (ValueError, TypeError):
            current_price = 0
        bucket_width = float(orderbook_data.get('bucketWidth', 0))
        time_str = orderbook_data.get('time', 'N/A')
        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)
        pip_range = int(self.pip_range_var.get()) if self.pip_range_var.get().isdigit() else 50
        price_diff = self.trading_strategy.pips_to_price_diff(instrument, pip_range)
        lower_bound = current_price - price_diff
        upper_bound = current_price + price_diff
        # Top N support/resistance detection based on composite score:
        top_n = int(self.top_n_var.get()) if self.top_n_var.get().isdigit() else 3
        support_candidates = []
        resistance_candidates = []
        for price, sp, lp in zip(prices, short_percents, long_percents):
            if lower_bound <= price <= upper_bound:
                if price < current_price:
                    score = 2*lp - sp
                    support_candidates.append((price, score, lp, sp))
                elif price > current_price:
                    score = 2*sp - lp
                    resistance_candidates.append((price, score, sp, lp))
        support_levels = sorted(support_candidates, key=lambda x: x[1], reverse=True)[:top_n]
        resistance_levels = sorted(resistance_candidates, key=lambda x: x[1], reverse=True)[:top_n]
        self.order_book_analysis_text.insert(tk.END, f"ORDER BOOK ANALYSIS FOR {instrument}\n", "header")
        self.order_book_analysis_text.insert(tk.END, f"Data as of: {time_str}\n\n")
        self.order_book_analysis_text.insert(tk.END, "CURRENT PRICE ANALYSIS:\n", "subheader")
        self.order_book_analysis_text.insert(tk.END, f"Current Price: {current_price:.5f}\n")
        if total_short := sum(short_percents):
            long_short_ratio = sum(long_percents) / total_short
            self.order_book_analysis_text.insert(tk.END, f"Overall Long/Short Ratio: {long_short_ratio:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "Overall Long/Short Ratio: ∞ (no short orders)\n")
        self.order_book_analysis_text.insert(tk.END, f"\nORDERS WITHIN {pip_range} PIPS OF CURRENT PRICE:\n", "subheader")
        near_longs = sum(l for p, s, l in zip(prices, short_percents, long_percents) if lower_bound <= p <= upper_bound)
        near_shorts = sum(s for p, s, l in zip(prices, short_percents, long_percents) if lower_bound <= p <= upper_bound)
        self.order_book_analysis_text.insert(tk.END, f"Long Orders: {near_longs:.2f}%\n")
        self.order_book_analysis_text.insert(tk.END, f"Short Orders: {near_shorts:.2f}%\n")
        if near_shorts > 0:
            self.order_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: {(near_longs/near_shorts):.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "Near Price Long/Short Ratio: ∞ (no short orders near price)\n")
        self.order_book_analysis_text.insert(tk.END, f"\nKEY SUPPORT LEVELS (Buy Limit Orders):\n", "subheader")
        if support_levels:
            for price, score, lp, sp in support_levels:
                diff_pips = abs(current_price - price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.order_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Composite Strength: {score:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "No significant support levels detected.\n")
        self.order_book_analysis_text.insert(tk.END, f"\nKEY RESISTANCE LEVELS (Sell Limit Orders):\n", "subheader")
        if resistance_levels:
            for price, score, sp, lp in resistance_levels:
                diff_pips = abs(price - current_price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.order_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Composite Strength: {score:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "No significant resistance levels detected.\n")
        self.order_book_analysis_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.order_book_analysis_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.order_book_analysis_text.configure(state='disabled')

    def refresh_position_book(self):
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        try:
            positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)
            self.update_position_book_visualization(instrument, positionbook_data)
            self.update_position_book_analysis(instrument, positionbook_data)
        except Exception as e:
            logging.error(f"Error refreshing position book: {e}")
            logging.debug(traceback.format_exc())
            
    def update_position_book_visualization(self, instrument: str, positionbook_data: Optional[Dict[str, Any]]):
        self.position_book_fig.clear()
        gs = self.position_book_fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        self.ax1_pb = self.position_book_fig.add_subplot(gs[0])
        self.ax2_pb = self.position_book_fig.add_subplot(gs[1], sharex=self.ax1_pb)
        self.ax3_pb = self.position_book_fig.add_subplot(gs[2], sharex=self.ax1_pb)
        if not positionbook_data or 'buckets' not in positionbook_data:
            self.ax1_pb.clear()
            self.ax2_pb.clear()
            self.ax3_pb.clear()
            self.ax1_pb.text(0.5, 0.5, f'No position book data available for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_pb.transAxes, fontsize=12)
            self.position_book_canvas.draw()
            return

        buckets = positionbook_data['buckets']
        try:
            current_price = float(positionbook_data.get('price', 0))
        except (ValueError, TypeError):
            current_price = 0
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        time_str = positionbook_data.get('time', 'N/A')
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_counts, long_counts, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []
        if not filtered_prices:
            self.ax1_pb.text(0.5, 0.5, f'No position book data within price range for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_pb.transAxes, fontsize=12)
            self.position_book_canvas.draw()
            return
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8
        else:
            bar_width = bucket_width
        self.ax1_pb.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7,
                        label='Long %', align='edge')
        self.ax1_pb.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width, color='red', alpha=0.7,
                        label='Short %', align='edge')
        diff_pct = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        self.ax1_pb.plot(filtered_prices, diff_pct, color='purple', alpha=0.7, label='Long-Short Diff', linewidth=2)
        if current_price > 0:
            self.ax1_pb.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
        self.initial_xlim_pb = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                                max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        self.ax1_pb.set_xlim(self.initial_xlim_pb)
        self.ax1_pb.set_title(f'Position Book for {instrument} at {time_str}')
        self.ax1_pb.set_xlabel('Price')
        self.ax1_pb.set_ylabel('Percentage')
        self.ax1_pb.legend()
        self.ax1_pb.grid(True, alpha=0.3)
        cumulative_long = np.cumsum(filtered_longs)
        cumulative_short = np.cumsum(filtered_shorts)
        self.ax2_pb.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-', marker='o', label='Cumulative Long')
        self.ax2_pb.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-', marker='o', label='Cumulative Short')
        self.ax2_pb.set_ylabel('Cumulative %')
        self.ax2_pb.legend()
        self.ax2_pb.grid(True, alpha=0.3)
        diff_array = np.array(diff_pct)
        heatmap_data = np.tile(diff_array, (10, 1))
        if diff_array.size == 0:
            norm = mcolors.Normalize(vmin=-1, vmax=1)
        elif np.all(diff_array == diff_array[0]):
            norm = mcolors.Normalize(vmin=diff_array[0]-1, vmax=diff_array[0]+1)
        else:
            max_val = max(abs(diff_array.min()), abs(diff_array.max()))
            norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
        im = self.ax3_pb.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', norm=norm,
                                 extent=[min(filtered_prices), max(filtered_prices), 0, 1])
        self.ax3_pb.set_title('Heatmap: Long-Short Difference')
        self.ax3_pb.set_xlabel('Price')
        self.ax3_pb.set_yticks([])
        self.position_book_canvas.draw()

    def update_position_book_analysis(self, instrument: str, positionbook_data: Optional[Dict[str, Any]]):
        self.position_book_analysis_text.configure(state='normal')
        self.position_book_analysis_text.delete(1.0, tk.END)
        if not positionbook_data or 'buckets' not in positionbook_data:
            self.position_book_analysis_text.insert(tk.END, f"No position book data available for {instrument}.")
            self.position_book_analysis_text.configure(state='disabled')
            return
        buckets = positionbook_data['buckets']
        try:
            current_price = float(positionbook_data.get('price', 0))
        except (ValueError, TypeError):
            current_price = 0
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        time_str = positionbook_data.get('time', 'N/A')
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        pip_range = int(self.pip_range_var.get()) if self.pip_range_var.get().isdigit() else 50
        price_diff = self.trading_strategy.pips_to_price_diff(instrument, pip_range)
        lower_bound = current_price - price_diff
        upper_bound = current_price + price_diff
        top_n = int(self.top_n_var.get()) if self.top_n_var.get().isdigit() else 3
        support_candidates = []
        resistance_candidates = []
        for price, sp, lp in zip(prices, short_counts, long_counts):
            if lower_bound <= price <= upper_bound:
                if price < current_price:
                    score = 2*lp - sp
                    support_candidates.append((price, score, lp, sp))
                elif price > current_price:
                    score = 2*sp - lp
                    resistance_candidates.append((price, score, sp, lp))
        support_levels = sorted(support_candidates, key=lambda x: x[1], reverse=True)[:top_n]
        resistance_levels = sorted(resistance_candidates, key=lambda x: x[1], reverse=True)[:top_n]
        self.position_book_analysis_text.insert(tk.END, f"POSITION BOOK ANALYSIS FOR {instrument}\n", "header")
        self.position_book_analysis_text.insert(tk.END, f"Data as of: {time_str}\n\n")
        self.position_book_analysis_text.insert(tk.END, "OVERALL POSITION ANALYSIS:\n", "subheader")
        self.position_book_analysis_text.insert(tk.END, f"Current Price: {current_price:.5f}\n")
        total_long = sum(long_counts)
        total_short = sum(short_counts)
        if total_long + total_short > 0:
            long_percentage = (total_long / (total_long + total_short)) * 100
            short_percentage = (total_short / (total_long + total_short)) * 100
            self.position_book_analysis_text.insert(tk.END, f"Long Positions: {long_percentage:.2f}%\n")
            self.position_book_analysis_text.insert(tk.END, f"Short Positions: {short_percentage:.2f}%\n")
        else:
            self.position_book_analysis_text.insert(tk.END, f"Long Positions: 0.00%\n")
            self.position_book_analysis_text.insert(tk.END, f"Short Positions: 0.00%\n")
        self.position_book_analysis_text.insert(tk.END, f"\nPOSITIONS WITHIN {pip_range} PIPS OF CURRENT PRICE:\n", "subheader")
        near_longs = sum(l for p, s, l in zip(prices, short_counts, long_counts) if lower_bound <= p <= upper_bound)
        near_shorts = sum(s for p, s, l in zip(prices, short_counts, long_counts) if lower_bound <= p <= upper_bound)
        self.position_book_analysis_text.insert(tk.END, f"Long Positions: {near_longs:.2f}%\n")
        self.position_book_analysis_text.insert(tk.END, f"Short Positions: {near_shorts:.2f}%\n")
        if near_shorts > 0:
            self.position_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: {(near_longs/near_shorts):.2f}\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "Near Price Long/Short Ratio: ∞ (no short positions near price)\n")
        self.position_book_analysis_text.insert(tk.END, f"\nKEY SUPPORT LEVELS (Buy Limit Orders):\n", "subheader")
        if support_levels:
            for price, score, lp, sp in support_levels:
                diff_pips = abs(current_price - price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.position_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Composite Strength: {score:.2f}\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "No significant support levels detected.\n")
        self.position_book_analysis_text.insert(tk.END, f"\nKEY RESISTANCE LEVELS (Sell Limit Orders):\n", "subheader")
        if resistance_levels:
            for price, score, sp, lp in resistance_levels:
                diff_pips = abs(price - current_price) / self.trading_strategy.pips_to_price_diff(instrument, 1)
                self.position_book_analysis_text.insert(tk.END, f"Price: {price:.5f} ({diff_pips:.1f} pips away) - Composite Strength: {score:.2f}\n")
        else:
            self.position_book_analysis_text.insert(tk.END, "No significant resistance levels detected.\n")
        self.position_book_analysis_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.position_book_analysis_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.position_book_analysis_text.configure(state='disabled')

    def zoom_plot(self, ax, factor):
        current_xlim = ax.get_xlim()
        x_center = (current_xlim[0] + current_xlim[1]) / 2
        new_width = (current_xlim[1] - current_xlim[0]) * factor
        new_xlim = (x_center - new_width / 2, x_center + new_width / 2)
        ax.set_xlim(new_xlim)
        if ax == self.ax1_ob:
            try:
                self.ax2_ob.set_xlim(new_xlim)
                self.ax3_ob.set_xlim(new_xlim)
            except Exception:
                pass
            if hasattr(self, 'order_book_canvas'):
                self.order_book_canvas.draw()
            elif hasattr(self, 'detail_order_book_canvas'):
                self.detail_order_book_canvas.draw()
        elif ax == self.ax1_pb:
            try:
                self.ax2_pb.set_xlim(new_xlim)
                self.ax3_pb.set_xlim(new_xlim)
            except Exception:
                pass
            if hasattr(self, 'position_book_canvas'):
                self.position_book_canvas.draw()
            elif hasattr(self, 'detail_position_book_canvas'):
                self.detail_position_book_canvas.draw()

    def reset_zoom(self, ax, book_type):
        if book_type == "ob":
            ax.set_xlim(self.initial_xlim_ob)
            try:
                self.ax2_ob.set_xlim(self.initial_xlim_ob)
                self.ax3_ob.set_xlim(self.initial_xlim_ob)
            except Exception:
                pass
            if hasattr(self, 'order_book_canvas'):
                self.order_book_canvas.draw()
            elif hasattr(self, 'detail_order_book_canvas'):
                self.detail_order_book_canvas.draw()
        elif book_type == "pb":
            ax.set_xlim(self.initial_xlim_pb)
            try:
                self.ax2_pb.set_xlim(self.initial_xlim_pb)
                self.ax3_pb.set_xlim(self.initial_xlim_pb)
            except Exception:
                pass
            if hasattr(self, 'position_book_canvas'):
                self.position_book_canvas.draw()
            elif hasattr(self, 'detail_position_book_canvas'):
                self.detail_position_book_canvas.draw()

    # Zoom methods for Order Book Tab
    def on_zoom_in_ob(self):
        if hasattr(self, 'ax1_ob'):
            self.zoom_plot(self.ax1_ob, 0.5)

    def on_zoom_out_ob(self):
        if hasattr(self, 'ax1_ob'):
            self.zoom_plot(self.ax1_ob, 2.0)

    def on_reset_zoom_ob(self):
        if hasattr(self, 'ax1_ob'):
            self.reset_zoom(self.ax1_ob, "ob")

    # Zoom methods for Position Book Tab
    def on_zoom_in_pb(self):
        if hasattr(self, 'ax1_pb'):
            self.zoom_plot(self.ax1_pb, 0.5)

    def on_zoom_out_pb(self):
        if hasattr(self, 'ax1_pb'):
            self.zoom_plot(self.ax1_pb, 2.0)

    def on_reset_zoom_pb(self):
        if hasattr(self, 'ax1_pb'):
            self.reset_zoom(self.ax1_pb, "pb")

    def export_data(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
                                                title="Export Signal History")
        if not filename:
            return
        try:
            instrument = self.selected_instrument.get()
            signals = self.history.get_signals(instrument)
            if filename.lower().endswith(".csv"):
                with open(filename, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Instrument", "Decision", "Total Score", 
                                     "Order Book Score", "Position Book Score", 
                                     "Currency Sentiment Score", "Retail Profit Score", "Price"])
                    for s in signals:
                        writer.writerow([s.timestamp.strftime("%Y-%m-%d %H:%M:%S"), s.instrument,
                                         s.decision, s.total_score, s.order_book_score,
                                         s.position_book_score, s.currency_sentiment_score,
                                         s.retail_profit_score, s.price])
            elif filename.lower().endswith(".txt"):
                with open(filename, "w") as f:
                    f.write("Timestamp\tInstrument\tDecision\tTotal Score\tOrder Book Score\tPosition Book Score\tCurrency Sentiment Score\tRetail Profit Score\tPrice\n")
                    for s in signals:
                        f.write(f"{s.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\t{s.instrument}\t{s.decision}\t{s.total_score}\t{s.order_book_score}\t{s.position_book_score}\t{s.currency_sentiment_score}\t{s.retail_profit_score}\t{s.price}\n")
            messagebox.showinfo("Export Data", f"Data exported successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")

# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    root = tk.Tk()
    app = TradingRobotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
