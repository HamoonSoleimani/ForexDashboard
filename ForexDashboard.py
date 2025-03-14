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
import random  # Import the random module

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
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
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

    def extract_retail_table_data(self, html_content: str) -> Optional[Dict[str, Dict[str, str]]]:
        if not html_content:
            return None
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table', {'id': 'table'})
        if not tables:
            logging.error("Error: Table with id 'table' not found on retail positions page.")
            return None
        currency_data = {"overview": {}, "symbol_data": []}
        if len(tables) >= 1:
            overview_table = tables[0]
            try:
                rows = overview_table.find('tbody').find_all('tr')
                thead = overview_table.find('thead')
                if thead:
                    headers = [th.text.strip() for th in thead.find_all('th')][1:]
                else:
                    headers = []
                if not headers:
                    logging.error("Error: Table headers not found in retail positions overview table.")
                else:
                    for row in rows:
                        cells = row.find_all('td')
                        if cells:
                            metric_name = cells[0].b.text.strip() if cells[0].b else cells[0].text.strip()
                            values = [cell.text.strip() for cell in cells[1:]]
                            for i, header in enumerate(headers):
                                if header not in currency_data["overview"]:
                                    currency_data["overview"][header] = {}
                                if i < len(values):
                                    currency_data["overview"][header][metric_name] = values[i]
            except AttributeError as e:
                logging.error(f"Error parsing retail positions overview table: {e}")
        if len(tables) >= 2:
            symbol_table = tables[1]
            try:
                rows = symbol_table.find('tbody').find_all('tr')
                thead = symbol_table.find('thead')
                if thead:
                    symbol_headers = [th.text.strip() for th in thead.find_all('th')]
                else:
                    symbol_headers = []
                if symbol_headers:
                    for row in rows:
                        cells = row.find_all('td')
                        if cells:
                            symbol_entry = {}
                            for i, header in enumerate(symbol_headers):
                                if i < len(cells):
                                    symbol_entry[header] = cells[i].text.strip()
                            currency_data["symbol_data"].append(symbol_entry)
            except AttributeError as e:
                logging.error(f"Error parsing retail positions symbol table: {e}")
        return currency_data

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
          dfs = pd.read_html(io.StringIO(html_content), attrs={'id': 'outlookSymbolsTable'})
          if dfs:
              df = dfs[0]
              df.dropna(axis=1, how='all', inplace=True)
              df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
              df.columns = [str(col).strip() for col in df.columns]
              for col in ['Long %', 'Short %', 'Long Lots', 'Short Lots']:
                  if col in df.columns:
                    try:
                        df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True).astype(float)
                    except ValueError:
                        logging.warning(f"Could not convert column {col} to numeric.")
              return df.to_dict(orient='records')
          else:
              return None
      except Exception as e:
          logging.error("Error extracting Myfxbook sentiment table: %s", e)
          logging.debug(traceback.format_exc())
          return None

    def extract_myfxbook_additional_data(self, html_content: Optional[str]) -> Optional[pd.DataFrame]:
      if not html_content:
          return None
      try:
        dfs = pd.read_html(io.StringIO(html_content))
        if not dfs:
            logging.warning("No tables found in Myfxbook HTML content.")
            return None
        if len(dfs) < 2:
            logging.warning("Expected at least two tables on Myfxbook, found fewer.")
            return None
        df = dfs[1]
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if df.shape[1] >= 4:
            df = df.iloc[:, :4]
            df.columns = ['Symbol', 'Avg. Short Price / Distance From Price', 'Avg. Long Price / Distance From Price', 'Current Price']
            try:
                # Process the "Avg. Short Price / Distance From Price" column
                short_split = df['Avg. Short Price / Distance From Price'].str.split(r'\s{2,}', expand=True)
                if short_split.shape[1] < 2:
                    short_split = df['Avg. Short Price / Distance From Price'].str.split(r'\s+', expand=True).iloc[:, :2]
                df['Avg. Short Price'] = pd.to_numeric(short_split[0].str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
                df['Short Distance'] = pd.to_numeric(short_split[1].str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')

                # Process the "Avg. Long Price / Distance From Price" column
                long_split = df['Avg. Long Price / Distance From Price'].str.split(r'\s{2,}', expand=True)
                if long_split.shape[1] < 2:
                    long_split = df['Avg. Long Price / Distance From Price'].str.split(r'\s+', expand=True).iloc[:, :2]
                df['Avg. Long Price'] = pd.to_numeric(long_split[0].str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
                df['Long Distance'] = pd.to_numeric(long_split[1].str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')

                df['Current Price'] = pd.to_numeric(df['Current Price'], errors='coerce')
                df = df.drop(columns=['Avg. Short Price / Distance From Price', 'Avg. Long Price / Distance From Price'])
                df = df[['Symbol', 'Avg. Short Price', 'Short Distance', 'Avg. Long Price', 'Long Distance', 'Current Price']]
            except Exception as e:
              logging.error(f"Error processing Myfxbook additional data columns {e}")
              return None
        else:
            logging.warning(f"Expected at least 4 columns in the Myfxbook table, found {len(df.columns)}.  Column renaming skipped.")
            return None
        return df
      except Exception as e:
        logging.error(f"Error extracting additional Myfxbook data: {e}", exc_info=True)
        return None

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
            'order_book': 0.30,
            'position_book': 0.225,
            'pair_sentiment': 0.225,
            'currency_sentiment': 0.15,
            'retail_profitability': 0.10
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

    def analyze_pair_specific_sentiment(self, instrument: str) -> int:
      score = 0
      pair = instrument.replace('_', '').upper()
      html = self.sentiment_fetcher.fetch_myfxbook_sentiment_data()
      if html:
          records = self.sentiment_fetcher.extract_myfxbook_table_data(html)
          if records:
              for record in records:
                if "Symbol" in record and record["Symbol"].strip().upper() == pair:
                    try:
                      long_pct = float(record.get("Long %", 0.0))
                      if long_pct < 40.0:
                        score = 2
                      elif long_pct > 60.0:
                        score = -2
                      break
                    except (ValueError, TypeError) as e:
                      logging.warning(f"Error parsing sentiment data for {instrument}: {e}")
                      break
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
                signal = TradingSignal(instrument)
                signal.total_score = total_score
                signal.order_book_score = details['order_score']
                signal.position_book_score = details['position_score']
                signal.pair_sentiment_score = details['pair_sentiment_score']
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
        self.create_header_frame()
        self.initialize_trading_components()
        self.history = SignalHistory(max_history=100)
        self.selected_instrument = tk.StringVar()
        self.selected_instrument.set(self.trading_strategy.instruments[0])
        self.selected_instrument.trace_add("write", self.on_instrument_selected)
        self.create_main_frame()
        self.create_footer_frame()
        self.start_trading_thread()
        self.refresh_data()

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
      self.retail_positions_tab = ttk.Frame(self.notebook)
      self.community_outlook_tab = ttk.Frame(self.notebook)
      self.notebook.add(self.dashboard_tab, text="Dashboard")
      self.notebook.add(self.detailed_analysis_tab, text="Detailed Analysis")
      self.notebook.add(self.signal_history_tab, text="Signal History")
      self.notebook.add(self.order_book_tab, text="Order Book")
      self.notebook.add(self.position_book_tab, text="Position Book")
      self.notebook.add(self.retail_positions_tab, text="Retail Positions")
      self.notebook.add(self.community_outlook_tab, text="Community Outlook")
      self.create_dashboard_tab()
      self.create_detailed_analysis_tab()
      self.create_signal_history_tab()
      self.create_order_book_tab()
      self.create_position_book_tab()
      self.create_retail_positions_tab()
      self.create_community_outlook_tab()

    def create_dashboard_tab(self):
      left_frame = ttk.Frame(self.dashboard_tab)
      left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
      right_frame = ttk.Frame(self.dashboard_tab)
      right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
      decisions_frame = ttk.LabelFrame(left_frame, text="Trade Decisions", padding=10)
      decisions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
      columns = ('Instrument', 'Decision', 'Score', 'Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit')
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
      self.orderbook_fig = Figure(figsize=(5, 3), dpi=100)
      self.orderbook_canvas = FigureCanvasTkAgg(self.orderbook_fig, master=orderbook_frame)
      self.orderbook_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
      self.positionbook_fig = Figure(figsize=(5, 3), dpi=100)
      self.positionbook_canvas = FigureCanvasTkAgg(self.positionbook_fig, master=positionbook_frame)
      self.positionbook_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        columns = ('Time', 'Decision', 'Total Score', 'Price', 'Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit')
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
        analysis_frame = ttk.LabelFrame(main_frame, text="Order Book Analysis", padding=10)
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0,5))
        info_frame = ttk.LabelFrame(main_frame, text="Raw Order Book Data", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(5,0))
        self.order_book_fig = Figure(figsize=(6, 4), dpi=100)
        self.order_book_canvas = FigureCanvasTkAgg(self.order_book_fig, master=chart_frame)
        self.order_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.order_book_analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD,
                                                                  background='#374B61', foreground='white')
        self.order_book_analysis_text.pack(fill=tk.BOTH, expand=True)
        self.order_book_analysis_text.configure(state='disabled')
        self.order_book_raw_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD,
                                                             background='#374B61', foreground='white',
                                                             font=('Courier New', 9))
        self.order_book_raw_text.pack(fill=tk.BOTH, expand=True)
        self.order_book_raw_text.configure(state='disabled')

    def create_position_book_tab(self):
       control_frame = ttk.Frame(self.position_book_tab)
       control_frame.pack(fill=tk.X, pady=10)
       ttk.Label(control_frame, text="Select Instrument:").pack(side=tk.LEFT, padx=5)
       positionbook_instrument_combo = ttk.Combobox(control_frame, textvariable=self.selected_instrument,
                                                     values=self.trading_strategy.instruments, state="readonly")
       positionbook_instrument_combo.pack(side=tk.LEFT, padx=5)
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
       analysis_frame = ttk.LabelFrame(main_frame, text="Position Book Analysis", padding=10)
       analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0,5))
       info_frame = ttk.LabelFrame(main_frame, text="Raw Position Book Data", padding=10)
       info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(5,0))
       self.position_book_fig = Figure(figsize=(6, 4), dpi=100)
       self.position_book_canvas = FigureCanvasTkAgg(self.position_book_fig, master=chart_frame)
       self.position_book_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
       self.position_book_analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD,
                                                                     background='#374B61', foreground='white')
       self.position_book_analysis_text.pack(fill=tk.BOTH, expand=True)
       self.position_book_analysis_text.configure(state='disabled')
       self.position_book_raw_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD,
                                                               background='#374B61', foreground='white',
                                                               font=('Courier New', 9))
       self.position_book_raw_text.pack(fill=tk.BOTH, expand=True)
       self.position_book_raw_text.configure(state='disabled')

    def create_retail_positions_tab(self):
        # Revised layout using a 2x2 grid (Issue 1)
        main_frame = ttk.Frame(self.retail_positions_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        refresh_button = ttk.Button(control_frame, text="Refresh Sentiment Data", command=self.refresh_retail_positions)
        refresh_button.pack(side=tk.LEFT, padx=10)
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)
        self.pie_chart_frame = ttk.LabelFrame(grid_frame, text="Retail Positions (Pie Chart)", padding=10)
        self.pie_chart_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.retail_pie_fig = Figure(figsize=(4, 3), dpi=100)
        self.retail_pie_canvas = FigureCanvasTkAgg(self.retail_pie_fig, master=self.pie_chart_frame)
        self.retail_pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.bar_chart_frame = ttk.LabelFrame(grid_frame, text="Retail Positions (Bar Chart)", padding=10)
        self.bar_chart_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.retail_bar_fig = Figure(figsize=(4, 3), dpi=100)
        self.retail_bar_canvas = FigureCanvasTkAgg(self.retail_bar_fig, master=self.bar_chart_frame)
        self.retail_bar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.twitter_frame = ttk.LabelFrame(grid_frame, text="Twitter Sentiment", padding=10)
        self.twitter_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.twitter_fig = Figure(figsize=(4, 3), dpi=100)
        self.twitter_canvas = FigureCanvasTkAgg(self.twitter_fig, master=self.twitter_frame)
        self.twitter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.table_frame = ttk.LabelFrame(grid_frame, text="Retail Positions (Table)", padding=10)
        self.table_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.retail_table_tree = ttk.Treeview(self.table_frame, show='headings')
        self.retail_table_tree.pack(fill=tk.BOTH, expand=True)

    def create_community_outlook_tab(self):
        # Create a PanedWindow to split the tab into two sections: tables on the left and a chart on the right.
        self.community_outlook_paned = ttk.PanedWindow(self.community_outlook_tab, orient=tk.HORIZONTAL)
        self.community_outlook_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Frame: MyFxBook Sentiment Tables
        left_frame = ttk.Frame(self.community_outlook_paned)

        # Control header with refresh button and title label
        left_control_frame = ttk.Frame(left_frame)
        left_control_frame.pack(fill=tk.X, pady=5)
        refresh_button = ttk.Button(left_control_frame, text="Refresh Community Outlook", command=self.refresh_community_outlook)
        refresh_button.pack(side=tk.LEFT, padx=5)
        header_label = ttk.Label(left_control_frame, text="MyFxBook Sentiment Data", style='Header.TLabel')
        header_label.pack(side=tk.LEFT, padx=10)

        # Sentiment Table
        myfxbook_frame = ttk.LabelFrame(left_frame, text="Sentiment Table", padding=10)
        myfxbook_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        columns = ('Pair', 'Long %', 'Short %', 'Long Lots', 'Short Lots')
        self.myfxbook_tree = ttk.Treeview(myfxbook_frame, columns=columns, show='headings')
        for col in columns:
            self.myfxbook_tree.heading(col, text=col)
            self.myfxbook_tree.column(col, width=100, anchor=tk.CENTER)
        scrollbar_y = ttk.Scrollbar(myfxbook_frame, orient=tk.VERTICAL, command=self.myfxbook_tree.yview)
        scrollbar_x = ttk.Scrollbar(myfxbook_frame, orient=tk.HORIZONTAL, command=self.myfxbook_tree.xview)
        self.myfxbook_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.myfxbook_tree.pack(fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Additional Data Table
        additional_frame = ttk.LabelFrame(left_frame, text="Additional Data", padding=10)
        additional_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.myfxbook_additional_tree = ttk.Treeview(additional_frame, show='headings')
        scrollbar_y2 = ttk.Scrollbar(additional_frame, orient=tk.VERTICAL, command=self.myfxbook_additional_tree.yview)
        scrollbar_x2 = ttk.Scrollbar(additional_frame, orient=tk.HORIZONTAL, command=self.myfxbook_additional_tree.xview)
        self.myfxbook_additional_tree.configure(yscrollcommand=scrollbar_y2.set, xscrollcommand=scrollbar_x2.set)
        self.myfxbook_additional_tree.pack(fill=tk.BOTH, expand=True)
        scrollbar_y2.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x2.pack(side=tk.BOTTOM, fill=tk.X)

        self.community_outlook_paned.add(left_frame, weight=3)

        # Right Frame: Sentiment Overview Chart
        right_frame = ttk.Frame(self.community_outlook_paned)
        chart_header = ttk.Label(right_frame, text="Sentiment Overview Chart", style='Header.TLabel')
        chart_header.pack(pady=5)
        self.myfxbook_chart_frame = ttk.LabelFrame(right_frame, text="Chart", padding=10)
        self.myfxbook_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.myfxbook_chart_fig = Figure(figsize=(5, 4), dpi=100)
        self.myfxbook_chart_canvas = FigureCanvasTkAgg(self.myfxbook_chart_fig, master=self.myfxbook_chart_frame)
        self.myfxbook_chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.community_outlook_paned.add(right_frame, weight=2)

    def create_footer_frame(self):
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, padx=10, pady=10)
        refresh_button = ttk.Button(footer_frame, text="Refresh Now", command=self.refresh_data)
        refresh_button.pack(side=tk.LEFT, padx=5)
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

    def start_trading_thread(self):
        self.running = True
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

    def trading_loop(self):
      while self.running:
        try:
            self.root.after(0, self.refresh_data)
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            logging.debug(traceback.format_exc())
            self.root.after(0, lambda err=str(e): self.update_status(f"Error: {err}"))
        try:
            interval = int(self.interval_var.get())
            time.sleep(interval)
        except ValueError:
            time.sleep(60)
        except Exception as e:
          logging.error(f"Error in trading loop sleep: {e}")
          time.sleep(60)

    def refresh_data(self):
        try:
            self.update_status("Analyzing trading data...")
            all_decisions = self.trading_strategy.decide_trade()
            for instrument, data in all_decisions.items():
                if "signal" in data and data["signal"]:
                   self.history.add_signal(data["signal"])
            self.root.after(0, lambda: self.update_decisions_table(all_decisions))
            self.root.after(0, lambda: self.update_details_text(all_decisions))
            self.root.after(0, lambda: self.update_chart(all_decisions))
            selected = self.selected_instrument.get()
            if selected:
                self.root.after(0, self.refresh_detailed_analysis)
                self.root.after(0, self.refresh_signal_history)
                self.root.after(0, self.refresh_order_book)
                self.root.after(0, self.refresh_position_book)
            if random.random() < 0.2:
              self.root.after(0, self.refresh_retail_positions)
            if random.random() < 0.2:
                self.root.after(0, self.refresh_community_outlook)
            if random.random() < 0.1:
                self.root.after(0, self.update_account_info)
            self.root.after(0, self.update_last_update_time)
            self.root.after(0, lambda: self.update_status("Trading analysis completed"))
        except Exception as e:
            logging.error(f"Error refreshing data: {e}")
            logging.debug(traceback.format_exc())
            self.root.after(0, lambda err=str(e): self.update_status(f"Error: {err}"))

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
                f"{details['pair_sentiment_score']:.1f}",
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
        self.details_text.insert(tk.END, f"• Pair-Specific Sentiment: {self.trading_strategy.weights['pair_sentiment']*100:.0f}%\n")
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
                self.details_text.insert(tk.END, f"PS: {details['pair_sentiment_score']} | ")
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
        tradable_instruments = [k for k, v in decisions.items()
                              if v["decision"] in ["Strong Bullish", "Bullish", "Strong Bearish", "Bearish"]]
        if tradable_instruments:
          detail_instruments = tradable_instruments[:5]
          components = ['Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit']
          x = np.arange(len(components))
          width = 0.15
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
        item_id = selected_items[0]
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
      ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
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
      components = ['Order Book', 'Position Book', 'Pair Sentiment', 'Currency Sentiment', 'Retail Profit']
      weights = [
          self.trading_strategy.weights['order_book'],
          self.trading_strategy.weights['position_book'],
          self.trading_strategy.weights['pair_sentiment'],
          self.trading_strategy.weights['currency_sentiment'],
          self.trading_strategy.weights['retail_profitability']
      ]
      raw_scores = [
          signal.order_book_score,
          signal.position_book_score,
          signal.pair_sentiment_score,
          signal.currency_sentiment_score,
          signal.retail_profit_score
      ]
      weighted_scores = [raw * weight for raw, weight in zip(raw_scores, weights)]
      x = np.arange(len(components))
      width = 0.35
      bars1 = ax.bar(x - width/2, raw_scores, width, label='Raw Score', color='skyblue')
      bars2 = ax.bar(x + width/2, weighted_scores, width, label='Weighted Score', color='navy')
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
        self.orderbook_fig.clear()
        ax = self.orderbook_fig.add_subplot(111)
        orderbook_data = self.oanda_fetcher.fetch_orderbook(instrument)
        if not orderbook_data or 'buckets' not in orderbook_data:
            ax.text(0.5, 0.5, f'No order book data available for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.orderbook_canvas.draw()
            return
        buckets = orderbook_data['buckets']
        current_price = float(orderbook_data.get('price', 0))
        bucket_width = float(orderbook_data.get('bucketWidth', 0))
        time_str = orderbook_data.get('time', 'N/A')
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        if current_price > 0:
          filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
              prices, short_counts, long_counts, current_price
          )
        else:
          filtered_prices, filtered_shorts, filtered_longs = [], [], []
        if not filtered_prices:
            ax.text(0.5, 0.5, f'No order book data within price range for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.orderbook_canvas.draw()
            return
        if len(filtered_prices) > 1:
          price_range = max(filtered_prices) - min(filtered_prices)
          bar_width = price_range / len(filtered_prices) * 0.8
        else:
          bar_width = bucket_width
        ax.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7, label='Long %', align='edge')
        ax.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width, color='red', alpha=0.7, label='Short %', align='edge')
        diff = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        ax.plot(filtered_prices, diff, color='purple', alpha=0.7, label='Long-Short Diff', linewidth=2)
        if current_price > 0:
          ax.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
        self.initial_xlim_ob = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                               max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        ax.set_xlim(self.initial_xlim_ob)
        min_y = min(0, -max(filtered_shorts)) if filtered_shorts else -1
        max_y = max(filtered_longs) if filtered_longs else 1
        ax.set_ylim(min_y, max_y)
        ax.set_title(f'Order Book for {instrument} at {time_str}')
        ax.set_xlabel('Price')
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')
        self.orderbook_fig.tight_layout()
        # Bind zoom buttons to self.ax1_ob for consistent zooming (Issue 4)
        self.zoom_in_button_ob.config(command=lambda: self.zoom_plot(self.ax1_ob, 0.8))
        self.zoom_out_button_ob.config(command=lambda: self.zoom_plot(self.ax1_ob, 1.25))
        self.reset_zoom_button_ob.config(command=lambda: self.reset_zoom(self.ax1_ob, "ob"))
        self.orderbook_canvas.draw()

    def update_detailed_positionbook_chart(self, instrument: str):
        self.positionbook_fig.clear()
        if not hasattr(self, 'ax1_pb'):
            self.ax1_pb = self.positionbook_fig.add_subplot(2, 1, 1)
            self.ax2_pb = self.positionbook_fig.add_subplot(2, 1, 2, sharex=self.ax1_pb)

        positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)
        if not positionbook_data or 'buckets' not in positionbook_data:
            self.ax1_pb.clear()
            self.ax2_pb.clear()
            self.ax1_pb.text(0.5, 0.5, f'No position book data available for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax1_pb.transAxes, fontsize=12)
            self.ax1_pb.set_xticks([])
            self.ax1_pb.set_yticks([])
            self.ax2_pb.set_xticks([])
            self.ax2_pb.set_yticks([])
            self.position_book_canvas.draw()
            return

        # Create two subplots (upper: bars, lower: cumulative)
        if not self.positionbook_fig.axes:
            gs = self.positionbook_fig.add_gridspec(2, 1, height_ratios=[2, 1])
            self.ax1_pb = self.positionbook_fig.add_subplot(gs[0])
            self.ax2_pb = self.positionbook_fig.add_subplot(gs[1], sharex=self.ax1_pb)
        else:
            self.ax1_pb.clear()
            self.ax2_pb.clear()

        buckets = positionbook_data['buckets']
        current_price = float(positionbook_data.get('price', 0))
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        time_str = positionbook_data.get('time', 'N/A')
        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)

        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_percents, long_percents, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []

        if not filtered_prices:
            self.ax1_pb.text(0.5, 0.5, f'No position book data within price range for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_pb.transAxes, fontsize=12)
            self.ax1_pb.set_xticks([])
            self.ax1_pb.set_yticks([])
            self.ax2_pb.set_xticks([])
            self.ax2_pb.set_yticks([])
            self.position_book_canvas.draw()
            return

        # Determine a suitable bar width
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8
        else:
            bar_width = bucket_width

        self.ax1_pb.bar(filtered_prices, filtered_longs, width=bar_width,
                        color='green', alpha=0.7, label='Long %', align='edge')
        self.ax1_pb.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width,
                        color='red', alpha=0.7, label='Short %', align='edge')
        diff = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        self.ax1_pb.plot(filtered_prices, diff, color='purple', alpha=0.7,
                         label='Long-Short Diff', linewidth=2)
        if current_price > 0:
            self.ax1_pb.axvline(x=current_price, color='blue', linestyle='-',
                                linewidth=2, label='Current Price')

        # Set consistent x-axis limits
        self.initial_xlim_pb = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                                max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        self.ax1_pb.set_xlim(self.initial_xlim_pb)
        self.ax2_pb.set_xlim(self.initial_xlim_pb)
        self.ax1_pb.set_ylabel('Percentage')
        self.ax1_pb.set_title(f'Position Book for {instrument}')
        self.ax1_pb.legend()
        self.ax1_pb.grid(True, alpha=0.3)

        # --- NEW: Plot cumulative values on the lower subplot ---
        import numpy as np
        cumulative_long = np.cumsum(filtered_longs)
        cumulative_short = np.cumsum(filtered_shorts)
        self.ax2_pb.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-',
                         marker='o', label='Cumulative Long')
        self.ax2_pb.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-',
                         marker='o', label='Cumulative Short')
        self.ax2_pb.set_ylabel('Cumulative %')
        self.ax2_pb.legend()
        self.ax2_pb.grid(True, alpha=0.3)

        self.position_book_fig.tight_layout()
        # Bind zoom buttons for the position book if needed
        self.zoom_in_button_pb.config(command=lambda: self.zoom_plot(self.ax1_pb, 0.8))
        self.zoom_out_button_pb.config(command=lambda: self.zoom_plot(self.ax1_pb, 1.25))
        self.reset_zoom_button_pb.config(command=lambda: self.reset_zoom(self.ax1_pb, "pb"))
        self.position_book_canvas.draw()

    def update_position_book_visualization(self, instrument: str, positionbook_data: Optional[Dict[str, Any]]):
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

        # Create two subplots (upper: bars, lower: cumulative)
        gs = self.position_book_fig.add_gridspec(2, 1, height_ratios=[2, 1])
        self.ax1_pb = self.position_book_fig.add_subplot(gs[0])
        self.ax2_pb = self.position_book_fig.add_subplot(gs[1], sharex=self.ax1_pb)

        buckets = positionbook_data['buckets']
        current_price = float(positionbook_data.get('price', 0))
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        prices, short_percents, long_percents = self.oanda_fetcher.parse_buckets(buckets)

        if current_price > 0:
            filtered_prices, filtered_shorts, filtered_longs = self.oanda_fetcher.filter_buckets(
                prices, short_percents, long_percents, current_price
            )
        else:
            filtered_prices, filtered_shorts, filtered_longs = [], [], []

        if not filtered_prices:
            self.ax1_pb.text(0.5, 0.5, f'No position book data within price range for {instrument}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax1_pb.transAxes, fontsize=12)
            self.ax1_pb.set_xticks([])
            self.ax1_pb.set_yticks([])
            self.ax2_pb.set_xticks([])
            self.ax2_pb.set_yticks([])
            self.position_book_canvas.draw()
            return

        # Determine a suitable bar width
        if len(filtered_prices) > 1:
            price_range = max(filtered_prices) - min(filtered_prices)
            bar_width = price_range / len(filtered_prices) * 0.8
        else:
            bar_width = bucket_width

        self.ax1_pb.bar(filtered_prices, filtered_longs, width=bar_width,
                        color='green', alpha=0.7, label='Long %', align='edge')
        self.ax1_pb.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width,
                        color='red', alpha=0.7, label='Short %', align='edge')
        diff = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
        self.ax1_pb.plot(filtered_prices, diff, color='purple', alpha=0.7,
                         label='Long-Short Diff', linewidth=2)
        if current_price > 0:
            self.ax1_pb.axvline(x=current_price, color='blue', linestyle='-',
                                linewidth=2, label='Current Price')

        # Set consistent x-axis limits
        self.initial_xlim_pb = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                                max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
        self.ax1_pb.set_xlim(self.initial_xlim_pb)
        self.ax2_pb.set_xlim(self.initial_xlim_pb)
        self.ax1_pb.set_ylabel('Percentage')
        self.ax1_pb.set_title(f'Position Book for {instrument}')
        self.ax1_pb.legend()
        self.ax1_pb.grid(True, alpha=0.3)

        # --- NEW: Plot cumulative values on the lower subplot ---
        import numpy as np
        cumulative_long = np.cumsum(filtered_longs)
        cumulative_short = np.cumsum(filtered_shorts)
        self.ax2_pb.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-',
                         marker='o', label='Cumulative Long')
        self.ax2_pb.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-',
                         marker='o', label='Cumulative Short')
        self.ax2_pb.set_ylabel('Cumulative %')
        self.ax2_pb.legend()
        self.ax2_pb.grid(True, alpha=0.3)

        self.position_book_fig.tight_layout()
        # Bind zoom buttons for the position book if needed
        self.zoom_in_button_pb.config(command=lambda: self.zoom_plot(self.ax1_pb, 0.8))
        self.zoom_out_button_pb.config(command=lambda: self.zoom_plot(self.ax1_pb, 1.25))
        self.reset_zoom_button_pb.config(command=lambda: self.reset_zoom(self.ax1_pb, "pb"))
        self.position_book_canvas.draw()

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
      pair_sentiment_scores = [s.pair_sentiment_score for s in signals]
      currency_sentiment_scores = [s.currency_sentiment_score for s in signals]
      retail_profit_scores = [s.retail_profit_score for s in signals]
      prices = [s.price for s in signals]
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
      ax1.axhline(y=self.trading_strategy.thresholds['strong_bullish'], color='#008000',
                  linestyle='--', alpha=0.7, label='Strong Bullish')
      ax1.axhline(y=self.trading_strategy.thresholds['bullish'], color='#4CAF50',
                  linestyle='--', alpha=0.7, label='Bullish')
      ax1.axhline(y=self.trading_strategy.thresholds['bearish'], color='#F44336',
                  linestyle='--', alpha=0.7, label='Bearish')
      ax1.axhline(y=self.trading_strategy.thresholds['strong_bearish'], color='#8B0000',
                  linestyle='--', alpha=0.7, label='Strong Bearish')
      ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
      ax1.set_title(f'Signal History for {instrument}')
      ax1.set_ylabel('Signal Score')
      ax1.set_ylim(-2.5, 2.5)
      ax1.grid(True, alpha=0.3)
      ax1.legend(loc='upper left', fontsize='small')
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
            f"{signal.pair_sentiment_score}",
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
            self.update_raw_order_book_data(orderbook_data)
        except Exception as e:
            logging.error(f"Error refreshing order book: {e}")
            logging.debug(traceback.format_exc())

    def update_order_book_visualization(self, instrument: str, orderbook_data: Optional[Dict[str, Any]]):
      self.order_book_fig.clear()
      if not hasattr(self, 'ax1_ob'):
          self.ax1_ob = self.order_book_fig.add_subplot(2, 1, 1)
          self.ax2_ob = self.order_book_fig.add_subplot(2, 1, 2, sharex=self.ax1_ob)

      if not orderbook_data or 'buckets' not in orderbook_data:
          self.ax1_ob.clear()
          self.ax2_ob.clear()
          self.ax1_ob.text(0.5, 0.5, f'No order book data available for {instrument}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax1_ob.transAxes, fontsize=12)
          self.ax1_ob.set_xticks([])
          self.ax1_ob.set_yticks([])
          self.ax2_ob.set_xticks([])
          self.ax2_ob.set_yticks([])
          self.order_book_canvas.draw()
          return
      # Create two subplots (upper: bars, lower: cumulative)
      if not self.order_book_fig.axes:
          gs = self.order_book_fig.add_gridspec(2, 1, height_ratios=[2, 1])
          self.ax1_ob = self.order_book_fig.add_subplot(gs[0])
          self.ax2_ob = self.order_book_fig.add_subplot(gs[1], sharex=self.ax1_ob)
      else:
          self.ax1_ob.clear()
          self.ax2_ob.clear()

      buckets = orderbook_data['buckets']
      current_price = float(orderbook_data.get('price', 0))
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
          self.ax1_ob.set_xticks([])
          self.ax1_ob.set_yticks([])
          self.ax2_ob.set_xticks([])
          self.ax2_ob.set_yticks([])
          self.order_book_canvas.draw()
          return
      if len(filtered_prices) > 1:
          price_range = max(filtered_prices) - min(filtered_prices)
          bar_width = price_range / len(filtered_prices) * 0.8
      else:
          bar_width = bucket_width
      self.ax1_ob.bar(filtered_prices, filtered_longs, width=bar_width, color='green', alpha=0.7, label='Long %', align='edge')
      self.ax1_ob.bar(filtered_prices, [-s for s in filtered_shorts], width=-bar_width, color='red', alpha=0.7,
                label='Short %', align='edge')
      diff = [l - s for l, s in zip(filtered_longs, filtered_shorts)]
      self.ax1_ob.plot(filtered_prices, diff, color='purple', alpha=0.7, label='Long-Short Diff', linewidth=2)
      if current_price > 0:
          self.ax1_ob.axvline(x=current_price, color='blue', linestyle='-', linewidth=2, label='Current Price')
      self.initial_xlim_ob = (min(filtered_prices) - (max(filtered_prices) - min(filtered_prices)) * 0.1,
                               max(filtered_prices) + (max(filtered_prices) - min(filtered_prices)) * 0.1)
      self.ax1_ob.set_xlim(self.initial_xlim_ob)
      self.ax2_ob.set_xlim(self.initial_xlim_ob)
      min_y = min(0, -max(filtered_shorts)) if filtered_shorts else -1
      max_y = max(filtered_longs) if filtered_longs else 1
      self.ax1_ob.set_ylim(min_y, max_y)
      self.ax1_ob.set_title(f'Order Book for {instrument} at {time_str}')
      self.ax1_ob.set_xlabel('Price')
      self.ax1_ob.set_ylabel('Percentage')
      self.ax1_ob.legend()
      self.ax1_ob.grid(True, alpha=0.3)

      # --- NEW: Plot cumulative values on the lower subplot ---
      import numpy as np
      cumulative_long = np.cumsum(filtered_longs)
      cumulative_short = np.cumsum(filtered_shorts)
      self.ax2_ob.plot(filtered_prices, cumulative_long, color='darkgreen', linestyle='-',
                       marker='o', label='Cumulative Long')
      self.ax2_ob.plot(filtered_prices, cumulative_short, color='darkred', linestyle='-',
                       marker='o', label='Cumulative Short')
      self.ax2_ob.set_ylabel('Cumulative %')
      self.ax2_ob.legend()
      self.ax2_ob.grid(True, alpha=0.3)

      self.order_book_fig.tight_layout()
      self.order_book_canvas.draw()

    def update_order_book_analysis(self, instrument: str, orderbook_data: Optional[Dict[str, Any]]):
        self.order_book_analysis_text.configure(state='normal')
        self.order_book_analysis_text.delete(1.0, tk.END)
        if not orderbook_data:
            self.order_book_analysis_text.insert(tk.END, f"No order book data available for {instrument}.")
            self.order_book_analysis_text.configure(state='disabled')
            return
        buckets = orderbook_data['buckets']
        current_price = float(orderbook_data.get('price', 0))
        bucket_width = float(orderbook_data.get('bucketWidth', 0))
        time_str = orderbook_data.get('time', 'N/A')
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        price_diff_50_pips = self.trading_strategy.pips_to_price_diff(instrument, 50)
        lower_bound_50pips = current_price - price_diff_50_pips
        upper_bound_50pips = current_price + price_diff_50_pips
        near_price_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                              if lower_bound_50pips <= p <= upper_bound_50pips]
        near_price_long_pct = 0
        near_price_short_pct = 0
        if near_price_buckets:
            _, near_shorts, near_longs = zip(*near_price_buckets)
            near_price_long_pct = sum(near_longs)
            near_price_short_pct = sum(near_shorts)
        support_levels = []
        resistance_levels = []
        for i, (price, short_pct, long_pct) in enumerate(zip(prices, short_counts, long_counts)):
            if price < current_price and long_pct > 5:
                support_levels.append((price, long_pct))
            if price > current_price and short_pct > 5:
                resistance_levels.append((price, short_pct))
        support_levels = sorted(support_levels, key=lambda x: x[1], reverse=True)[:3]
        resistance_levels = sorted(resistance_levels, key=lambda x: x[1], reverse=True)[:3]
        self.order_book_analysis_text.insert(tk.END, f"ORDER BOOK ANALYSIS FOR {instrument}\n", "header")
        self.order_book_analysis_text.insert(tk.END, f"Data as of: {time_str}\n\n")
        self.order_book_analysis_text.insert(tk.END, "CURRENT PRICE ANALYSIS:\n", "subheader")
        self.order_book_analysis_text.insert(tk.END, f"Current Price: {current_price:.5f}\n")
        if total_short := sum(short_counts):
            long_short_ratio = sum(long_counts) / total_short
            self.order_book_analysis_text.insert(tk.END, f"Overall Long/Short Ratio: {long_short_ratio:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "Overall Long/Short Ratio: ∞ (no short orders)\n")
        self.order_book_analysis_text.insert(tk.END, f"\nORDERS WITHIN 50 PIPS OF CURRENT PRICE:\n", "subheader")
        self.order_book_analysis_text.insert(tk.END, f"Long Orders: {near_price_long_pct:.2f}%\n")
        self.order_book_analysis_text.insert(tk.END, f"Short Orders: {near_price_short_pct:.2f}%\n")
        if near_price_short_pct > 0:
            near_long_short_ratio = near_price_long_pct / near_price_short_pct
            self.order_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: {near_long_short_ratio:.2f}\n")
        else:
            self.order_book_analysis_text.insert(tk.END, "Near Price Long/Short Ratio: ∞ (no short orders near price)\n")
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
        self.order_book_analysis_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.order_book_analysis_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.order_book_analysis_text.tag_configure("bullish", foreground="#4CAF50")
        self.order_book_analysis_text.tag_configure("bearish", foreground="#F44336")
        self.order_book_analysis_text.tag_configure("neutral", foreground="#607D8B")
        self.order_book_analysis_text.configure(state='disabled')

    def update_raw_order_book_data(self, orderbook_data: Optional[Dict[str, Any]]):
        self.order_book_raw_text.configure(state='normal')
        self.order_book_raw_text.delete(1.0, tk.END)
        if not orderbook_data:
            self.order_book_raw_text.insert(tk.END, "No order book data available.")
            self.order_book_raw_text.configure(state='disabled')
            return
        formatted_json = json.dumps(orderbook_data, indent=2)
        self.order_book_raw_text.insert(tk.END, formatted_json)
        self.order_book_raw_text.configure(state='disabled')

    def refresh_position_book(self):
        instrument = self.selected_instrument.get()
        if not instrument:
            return
        try:
            positionbook_data = self.oanda_fetcher.fetch_positionbook(instrument)
            self.update_position_book_visualization(instrument, positionbook_data)
            self.update_position_book_analysis(instrument, positionbook_data)
            self.update_raw_position_book_data(positionbook_data)
        except Exception as e:
            logging.error(f"Error refreshing position book: {e}")
            logging.debug(traceback.format_exc())

    def update_position_book_analysis(self, instrument: str, positionbook_data: Optional[Dict[str, Any]]):
        self.position_book_analysis_text.configure(state='normal')
        self.position_book_analysis_text.delete(1.0, tk.END)
        if not positionbook_data or 'buckets' not in positionbook_data:
            self.position_book_analysis_text.insert(tk.END, f"No position book data available for {instrument}.")
            self.position_book_analysis_text.configure(state='disabled')
            return
        buckets = positionbook_data['buckets']
        current_price = float(positionbook_data.get('price', 0))
        bucket_width = float(positionbook_data.get('bucketWidth', 0))
        time_str = positionbook_data.get('time', 'N/A')
        prices, short_counts, long_counts = self.oanda_fetcher.parse_buckets(buckets)
        price_diff_50_pips = self.trading_strategy.pips_to_price_diff(instrument, 50)
        lower_bound_50pips = current_price - price_diff_50_pips
        upper_bound_50pips = current_price + price_diff_50_pips
        near_price_buckets = [(p, s, l) for p, s, l in zip(prices, short_counts, long_counts)
                              if lower_bound_50pips <= p <= upper_bound_50pips]
        near_price_long_pct = 0
        near_price_short_pct = 0
        if near_price_buckets:
            _, near_shorts, near_longs = zip(*near_price_buckets)
            near_price_long_pct = sum(near_longs)
            near_price_short_pct = sum(near_shorts)
        long_concentration = []
        short_concentration = []
        for price, short_pct, long_pct in zip(prices, short_counts, long_counts):
            if long_pct > 5:
                long_concentration.append((price, long_pct))
            if short_pct > 5:
                short_concentration.append((price, short_pct))
        long_concentration = sorted(long_concentration, key=lambda x: x[1], reverse=True)[:3]
        short_concentration = sorted(short_concentration, key=lambda x: x[1], reverse=True)[:3]
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
        self.position_book_analysis_text.insert(tk.END, f"\nPOSITIONS WITHIN 50 PIPS OF CURRENT PRICE:\n", "subheader")
        self.position_book_analysis_text.insert(tk.END, f"Long Positions: {near_price_long_pct:.2f}%\n")
        self.position_book_analysis_text.insert(tk.END, f"Short Positions: {near_price_short_pct:.2f}%\n")
        if near_price_short_pct > 0:
          near_long_short_ratio = near_price_long_pct / near_price_short_pct
          self.position_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: {near_long_short_ratio:.2f}\n")
        else:
          self.position_book_analysis_text.insert(tk.END, f"Near Price Long/Short Ratio: ∞ (no short positions near price)\n")
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
        self.position_book_analysis_text.tag_configure("header", font=("Segoe UI", 12, "bold"))
        self.position_book_analysis_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.position_book_analysis_text.tag_configure("bullish", foreground="#4CAF50")
        self.position_book_analysis_text.tag_configure("bearish", foreground="#F44336")
        self.position_book_analysis_text.tag_configure("neutral", foreground="#607D8B")
        self.position_book_analysis_text.configure(state='disabled')

    def update_raw_position_book_data(self, positionbook_data: Optional[Dict[str, Any]]):
      self.position_book_raw_text.configure(state='normal')
      self.position_book_raw_text.delete(1.0, tk.END)
      if not positionbook_data:
        self.position_book_raw_text.insert(tk.END, "No position book data available.")
        self.position_book_raw_text.configure(state='disabled')
        return
      formatted_json = json.dumps(positionbook_data, indent=2)
      self.position_book_raw_text.insert(tk.END, formatted_json)
      self.position_book_raw_text.configure(state='disabled')

    def refresh_retail_positions(self):
      try:
        self.update_retail_positions_chart()
        self.update_twitter_sentiment_chart()
      except Exception as e:
          logging.error(f"Error refreshing retail positions analysis: {e}")
          logging.debug(traceback.format_exc())

    def update_retail_positions_chart(self):
        self.retail_pie_fig.clear()
        self.retail_bar_fig.clear()
        html_content = self.sentiment_fetcher.fetch_retail_positions_data()
        pie_data = self.sentiment_fetcher.extract_pie_chart_data(html_content)
        table_data = self.sentiment_fetcher.extract_retail_table_data(html_content)
        ax1 = self.retail_pie_fig.add_subplot(111)
        if not pie_data:
            ax1.text(0.5, 0.5, 'No retail positions data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
        else:
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
                except (ValueError, TypeError) as e:
                    logging.warning(f"Skipping currency {currency} due to parsing error: {e}")
                    continue
            if not labels:
                ax1.text(0.5, 0.5, 'Could not parse retail positions data',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes, fontsize=12)
                ax1.set_xticks([])
                ax1.set_yticks([])
            else:
                avg_long = sum(long_values) / len(long_values) if long_values else 0
                avg_short = 100 - avg_long
                pie_sizes = [avg_long, avg_short]
                pie_labels = ['Long', 'Short']
                pie_colors = ['green', 'red']
                pie_explode = (0.1, 0)
                ax1.pie(pie_sizes, explode=pie_explode, labels=pie_labels, colors=pie_colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
                ax1.set_title('Average Retail Position Sentiment')
                ax1.axis('equal')
        self.retail_pie_canvas.draw()
        ax2 = self.retail_bar_fig.add_subplot(111)
        if pie_data:
          if labels:
              x = np.arange(len(labels))
              width = 0.35
              ax2.bar(x - width/2, long_values, width, label='Long %', color='green')
              ax2.bar(x + width/2, short_values, width, label='Short %', color='red')
              ax2.set_title('Retail Positions by Currency')
              ax2.set_ylabel('Percentage')
              ax2.set_xticks(x)
              ax2.set_xticklabels(labels)
              ax2.legend()
              ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
              plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
          else:
              ax2.text(0.5, 0.5, 'Could not parse retail positions data',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax2.transAxes, fontsize=12)
              ax2.set_xticks([])
              ax2.set_yticks([])
        else:
            ax2.text(0.5, 0.5, 'No retail positions data available',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax2.transAxes, fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
        self.retail_bar_canvas.draw()
        if table_data and 'overview' in table_data:
            for item in self.retail_table_tree.get_children():
                self.retail_table_tree.delete(item)
            if not self.retail_table_tree['columns']:
                all_metrics = set()
                for currency_data in table_data['overview'].values():
                    all_metrics.update(currency_data.keys())
                columns = ['Currency'] + sorted(list(all_metrics))
                self.retail_table_tree['columns'] = columns
                for col in columns:
                  self.retail_table_tree.heading(col, text=col)
                  self.retail_table_tree.column(col, anchor=tk.CENTER)
            for currency, metrics in table_data['overview'].items():
              row_values = [currency]
              for metric in self.retail_table_tree['columns'][1:]:
                row_values.append(metrics.get(metric, 'N/A'))
              self.retail_table_tree.insert('', tk.END, values=row_values)
        else:
            for item in self.retail_table_tree.get_children():
                self.retail_table_tree.delete(item)

    def update_twitter_sentiment_chart(self):
        self.twitter_fig.clear()
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
        ax = self.twitter_fig.add_subplot(111)
        pairs = list(twitter_chart.keys())
        colors = list(twitter_chart.values())
        positive_colors = ["#91DB57", "#57DB80", "#57D3DB", "#5770DB", "#A157DB"]
        negative_colors = ["#DB5F57", "#DBC257"]
        neutral_colors = ["#C0C0C0", "#A0A0A0"]
        sentiment_scores = []
        bar_colors = []
        for color in colors:
            if color in positive_colors:
                idx = positive_colors.index(color)
                score = 0.5 + (idx + 1) * (0.5 / len(positive_colors))
                sentiment_scores.append(score)
                bar_colors.append('green')
            elif color in negative_colors:
                idx = negative_colors.index(color)
                score = -1.0 + idx * (0.5 / len(negative_colors))
                sentiment_scores.append(score)
                bar_colors.append('red')
            else:
                sentiment_scores.append(0)
                bar_colors.append('gray')
        sorted_data = sorted(zip(pairs, sentiment_scores, bar_colors), key=lambda x: x[1])
        if sorted_data:
            sorted_pairs, sorted_scores, sorted_colors = zip(*sorted_data)
        else:
            sorted_pairs, sorted_scores, sorted_colors = ([], [], [])
        if not sorted_pairs:
            ax.text(0.5, 0.5, 'Could not parse Twitter sentiment data',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.twitter_canvas.draw()
            return
        y_pos = np.arange(len(sorted_pairs))
        bars = ax.barh(y_pos, sorted_scores, align='center', color=sorted_colors)
        for i, score in enumerate(sorted_scores):
          if score >= 0.5:
            ax.text(score + 0.05 , i, "Bullish", va='center', color='green')
          elif score <= -0.5:
            ax.text(score - 0.25, i, "Bearish", va='center', color='red')
          else:
            ax.text(score + 0.05, i, "Neutral", va='center', color='gray')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_pairs)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel('Sentiment Score')
        ax.set_title('Twitter Sentiment Analysis by Currency Pair')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3)
        ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.3)
        ax.grid(True, axis='x', alpha=0.3)
        self.twitter_fig.tight_layout()
        self.twitter_canvas.draw()

    def refresh_community_outlook(self):
      try:
        self.update_myfxbook_sentiment_table()
        self.update_myfxbook_additional_table()
        self.update_myfxbook_sentiment_chart()
      except Exception as e:
        logging.error(f"Error refreshing community outlook: {e}")
        logging.debug(traceback.format_exc())

    def update_myfxbook_sentiment_table(self):
        for item in self.myfxbook_tree.get_children():
            self.myfxbook_tree.delete(item)
        html = self.sentiment_fetcher.fetch_myfxbook_sentiment_data()
        records = self.sentiment_fetcher.extract_myfxbook_table_data(html)
        if not records:
          return
        for record in records:
          if "Symbol" not in record:
            continue
          pair = record["Symbol"]
          try:
            long_pct = record.get("Long %", 0.0)
            short_pct = record.get("Short %", 0.0)
            long_lots = record.get("Long Lots", 0.0)
            short_lots = record.get("Short Lots", 0.0)
            values = (pair, f"{long_pct:.2f}%", f"{short_pct:.2f}%", f"{long_lots:.2f}", f"{short_lots:.2f}")
            row_tag = "neutral"
            if long_pct >= 60:
                row_tag = "long"
            elif long_pct <= 40:
                row_tag = "short"
            self.myfxbook_tree.insert("", tk.END, values=values, tags=(row_tag,))
          except Exception as e:
            logging.error(f"Error processing MyFxBook data for {pair}: {e}")
        self.myfxbook_tree.tag_configure("long", background="#4CAF50")
        self.myfxbook_tree.tag_configure("short", background="#F44336")
        self.myfxbook_tree.tag_configure("neutral", background="#607D8B")

    def update_myfxbook_additional_table(self):
        html_content = self.sentiment_fetcher.fetch_myfxbook_sentiment_data()
        if not html_content:
          return
        df = self.sentiment_fetcher.extract_myfxbook_additional_data(html_content)
        if df is None:
          return
        for item in self.myfxbook_additional_tree.get_children():
            self.myfxbook_additional_tree.delete(item)
        if not self.myfxbook_additional_tree['columns']:
            self.myfxbook_additional_tree['columns'] = list(df.columns)
            for col in df.columns:
                self.myfxbook_additional_tree.heading(col, text=col)
                self.myfxbook_additional_tree.column(col, anchor=tk.CENTER)
        for _, row in df.iterrows():
            self.myfxbook_additional_tree.insert("", tk.END, values=list(row))

    def update_myfxbook_sentiment_chart(self):
        self.myfxbook_chart_fig.clear()
        ax = self.myfxbook_chart_fig.add_subplot(111)
        # Fetch and parse records from MyFxBook sentiment data
        html = self.sentiment_fetcher.fetch_myfxbook_sentiment_data()
        records = self.sentiment_fetcher.extract_myfxbook_table_data(html)
        if not records:
            ax.text(0.5, 0.5, "No chart data available", horizontalalignment='center', verticalalignment='center')
        else:
            pairs = []
            long_percents = []
            short_percents = []
            for record in records:
                if "Symbol" in record:
                    pairs.append(record["Symbol"])
                    try:
                        long_val = float(str(record.get("Long %", "0")).replace('%',''))
                        short_val = float(str(record.get("Short %", "0")).replace('%',''))
                    except Exception:
                        long_val, short_val = 0, 0
                    long_percents.append(long_val)
                    short_percents.append(short_val)
            x = np.arange(len(pairs))
            width = 0.35
            ax.bar(x - width/2, long_percents, width, label="Long %", color="green")
            ax.bar(x + width/2, short_percents, width, label="Short %", color="red")
            ax.set_xticks(x)
            ax.set_xticklabels(pairs, rotation=45, ha='right')
            ax.set_ylabel("Percentage")
            ax.set_title("MyFxBook Sentiment Chart")
            ax.legend()
            ax.grid(True, alpha=0.3)
        self.myfxbook_chart_canvas.draw()

    def zoom_plot(self, ax, factor):
      current_xlim = ax.get_xlim()
      x_center = (current_xlim[0] + current_xlim[1]) / 2
      new_width = (current_xlim[1] - current_xlim[0]) * factor
      new_xlim = (x_center - new_width / 2, x_center + new_width / 2)
      ax.set_xlim(new_xlim)
      if ax == self.ax1_ob:
          self.ax2_ob.set_xlim(new_xlim)
          self.order_book_canvas.draw()
      elif ax == self.ax1_pb:
          self.ax2_pb.set_xlim(new_xlim)
          self.position_book_canvas.draw()

    def reset_zoom(self, ax, book_type):
        if book_type == "ob":
           ax.set_xlim(self.initial_xlim_ob)
           self.ax2_ob.set_xlim(self.initial_xlim_ob)
           self.order_book_canvas.draw()
        elif book_type == "pb":
           ax.set_xlim(self.initial_xlim_pb)
           self.ax2_pb.set_xlim(self.initial_xlim_pb)
           self.position_book_canvas.draw()

# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    root = tk.Tk()
    app = TradingRobotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
