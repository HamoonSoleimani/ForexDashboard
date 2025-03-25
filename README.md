# Forex Trading Robot with Multi-Source Data Analysis and Visualization
![2](https://github.com/user-attachments/assets/38216972-1f9f-4c3b-9eb8-b8f262e5f724)
![1](https://github.com/user-attachments/assets/7fea95dc-7d65-4b96-9727-62b9723ac803)


## Overview

This Python-based trading robot is designed to analyze Forex market data from multiple sources and provide trading signals based on a configurable rule-based strategy. The robot features a robust graphical user interface (GUI) built with Tkinter, providing real-time visualization of trading decisions, order book and position book data, sentiment analysis, and detailed signal history. The core data sources include:

*   **OANDA Order Book and Position Book:**  Leverages the `oandapyV20` library to access real-time order book and position book data directly from OANDA.  Provides insights into market depth and trader positioning.
*   **Retail Sentiment (ForexBenchmark):** Scrapes the ForexBenchmark website to gather retail trader sentiment data, offering a contrarian perspective.
*   **Twitter Sentiment (ForexBenchmark):**  Also scrapes ForexBenchmark to obtain Twitter-based sentiment analysis for various currency pairs.
*   **Myfxbook Sentiment:** Scrapes the Myfxbook community outlook page to gather sentiment data from a large community of Forex traders.

The robot combines these diverse data sources, applies a weighted scoring system, and generates trading signals (Strong Bullish, Bullish, Bearish, Strong Bearish, No Trade) based on user-defined thresholds.

## Features

*   **Multi-Source Data Integration:**  Combines order book, position book, and multiple sentiment data sources for a comprehensive market view.
*   **Rule-Based Trading Strategy:**  Employs a configurable rule-based strategy to generate trading signals.  Users can adjust weights and thresholds.
*   **Real-Time Visualization:**
    *   **Dashboard:**  Provides a summary of trade decisions, a detailed log of system activity, and a high-level overview of trading signal history.
    *   **Detailed Analysis Tab:**  Offers in-depth analysis of a selected instrument, including signal history, component analysis, and visualizations of the order book and position book.
    *   **Signal History Tab:**  Displays a detailed table and chart of the trading signal history for a selected instrument, allowing users to track signal changes over time.
    *   **Order Book Tab:**  Visualizes the OANDA order book, showing long and short order percentages at different price levels, along with cumulative order data.
    *   **Position Book Tab:**  Visualizes the OANDA position book, showing long and short position percentages, along with cumulative position data.
    *   **Sentiment Analysis Tab:**  Displays visualizations of retail positions, Twitter sentiment, and Myfxbook sentiment data.
*   **Interactive GUI:**  Built with Tkinter, the GUI provides a user-friendly interface for monitoring and analyzing trading data.
*   **Configurable Parameters:**  Allows users to customize the API key, account ID, environment (practice or live), update interval, and trading strategy parameters.
*   **Robust Error Handling:** Includes extensive error handling and logging to ensure smooth operation and provide informative feedback.
*   **Caching:** Caches sentiment data to minimize redundant requests and improve performance.
*   **Data Structures**: Uses custom classes (`TradingSignal`, `SignalHistory`) for clean data management.
*   **Modular Design**: Organized into classes (`OandaDataFetcher`, `SentimentDataFetcher`, `TradingStrategy`, `TradingRobotGUI`) for better maintainability.

## Installation

1.  **Prerequisites:**
    *   Python 3.7 or higher.
    *   An OANDA practice or live account.
    *   An OANDA API key.

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/HamoonSoleimani/ForexDashboard
    cd ForexDashboard
    ```

3.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure the script:**
    *   Open `ForexDashboard.py` (or whatever you named the main script file) in a text editor.
    *   Locate the `CONFIGURATION` section near the top of the file.
    *   Replace `"YOUR_API_KEY"` with your actual OANDA API key.
    *   Replace `"YOUR_ACCOUNT_ID"` with your actual OANDA account ID.
    *   Set `ENVIRONMENT` to `"practice"` for a practice account or `"live"` for a live account.  **Use extreme caution with a live account.**

2.  **Run the script:**

    ```bash
    python ForexDashboard.py
    ```

    The GUI will launch, and the robot will begin fetching data and generating trading signals.

## Configuration

The following parameters can be configured within the `ForexDashboard.py` file:

*   **`API_KEY`:** Your OANDA API key.
*   **`ACCOUNT_ID`:** Your OANDA account ID.
*   **`ENVIRONMENT`:**  `"practice"` or `"live"`.
*    **`instruments`**: (In `TradingStrategy`) The instruments that will be used for trading signals.
*   **`weights`:** (In `TradingStrategy`)  The weights assigned to each data source in the trading strategy (order book, position book, pair sentiment, currency sentiment, retail profitability).  These weights should sum to 1.0.
*   **`thresholds`:** (In `TradingStrategy`)  The signal thresholds for each trading decision (Strong Bullish, Bullish, Bearish, Strong Bearish).
*   **`cache_duration`:** (In `SentimentDataFetcher`) The duration (in seconds) for which sentiment data is cached.
*   **Update Interval:**  The time interval (in seconds) between data refreshes.  This can be adjusted in the GUI's footer.

