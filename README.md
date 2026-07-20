Here is the complete, professional-grade **Project Documentation and Architectural Design Specification** for your **`deep-oil-agent`** repository. 

This document is formatted as a master technical specification (`README.md` style) that you can copy, save locally as a markdown file, or paste directly into your GitHub repository’s main description page.

***

# Project Documentation: Deep-Oil Neuro-Symbolic Trading Agent

A real-time, self-tuning commodities trading system designed to parse global geopolitical news, vectorize alternative sentiment using Large Language Models, and execute probabilistic trade predictions on **WTI Crude Oil (WTICO-USD)** via a PyTorch LSTM neural network. All trade intents are passed through a deterministic, hardcoded Python risk gate before executing live orders on OANDA.

---

## 1. Core Architectural Philosophy: "Neuro-Symbolic" Integration

Traditional quantitative models are often rigid, failing to adapt to sudden geopolitical escalations (such as Middle East supply risks) [19]. Conversely, pure machine learning or LLM-based bots are highly dangerous when given direct access to exchange execution engines because they suffer from hallucinations and lack structural safety boundaries.

This system resolves this challenge by implementing a **Neuro-Symbolic Architecture** [19, 20]:
* **The Neuro (AI) Layer:** Evaluates qualitative, unstructured textual data (Google News RSS) using **Gemini 3.5**, converting geopolitical concepts into structured sentiment tensors [19, 20]. A **PyTorch LSTM** ingests these tensors alongside price history to discover non-linear market patterns [20].
* **The Symbolic (Hard Logic) Layer:** Implements strict, mathematical rules written in pure Python [20, 24]. The AI can only **propose** a trading direction; the deterministic risk gate has the final authority to **dispose**, resize, or veto the order based on volatility (ATR), account equity, and hardcoded exposure limits [19, 20].

---

## 2. High-Level System Architecture

This diagram illustrates the dual-loop execution flow, detailing how live market price streams and unstructured news headlines are processed, analyzed, filtered, and securely routed to OANDA and your phone:

```
                          HOURLY CRON TRIGGER (0 * * * *)
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION & PERCEPTION                         │
│                                                                                 │
│   [Google News RSS Feed]                                [OANDA REST API]        │
│   Gathers latest OPEC/Iran/US news                      Fetches 5 daily candles │
│             │                                                     │             │
│             ▼                                                     ▼             │
│   [Gemini 3.5 Flash-Lite]                                 [Price Normalizer]    │
│   Parses text into float tensors                          Scales OHLCV values   │
│   [Bullish, Bearish, Supply Shock]                        between 0.0 and 1.0   │
└─────────────┬─────────────────────────────────────────────────────┬─────────────┘
              │                                                     │
              └──────────────────────────┬──────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      2. NEURAL DECISION CORE (PyTorch)                          │
│                                                                                 │
│   Inference Engine: 2-Layer LSTM Neural Network                                 │
│   - Ingests sequence: [Open, High, Low, Close, Volume, Bull, Bear, Shock]       │
│   - Evaluates multi-dimensional time-series correlations                        │
│   - Calculates directional probability score (0.0 to 1.0)                       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Probability Output (e.g., 0.94)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       3. SYMBOLIC RISK SHIELD (Python)                          │
│                                                                                 │
│   Deterministic Risk Gate:                                                      │
│   - Pulls live OANDA wallet balance (e.g., £100,000)                            │
│   - Calculates volatility-adjusted stop-loss (ATR-based)                        │
│   - Limits maximum capital exposure to exactly 1.0% per trade                   │
│   - Approves exact execution units (e.g., 500 units)                            │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Passed All Safety Gates
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       4. ORDER EXECUTION & NOTIFICATION                         │
│                                                                                 │
│         [OANDA V20 API]                ──►               [Telegram API]         │
│     Dispatches Market Order                          Pings your phone with a    │
│   (WTI Perpetual Futures/CFD)                        formatted trade report     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure & Module Breakdown

The project workspace is modularly structured to ensure high security, separation of concerns, and clean operational runs [28]:

```text
deep-oil-agent/
│
├── .env                        # Private credentials (OANDA keys, Telegram token, Gemini API)
├── .gitignore                  # Prevents committing secrets, venv, or temporary logs
├── requirements.txt            # Project dependencies (ccxt, google-genai, torch, pandas)
├── wti_historical_data.csv     # 2-year history of daily WTI futures prices
├── oil_brain_weights.pth       # Saved mathematical weights of the trained LSTM
│
├── data_ingestion/
│   ├── news_scraper.py         # Connects to Google News RSS to scrape target macro headlines
│   └── price_feed.py           # Ingests real-time Klines and tickers
│
├── ai_models/
│   ├── gemini_nlp.py           # Uses Gemini 3.5 to convert headlines into float tensors
│   └── lstm_network.py         # Defines the PyTorch Neural Network architecture
│
├── training/
│   └── train_model.py          # Pre-trains the PyTorch model on historical WTI data
│
├── services/
│   ├── __init__.py
│   ├── exchange_service.py     # Unified wrapper for OANDA V20 execution
│   └── telegram_service.py     # Dispatches formatted Markdown alerts to your phone
│
├── risk_manager.py              # Pure Python Risk Gate (ATR stop placement, Kelly sizing)
├── master_agent.py             # Hourly Orchestrator (Binds Senses, Brain, Shield, and Hands)
└── test_alert_format.py        # Independent testing utility for Telegram parsing
```

---

## 4. Mathematical Sizing & Risk Management

Rather than letting the neural network dictate position sizing, the system uses a deterministic risk-mitigation framework to prevent catastrophic capital drawdowns [20]:

### Volatility-Adjusted Sizing
The trade quantity is calculated dynamically. If the market is highly volatile, your stop-loss is placed further away, and the system automatically scales down your entry size to keep the capital risk identical:

$$\text{Quantity} = \frac{\text{Capital at Risk (1.0\% of Balance)}}{\text{ATR Stop Distance}}$$

Where:
* **Capital at Risk:** Calculated as $\text{Balance} \times 0.01$ (hard-clamped to a maximum of 2% of capital) [20].
* **ATR Stop Distance:** Calculated as $2 \times \text{Average True Range (ATR)}$ [20]. If the average daily move is $1.00, the stop-loss is placed $2.00 away from the entry price [20].

---

## 5. The Self-Learning Paradigm (Dual-Loop Optimization)

Your agent operates on two distinct feedback loops, allowing it to adapt to changing geopolitical environments without overfitting:

### The Fast Loop (Hourly Inference & Execution)
* Triggered hourly by the system cron scheduler [21].
* Collects the newest market snapshots and news headlines, runs them through the trained PyTorch network, passes the result to the risk shield, executes on the exchange, and writes the entire decision context into local logs [20, 24].

### The Slow Loop (Weekly Self-Learning & Fine-Tuning)
* Runs weekly when the commodities markets are closed [19, 20].
* **Step 1:** The system analyzes the database logs of all trades taken over the past 30 days [20].
* **Step 2:** A high-reasoning LLM (acting as a Quantitative Auditor) analyzes under what geopolitical conditions the LSTM's predictions failed [20].
* **Step 3:** The auditor automatically adjusts parameters (like the ATR multiplier or minimum sentiment threshold) inside `hyperparameters.json` [20].
* **Step 4 (Safety Clamp):** Python code clamps these AI adjustments to strict, pre-approved bounds (e.g., maximum allowable risk per trade is clamped to 2% max, overriding any larger requests) [20].
* **Step 5 (Retraining):** The PyTorch model is retrained on your newly accumulated historical news vectors and price outcomes, adjusting its neural weights to better understand macro-economic impacts (e.g., Middle East escalations) [20].

---

## 6. Installation & Operational Playbook

### Installation
Activate your virtual environment and install the required dependencies:
```bash
git clone git@github.com:Durba01/neuro-symbolic-oil-trader.git
cd neuro-symbolic-oil-trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pre-Training the Brain
Before deploying the bot, you must run the data downloader and pre-train your neural network on historical price action:
```bash
# 1. Download 2 years of WTI Crude Oil history
python build_dataset.py

# 2. Train the PyTorch LSTM model
python train_brain.py
```

### Running the Agent
To run a manual trading cycle, verify your OANDA credentials, and run:
```bash
python master_agent.py
```

### Background Automation
To let the bot run autonomously in the background on your system's schedule, add the execution line to your crontab [21]:
```bash
crontab -e
```
*Paste this line at the bottom of the file:*
```text
0 * * * * cd /Users/mj/deep-oil-agent && /Users/mj/deep-oil-agent/venv/bin/python master_agent.py >> master_agent.log 2>&1
```

To monitor the background actions in real-time, run [21]:
```bash
tail -f master_agent.log
```
