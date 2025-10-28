# 📈 Financial Analysis Dashboard (v2.0)

This repository contains the source code for a full-stack financial analysis web application built entirely in Python. It leverages **Plotly Dash** as the core framework, **Flask** for the web server and backend logic, **Flask-SQLAlchemy** for database integration, and **GitHub Actions** for scheduled data processing.

**Version 2.0** marks a significant backend refactoring, introducing a structured **Data Warehouse** and moving the **Extract, Transform, Load (ETL)** process to a scheduled **GitHub Actions workflow**. This ensures data consistency, improves performance, and separates the web application from background data tasks.

The dashboard provides a comprehensive suite of tools for investors to perform comparative stock analysis, conduct deep-dive research on individual companies (including technicals, fundamentals, and AI-powered news sentiment), and save their personal preferences and assumption models.

---

## 🌟 Key Features (v2.0)

### 1. Data Warehouse & ETL Pipeline
* **Structured Data:** Implements a data warehouse schema using `Flask-SQLAlchemy` (supporting PostgreSQL or SQLite) with distinct dimension (`DimCompany`) and fact tables (`FactCompanySummary`, `FactDailyPrices`, `FactFinancialStatements`, `FactNewsSentiment`) to store historical and daily financial data efficiently.
* **Automated ETL:** A robust ETL pipeline (`etl.py`) fetches data from `yfinance` and `NewsAPI`, performs necessary transformations, and upserts it into the data warehouse tables.
* **Scheduled Updates:** The ETL process is triggered daily by a **GitHub Actions workflow** (`.github/workflows/etl.yml`), ensuring the data warehouse stays up-to-date without requiring a separate, always-on server process.

### 2. User Authentication System
* **Register & Login:** Users can create secure accounts and log in. Passwords are securely hashed using `Werkzeug`.
* **Persistent Storage:** User selections (stocks, indices) and custom financial model assumptions (DCF, Exit Multiple) are saved securely to the database linked to their account.

### 3. Main Dashboard (Comparative Analysis)
Powered by data queried directly from the warehouse, this dashboard facilitates comparison across selected stocks and benchmarks.

* **Graphs (Visual Comparison):**
    * Performance (YTD)
    * Drawdown (1-Year)
    * Valuation vs. Quality (EV/EBITDA vs. EBITDA Margin)
    * Margin of Safety (Monte Carlo DCF) with customizable assumption ranges.
* **Tables (Data Comparison):**
    * Valuation Metrics
    * Growth Metrics
    * Fundamentals Metrics
    * Target/Forecast (Exit Multiple Model) with customizable assumptions.
* **Metric Definitions:** On-demand explanations for all metrics via modal pop-ups, with formulas rendered using MathJax.

### 4. Deep Dive Page (`/deepdive/<TICKER>`)
Offers in-depth analysis for individual companies, pulling relevant data from the warehouse or fetching live news via API for non-ETL tickers.

* **Company Info & Key Stats:** Header displaying logo, name, price, daily change, and essential statistics.
* **Technicals Tab:** Interactive candlestick chart with Moving Averages, Bollinger Bands, RSI, and MACD.
* **Financials Tab:** Quarterly Income Statements, Balance Sheets, and Cash Flow Statements presented in tables.
* **News Tab (AI-Powered Sentiment):**
    * Fetches recent news articles (from DB for Top 20 Market Cap stocks, live via NewsAPI otherwise).
    * Performs batch sentiment analysis using Hugging Face Inference API.
    * Displays overall sentiment summary and individual article sentiments.

---

## 🛠️ Tech Stack

* **Backend:** Flask, Gunicorn (for deployment)
* **Web Framework:** Plotly Dash, Dash Bootstrap Components (DBC)
* **Database:** Flask-SQLAlchemy (supports PostgreSQL/SQLite), Flask-Migrate (for schema management)
* **Authentication:** Flask-Login, Werkzeug (for password hashing)
* **Data Analysis:** Pandas, NumPy
* **Data Sources:**
    * `yfinance` (Stock price, financials, statistics)
    * `newsapi-python` (News articles)
* **AI / Sentiment Analysis:**
    * Hugging Face Inference API (`requests`)
* **Scheduled Tasks:** GitHub Actions
* **Configuration:** `python-dotenv`

---

## ⚙️ Local Installation & Setup

1.  **Clone:** `git clone https://github.com/your-username/financial-dashboard-py.git && cd financial-dashboard-py`
2.  **Environment:** Create and activate a Python virtual environment (e.g., `python -m venv venv && source venv/bin/activate`).
3.  **Install:** `pip install -r requirements.txt`
4.  **Environment Variables:** Create a `.env` file in the root directory. Add your `SECRET_KEY`, API keys (`NEWS_API_KEY`, `HUGGING_FACE_TOKEN`), and optionally your `SQLALCHEMY_DATABASE_URI`. If `SQLALCHEMY_DATABASE_URI` is omitted, it defaults to a local SQLite DB (`instance/app.db`).
5.  **Database Setup:**
    * Run `python create_db.py` *once* if using SQLite to create the initial DB file.
    * Run `flask db upgrade` to apply all database migrations.
6.  **Run ETL Manually (First Time):** Populate the database initially by running `python run_etl_manually.py`. This might take a while.
7.  **(Optional) Update Tickers:** Refresh `constants.py` with latest market cap sorting using `python update_constants.py`.
8.  **Run Web App:** Start the local development server with `python index.py`.

---

## 🔮 Future Roadmap (v3.0+)

With a solid data foundation in place, v3.0 can focus on integrating Machine Learning capabilities:

* **ML Predictions:** Train models (e.g., Time Series Forecasting, Classification) on the warehouse data to predict future price movements or classify stock profiles.
* **Enhanced Sentiment Analysis:** Fine-tune sentiment models or use more advanced NLP techniques on the collected news data.
* **Factor Analysis:** Use ML to identify key financial factors driving stock performance within different sectors based on historical data.
* **Caching:** Implement Redis to cache database queries and API results for faster UI loading.
* **Asynchronous Tasks:** Potentially use Celery/Redis for any *new*, long-running ML model training or inference tasks initiated by user interaction (though daily ETL remains on GitHub Actions).
* **Dockerization:** Containerize the application for easier deployment and scalability.
