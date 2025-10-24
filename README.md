# 📈 Financial Analysis Dashboard (v1.0)

This repository contains the source code for a full-stack financial analysis web application built entirely in Python. It leverages **Plotly Dash** as the core framework, **Flask** for the web server and backend logic, and **Flask-SQLAlchemy** for database integration and user authentication.

This dashboard provides a comprehensive suite of tools for investors to perform comparative stock analysis, conduct deep-dive research on individual companies (including technicals, fundamentals, and AI-powered news sentiment), and save their personal preferences and assumption models.

**This marks the completion of Version 1.0.** Future development is planned to enhance the platform, but the timeline is indefinite as I am moving on to other projects to gain experience with different tools.

---

## 🌟 Key Features (v1.0)

### 1. User Authentication System
* **Register & Login:** Users can create a secure account and log in. Passwords are securely hashed using `Werkzeug`.
* **Persistent Storage:** The application saves each user's selections and assumptions to the database (supporting SQLite or PostgreSQL). This includes:
    * Selected stocks and benchmark indices.
    * Custom assumptions for the **Monte Carlo DCF Model**.
    * Custom assumptions for the **Exit Multiple (Target) Model**.

### 2. Main Dashboard (Comparative Analysis)
The main dashboard is designed for comparing a portfolio of selected stocks and benchmarks.

* **Graphs (Visual Comparison):**
    * **Performance (YTD):** Compares the year-to-date percentage return of all selected assets.
    * **Drawdown (1-Year):** Plots the 1-year rolling max drawdown to visualize and compare risk.
    * **Valuation vs. Quality:** A 2x2 scatter plot mapping `EV/EBITDA` (Valuation) against `EBITDA Margin` (Quality).
    * **Margin of Safety (Monte Carlo DCF):**
        * Runs thousands of DCF simulations to generate a probability distribution of a stock's intrinsic value.
        * Users can set assumption ranges (Min, Most Likely, Max) for `Forecast Growth Rate`, `Perpetual Growth Rate`, and `Discount Rate (WACC)`.
        * Displays a histogram of outcomes plotted against the current market price.

* **Tables (Data Comparison):**
    * **Valuation:** (P/E, P/B, EV/EBITDA, etc.)
    * **Growth:** (Revenue Growth YoY, Revenue CAGR 3Y, etc.)
    * **Fundamentals:** (Operating Margin, ROE, D/E Ratio, etc.)
    * **Target (Exit Multiple Model):**
        * Calculates a `Target Price`, `Target Upside (%)`, and `IRR (%)` based on a simple exit multiple valuation.
        * Users can set custom assumptions for `Forecast Years`, `Annual EPS Growth (%)`, and `Terminal P/E Ratio`.

* **Metric Definitions:**
    * Every chart and table includes an "Info" (i) button that opens a modal explaining the definition, calculation formula, and interpretation of each metric.
    * Formulas are rendered cleanly using **MathJax**.

### 3. Deep Dive Page (`/deepdive/<TICKER>`)
Clicking any stock ticker in the main table navigates to a dedicated page for in-depth analysis of that single company.

* **Company Info:** Displays the company logo, name, price, daily change, and a card-based layout of key statistics (Market Cap, P/E, Analyst Target, etc.).
* **Technicals Tab:**
    * A multi-pane chart featuring:
    * Candlestick Price Chart
    * Moving Averages (SMA50, SMA200, EMA20)
    * Bollinger Bands (BB)
    * Relative Strength Index (RSI)
    * MACD (Histogram, MACD Line, Signal Line)
* **Financials Tab:**
    * Provides quarterly Income Statements, Balance Sheets, and Cash Flow Statements in a clean table format.
* **News Tab (AI-Powered Sentiment):**
    * Fetches the latest 20 articles for the company from the **NewsAPI**.
    * Sends all article headlines and descriptions to the **Hugging Face Inference API** for batch sentiment analysis using the `distilroberta-finetuned-financial-news` model.
    * Displays an overall sentiment summary (Positive, Neutral, Negative) as a stacked progress bar.
    * Lists all articles, grouped by date, with their individual sentiment badge.

---

## 🛠️ Tech Stack

* **Backend:** Flask, Gunicorn (for deployment)
* **Web Framework:** Plotly Dash, Dash Bootstrap Components (DBC)
* **Database:** Flask-SQLAlchemy (supports SQLite/PostgreSQL), Flask-Migrate (for schema management)
* **Authentication:** Flask-Login, Werkzeug (for password hashing)
* **Data Analysis:** Pandas, NumPy
* **Data Sources:**
    * `yfinance` (For all stock price data, financials, and statistics)
    * `newsapi-python` (For news articles)
* **AI / Sentiment Analysis:**
    * Hugging Face Inference API (`requests`)
* **Configuration:** `python-dotenv` (for managing environment variables)

---

## ⚙️ Local Installation & Setup

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/financial-dashboard-py.git](https://github.com/your-username/financial-dashboard-py.git)
cd financial-dashboard-py
```

### 2. Create and Activate a Virtual Environment
(Recommended) This isolates the project's dependencies.

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\\venv\\Scripts\\activate
```

### 3. Install Dependencies
Install all required libraries from `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a file named `.env` in the root project directory (at the same level as `config.py`). Add the following necessary keys:

```.env
# 1. Flask Secret Key (Required for sessions & login)
# Run this in your terminal to generate a key:
# python -c 'import os; print(os.urandom(24).hex())'
SECRET_KEY='your_random_secret_key_here'

# 2. (Optional) Database URL
# If left blank, the app will default to a local SQLite database
# located at /instance/app.db
# Example for PostgreSQL:
# SQLALCHEMY_DATABASE_URI='postgresql://user:password@localhost/dbname'

# 3. API Keys (Required for News & Sentiment features)
NEWS_API_KEY='your_newsapi_key_here'
HUGGING_FACE_TOKEN='your_huggingface_read_token_here'
```
* You can get a `NEWS_API_KEY` from [newsapi.org](https://newsapi.org/).
* You can get a `HUGGING_FACE_TOKEN` from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. (Optional) Update Ticker Lists
The `constants.py` file contains ticker lists pre-sorted by market cap. If you wish to refresh this data, you can run the update script:
```bash
python update_constants.py
```

### 6. Run the Application
The `index.py` script will automatically create the database tables if they don't exist and start the server.
```bash
python index.py
```
Open your browser and navigate to **http://127.0.0.1:8050**

---

## 📁 Project Structure

```
financial-dashboard-py/
│
├── .gitignore
├── Procfile             # (For Gunicorn deployment)
├── app.py               # (Core Flask server, Dash app, & DB Models)
├── auth.py              # (Layout & Callbacks for Login/Register)
├── callbacks.py         # (All callbacks for the main dashboard)
├── config.py            # (Handles Configuration and .env)
├── constants.py         # (Stores sorted ticker lists, sectors, colors)
├── index.py             # (Main entry point, controls routing & layout)
├── layout.py            # (Builds the main dashboard layout & modals)
├── data_handler.py      # (Handles all data fetching from yfinance, NewsAPI, HF)
├── requirements.txt     # (Project dependencies)
├── update_constants.py  # (Script to refresh constants.py)
│
├── assets/              # (Static files: CSS and images)
│   ├── logo.png
│   └── style.css
│
├── pages/               # (Sub-pages of the app)
│   └── deep_dive.py     # (Layout & Callbacks for the deep dive page)
│
├── migrations/          # (Flask-Migrate schema versioning)
│   └── ...
│
└── instance/            # (Holds local DB file - ignored by .gitignore)
    └── app.db
```

---

## 🔮 Future Roadmap (v2.0+)

This v1.0 release is stable and feature-complete based on the initial project scope. As I am now shifting focus to other projects, the timeline for v2.0 is indefinite. However, planned future enhancements include:

* **Caching:** Implement Redis to cache results from `yfinance`, `NewsAPI`, and `Hugging Face` to dramatically improve load times and reduce API calls.
* **Asynchronous Tasks:** Move slow, blocking tasks (like API calls in the Deep Dive page) to a background worker (e.g., Celery & Redis) so the UI loads instantly.
* **Advanced Charting:** Integrate more advanced technical indicators or fundamental charts.
* **Dockerization:** Create a `Dockerfile` and `docker-compose.yml` to make the application fully containerized and easily deployable.
