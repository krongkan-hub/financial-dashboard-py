# data_handler.py (Final Robust Version)

import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from functools import lru_cache
import logging
import warnings
from datetime import datetime, timedelta
from urllib.parse import urlparse
import requests 
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from newsapi import NewsApiClient

# --- Initialize NewsAPI ---
if Config.NEWS_API_KEY:
    newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
else:
    newsapi = None
    logging.warning("NEWS_API_KEY not found in config. News fetching will be disabled.")

# --- Hugging Face API Function ---
def analyze_sentiment_via_api(text_to_analyze: str) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    hf_token = os.environ.get('HUGGING_FACE_TOKEN')
    if not hf_token:
        logging.warning("HUGGING_FACE_TOKEN not found. Returning neutral sentiment.")
        return {'label': 'neutral', 'score': 0.0}
        
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text_to_analyze[:512]}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        if result and isinstance(result, list) and isinstance(result[0], list):
            best_prediction = max(result[0], key=lambda x: x['score'])
            return {'label': best_prediction['label'].lower(), 'score': best_prediction['score']}
    except Exception as e:
        logging.error(f"API request failed or parsing failed: {e}")
        
    return {'label': 'neutral', 'score': 0.0}

# --- News Fetching Function ---
@lru_cache(maxsize=32)
def get_news_and_sentiment(company_name: str) -> dict:
    if not newsapi:
        return {"error": "News service is not configured."}
    try:
        # ... (rest of the function is unchanged)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        all_articles = newsapi.get_everything(q=f'"{company_name}"', language='en', sort_by='relevancy', from_param=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'), page_size=20)
        if all_articles['status'] != 'ok' or all_articles['totalResults'] == 0: return {"articles": [], "summary": {}}
        analyzed_articles, sentiment_scores = [], {'positive': 0, 'neutral': 0, 'negative': 0}
        for article in all_articles['articles']:
            if not article.get('description'): continue
            text_to_analyze = article['title'] + ". " + article['description']
            result = analyze_sentiment_via_api(text_to_analyze)
            article['sentiment'], article['sentiment_score'] = result['label'], result['score']
            analyzed_articles.append(article)
            sentiment_scores[result['label']] += 1
        total_analyzed = len(analyzed_articles)
        summary = {
            'positive_count': sentiment_scores['positive'], 'neutral_count': sentiment_scores['neutral'], 'negative_count': sentiment_scores['negative'], 'total_count': total_analyzed,
            'positive_pct': (sentiment_scores['positive'] / total_analyzed) * 100 if total_analyzed > 0 else 0,
            'neutral_pct': (sentiment_scores['neutral'] / total_analyzed) * 100 if total_analyzed > 0 else 0,
            'negative_pct': (sentiment_scores['negative'] / total_analyzed) * 100 if total_analyzed > 0 else 0,
        }
        return {"articles": analyzed_articles, "summary": summary}
    except Exception as e:
        logging.error(f"Error in get_news_and_sentiment for {company_name}: {e}", exc_info=True)
        return {"error": str(e)}

# --- Helper Functions ---
FIN_KEYS = { "revenue": ["Total Revenue", "Revenues", "Revenue"], "cost_of_revenue": ["Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold"], "gross_profit": ["Gross Profit"], "op_income": ["Operating Income", "Operating Income or Loss", "Ebit", "Earnings Before Interest and Taxes"], "net_income": ["Net Income", "Net Income From Continuing Ops", "Net Income Applicable To Common Shares"],}
CF_KEYS = { "cfo": ["Total Cash From Operating Activities", "Operating Cash Flow"], "capex": ["Capital Expenditures"] }

def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    clean_lower_index = df.index.str.lower().str.strip()
    for candidate in candidates:
        try:
            clean_candidate = candidate.lower().strip()
            if clean_candidate in clean_lower_index.to_list():
                idx_pos = clean_lower_index.to_list().index(clean_candidate)
                return df.iloc[idx_pos]
        except Exception: continue
    return None

def _cagr(series: pd.Series, years: int) -> float:
    series = series.dropna().sort_index().tail(years + 1)
    if len(series) < 2: return np.nan
    num_years = (series.index[-1].year - series.index[0].year) if pd.api.types.is_datetime64_any_dtype(series.index) else len(series) - 1
    if num_years <= 0: return np.nan
    start_val, end_val = series.iloc[0], series.iloc[-1]
    if pd.notna(start_val) and pd.notna(end_val) and start_val > 0:
        return (end_val / start_val)**(1.0 / num_years) - 1.0
    return np.nan

def get_financial_data(statement_series, possible_keys):
    if statement_series is None: return None
    for key in possible_keys:
        if key in statement_series.index:
            value = statement_series[key]
            return abs(value) if pd.notna(value) else None
    return None

def _get_logo_url(info: dict) -> Optional[str]:
    if not info: return None
    logo_url = info.get('logo_url')
    if not logo_url:
        website = info.get('website')
        if website:
            try:
                domain = urlparse(website).netloc.replace('www.', '')
                if domain: logo_url = f"https://logo.clearbit.com/{domain}"
            except Exception: logo_url = None
    return logo_url

# --- Data Fetching Functions ---
@lru_cache(maxsize=32)
def get_revenue_series(ticker: str) -> pd.Series:
    try:
        tkr = yf.Ticker(ticker)
        fin = tkr.financials
        if fin is not None and not fin.empty:
            revenue = _pick_row(fin, FIN_KEYS["revenue"])
            if revenue is not None:
                revenue.index = pd.to_datetime(revenue.index)
                return revenue.sort_index()
    except Exception as e: logging.error(f"Failed to fetch revenue for {ticker}: {e}")
    return pd.Series(dtype=float)

@lru_cache(maxsize=20)
def calculate_drawdown(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    try:
        data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
        if data.empty: return pd.DataFrame()
        prices = data.get('Close', pd.DataFrame())
        if isinstance(prices, pd.Series): prices = prices.to_frame(name=tickers[0])
        if prices.empty: return pd.DataFrame()
        rolling_max = prices.cummax()
        return (prices / rolling_max) - 1
    except Exception as e:
        logging.error(f"Error in calculate_drawdown for {tickers}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=10)
def get_competitor_data(tickers: tuple) -> pd.DataFrame:
    all_data = []
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info
            if not info or info.get('quoteType') != 'EQUITY': continue
            logo_url, revenue_series = _get_logo_url(info), get_revenue_series(ticker)
            cagr_3y = _cagr(revenue_series, 3) if not revenue_series.empty else np.nan
            cfo_series, ni_series = _pick_row(tkr.cashflow, CF_KEYS['cfo']), _pick_row(tkr.financials, FIN_KEYS['net_income'])
            cash_conversion = np.nan
            if cfo_series is not None and ni_series is not None and not cfo_series.empty and not ni_series.empty:
                latest_cfo, latest_ni = cfo_series.iloc[0], ni_series.iloc[0]
                if pd.notna(latest_cfo) and pd.notna(latest_ni) and latest_ni != 0: cash_conversion = latest_cfo / latest_ni
            all_data.append({
                "Ticker": ticker, "logo_url": logo_url, "Price": info.get('currentPrice') or info.get('previousClose'),
                "Market Cap": info.get("marketCap"), "Beta": info.get("beta"), "P/E": info.get('trailingPE'), "P/B": info.get('priceToBook'),
                "EV/EBITDA": info.get('enterpriseToEbitda'), "Revenue Growth (YoY)": info.get('revenueGrowth'), "Revenue CAGR (3Y)": cagr_3y,
                "Net Income Growth (YoY)": info.get('earningsGrowth'), "ROE": info.get('returnOnEquity'),
                "D/E Ratio": info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') is not None else np.nan,
                "Operating Margin": info.get('operatingMargins'), "Cash Conversion": cash_conversion
            })
        except Exception as e: logging.error(f"An error occurred processing summary for {ticker}: {e}")
    return pd.DataFrame(all_data)

@lru_cache(maxsize=10)
def get_scatter_data(tickers: tuple) -> pd.DataFrame:
    scatter_data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if not info: continue
            scatter_data.append({'Ticker': ticker, 'EV/EBITDA': info.get('enterpriseToEbitda'), 'EBITDA Margin': info.get('ebitdaMargins')})
        except Exception as e: logging.warning(f"Could not fetch scatter data for {ticker}: {e}")
    return pd.DataFrame(scatter_data).dropna()

@lru_cache(maxsize=32)
def get_deep_dive_data(ticker: str) -> dict:
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        if not info or info.get('quoteType') != 'EQUITY': return {"error": "Invalid ticker or no data available."}
        # ... (rest of the function is unchanged)
        company_name, result = info.get('longName', ticker), {"company_name": info.get('longName', ticker)}
        income_stmt_quarterly = tkr.quarterly_financials.iloc[:, :16]
        revenue = _pick_row(income_stmt_quarterly, FIN_KEYS['revenue'])
        net_income = _pick_row(income_stmt_quarterly, FIN_KEYS['net_income'])
        financial_trends = pd.DataFrame({'Revenue': revenue, 'Net Income': net_income}).dropna(how='all')
        if not financial_trends.empty:
            financial_trends.index = pd.to_datetime(financial_trends.index).to_period('Q').strftime('%Y-Q%q')
        result["financial_trends"] = financial_trends.sort_index()
        margin_trends = pd.DataFrame(index=financial_trends.index)
        if revenue is not None and not revenue.empty:
            revenue_safe = revenue.replace(0, np.nan)
            gross_profit, op_income = _pick_row(income_stmt_quarterly, FIN_KEYS['gross_profit']), _pick_row(income_stmt_quarterly, FIN_KEYS['op_income'])
            if gross_profit is not None: margin_trends['Gross Margin'] = gross_profit / revenue_safe
            if op_income is not None: margin_trends['Operating Margin'] = op_income / revenue_safe
            if net_income is not None: margin_trends['Net Margin'] = net_income / revenue_safe
        result["margin_trends"] = margin_trends.sort_index().dropna(how='all')
        result.update({
            "financial_statements": {"income": tkr.quarterly_financials.iloc[:,:16].dropna(how='all', axis=1), "balance": tkr.quarterly_balance_sheet.iloc[:,:16].dropna(how='all', axis=1), "cashflow": tkr.quarterly_cashflow.iloc[:,:16].dropna(how='all', axis=1)},
            "price_history": tkr.history(period="5y")
        })
        return result
    except Exception as e:
        logging.error(f"Critical failure in get_deep_dive_data for {ticker}: {e}", exc_info=True)
        return {"error": str(e)}

# --- [MODIFIED] Functions with better error handling ---
@lru_cache(maxsize=32)
def calculate_dcf_intrinsic_value(ticker: str, forecast_growth_rate: float, perpetual_growth_rate: float, wacc_override: Optional[float]) -> dict:
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info: return {'Ticker': ticker, 'error': 'Could not fetch company info from yfinance.'}
        
        # ... (rest of the DCF logic is unchanged)
        ASSUMED_PERPETUAL_GROWTH, PROJECTION_YEARS, ASSUMED_MARKET_RETURN = perpetual_growth_rate, 5, 0.08
        income_stmt, balance_sheet, cashflow = ticker_obj.financials, ticker_obj.balance_sheet, ticker_obj.cashflow
        if any(df.empty for df in [income_stmt, balance_sheet, cashflow]): return {'Ticker': ticker, 'error': 'Financial statements missing.'}
        last_year_income, last_year_balance, last_year_cashflow = income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0], cashflow.iloc[:, 0]
        ebit, tax_provision, pretax_income = get_financial_data(last_year_income, ['EBIT', 'Ebit']), get_financial_data(last_year_income, ['Tax Provision', 'Income Tax Expense']), get_financial_data(last_year_income, ['Pretax Income', 'Income Before Tax'])
        d_and_a, capex = get_financial_data(last_year_cashflow, ['Depreciation And Amortization', 'Depreciation']), get_financial_data(last_year_cashflow, ['Capital Expenditure', 'CapEx'])
        cash, total_debt = get_financial_data(last_year_balance, ['Cash And Cash Equivalents', 'Cash']), get_financial_data(last_year_balance, ['Total Debt'])
        interest_expense = get_financial_data(last_year_income, ['Interest Expense', 'Interest Expense Net'])
        market_cap, beta = info.get('marketCap'), info.get('beta')
        shares_outstanding, current_price = info.get('sharesOutstanding'), info.get('currentPrice') or info.get('previousClose')
        required = [ebit, tax_provision, pretax_income, d_and_a, capex, cash, shares_outstanding, current_price]
        if any(v is None for v in required): return {'Ticker': ticker, 'error': 'Missing essential data for DCF.'}
        tax_rate = (tax_provision / pretax_income) if pretax_income and pretax_income != 0 else 0.21
        base_fcff = (ebit * (1 - tax_rate)) + d_and_a - capex
        wacc = wacc_override
        if wacc is None:
            if any(v is None for v in [beta, market_cap, total_debt, interest_expense]): return {'Ticker': ticker, 'error': 'Missing data for WACC calc.'}
            cost_of_debt_rd = (interest_expense / total_debt) if total_debt != 0 else 0.05
            tnx_history = yf.Ticker('^TNX').history(period='1d')
            risk_free_rate = (tnx_history['Close'].iloc[0] / 100) if not tnx_history.empty else 0.04
            cost_of_equity_re = risk_free_rate + beta * (ASSUMED_MARKET_RETURN - risk_free_rate)
            total_capital = market_cap + total_debt
            if total_capital == 0: return {'Ticker': ticker, 'error': 'Total capital is zero.'}
            wacc = ((market_cap / total_capital) * cost_of_equity_re) + ((total_debt / total_capital) * cost_of_debt_rd * (1 - tax_rate))
        if wacc <= ASSUMED_PERPETUAL_GROWTH: return {'Ticker': ticker, 'error': f'WACC ({wacc:.2%}) <= perpetual growth.'}
        future_fcffs = [base_fcff * ((1 + forecast_growth_rate) ** year) for year in range(1, PROJECTION_YEARS + 1)]
        discounted_fcffs = [fcff / ((1 + wacc) ** (year + 1)) for year, fcff in enumerate(future_fcffs)]
        terminal_value = (future_fcffs[-1] * (1 + ASSUMED_PERPETUAL_GROWTH)) / (wacc - ASSUMED_PERPETUAL_GROWTH)
        discounted_terminal_value = terminal_value / ((1 + wacc) ** PROJECTION_YEARS)
        enterprise_value = sum(discounted_fcffs) + discounted_terminal_value
        net_debt = (total_debt or 0) - cash
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding
        return {'Ticker': ticker, 'intrinsic_value': intrinsic_value_per_share, 'current_price': current_price, 'wacc': wacc}
    except Exception as e:
        logging.error(f"Critical DCF failure for {ticker}: {e}", exc_info=True)
        return {'Ticker': ticker, 'error': 'Unexpected error during calculation.'}

def get_technical_analysis_data(price_history_df: pd.DataFrame) -> dict:
    if not isinstance(price_history_df, pd.DataFrame) or price_history_df.empty: return {"error": "Price history data is missing."}
    df = price_history_df.copy()
    # ... (rest of the function is unchanged)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12, ema26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return {"data": df}

@lru_cache(maxsize=32)
def calculate_exit_multiple_valuation(ticker: str, forecast_years: int, eps_growth_rate: float, terminal_pe: float) -> dict:
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return {'error': 'Could not fetch company info from yfinance.'}
            
        current_price = info.get('currentPrice') or info.get('previousClose')
        trailing_eps = info.get('trailingEps')
        
        if not all([current_price, trailing_eps, trailing_eps > 0]):
            return {'error': 'Missing essential data (Price or EPS) for valuation.'}
        
        eps_growth_decimal = eps_growth_rate / 100.0
        future_eps = trailing_eps * ((1 + eps_growth_decimal) ** forecast_years)
        target_price = future_eps * terminal_pe
        target_upside = (target_price / current_price) - 1 if current_price > 0 else 0
        irr = ((target_price / current_price) ** (1 / forecast_years)) - 1 if current_price > 0 and forecast_years > 0 else 0
        return {'Target Price': target_price, 'Target Upside': target_upside, 'IRR %': irr}
    except Exception as e:
        logging.error(f"Exit multiple valuation failed for {ticker}: {e}")
        return {'error': 'Calculation failed unexpectedly.'}