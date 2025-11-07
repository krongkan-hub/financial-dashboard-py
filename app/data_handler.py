# data_handler.py (V-FINAL: ใช้วิธี Groupby-Apply (V8) เพื่อแก้ Bug merge_asof)
# (MODIFIED: 2025-11-01 - แก้ไข _get_dcf_base_data_from_db ให้รองรับ Metric ทางเลือก)
# (MODIFIED: 2025-11-02 - แก้ไข Timeout โดยการดึงราคาแบบ Batch รายปี)

import os
import requests
import logging
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional, Tuple # <<< Added Tuple
from functools import lru_cache
import logging
import warnings
from datetime import datetime, timedelta
from urllib.parse import urlparse
import requests
import os
import time
from . import db, server 
from .models import FactFinancialStatements, FactCompanySummary, DimCompany, FactDailyPrices
from sqlalchemy import func, distinct, text, desc
from .config import Config 
from .constants import ALL_TICKERS_SORTED_BY_GROWTH, INDEX_TICKER_TO_NAME, HISTORICAL_START_DATE




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)


from newsapi import NewsApiClient

load_dotenv()
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

# --- [WACC CONSTANTS] ---
ASSUMED_MARKET_RETURN = 0.08  # 8.0% Market Risk Premium assumption
ASSUMED_RISK_FREE_RATE = 0.04 # 4.0% Risk-Free Rate assumption (simulated for simplicity)
# --- [END WACC CONSTANTS] ---

# --- Initialize NewsAPI (เหมือนเดิม) ---
if Config.NEWS_API_KEY:
    newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
else:
    newsapi = None
    logging.warning("NEWS_API_KEY not found in config. News fetching will be disabled.")

# --- Hugging Face API Function (เหมือนเดิม) ---
def analyze_sentiment_batch(texts_to_analyze: List[str]) -> List[dict]:
    # ... (โค้ดเดิม) ...
    API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    hf_token = os.environ.get('HUGGING_FACE_TOKEN')
    default_result = {'label': 'neutral', 'score': 0.0}

    if not hf_token:
        logging.warning("HUGGING_FACE_TOKEN not found. Returning neutral sentiment for all.")
        return [default_result] * len(texts_to_analyze)
    if not texts_to_analyze:
        return []

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": texts_to_analyze, "options": {"wait_for_model": True}}

    final_results = []

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        results_batch = response.json()

        if isinstance(results_batch, list) and len(results_batch) > 0:
            inner_results_list = results_batch[0]
            if isinstance(inner_results_list, list) and len(inner_results_list) == len(texts_to_analyze):
                for result_for_one_article in inner_results_list:
                    if isinstance(result_for_one_article, dict) and 'label' in result_for_one_article:
                        final_results.append({
                            'label': result_for_one_article.get('label', 'neutral').lower(),
                            'score': result_for_one_article.get('score', 0.0)
                        })
                    else:
                        logging.warning(f"Unexpected inner structure for one article: {result_for_one_article}")
                        final_results.append(default_result)
            else:
                logging.error(f"API response *inner* list validation failed (length mismatch or not a list): {inner_results_list}")
                return [default_result] * len(texts_to_analyze)
        else:
            logging.error(f"API response *outer* structure unexpected or empty: {results_batch}")
            return [default_result] * len(texts_to_analyze)

    except requests.exceptions.Timeout:
        logging.error("API request timed out after 120 seconds.", exc_info=False)
        return [default_result] * len(texts_to_analyze)
    except requests.exceptions.RequestException as req_e:
        logging.error(f"API request failed: {req_e}", exc_info=True)
        return [default_result] * len(texts_to_analyze)
    except Exception as e:
        logging.error(f"API batch request failed or parsing failed: {e}", exc_info=True)
        return [default_result] * len(texts_to_analyze)

    return final_results

# --- News Fetching Function (เหมือนเดิม) ---
@lru_cache(maxsize=32)
def get_news_and_sentiment(company_name: str) -> dict:
    # ... (โค้ดเดิม) ...
    if not newsapi: return {"error": "News service is not configured."}
    try:
        to_date, from_date = datetime.now(), datetime.now() - timedelta(days=7)
        all_articles = newsapi.get_everything(q=f'"{company_name}"', language='en', sort_by='relevancy', from_param=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'), page_size=20)

        if all_articles['status'] != 'ok' or all_articles['totalResults'] == 0:
            return {"articles": [], "summary": {}}

        articles_to_process = []
        texts_to_analyze = []
        for article in all_articles['articles']:
            if not article.get('description') or not article.get('title'): continue
            text = (article['title'] + ". " + article['description'])[:512]
            articles_to_process.append(article)
            texts_to_analyze.append(text)

        if not articles_to_process:
             return {"articles": [], "summary": {}}

        logging.info(f"Sending {len(texts_to_analyze)} articles to batch sentiment analysis...")
        sentiment_results = analyze_sentiment_batch(texts_to_analyze)
        logging.info("Batch analysis complete.")

        analyzed_articles = []
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}

        for article, result in zip(articles_to_process, sentiment_results):
            article['sentiment'] = result['label']
            article['sentiment_score'] = result['score']
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

# --- Helper Functions & Constants (เหมือนเดิม) ---
FIN_KEYS = {"revenue": ["Total Revenue", "Revenues", "Revenue"],"cost_of_revenue": ["Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold"],"gross_profit": ["Gross Profit"],"op_income": ["Operating Income", "Operating Income or Loss", "Ebit", "Earnings Before Interest and Taxes"],"net_income": ["Net Income", "Net Income From Continuing Ops", "Net Income Applicable To Common Shares"],}
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
        try:
            result = (end_val / start_val)**(1.0 / num_years) - 1.0
            if isinstance(result, complex):
                return np.nan
            return float(result)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return np.nan
    return np.nan

def get_financial_data(statement_series, possible_keys):
    if statement_series is None: return None
    for key in possible_keys:
        if key in statement_series.index:
            value = statement_series[key]
            return abs(value) if pd.notna(value) else None
    return None

# --- Core Data Fetching Functions (เหมือนเดิม ยกเว้น get_deep_dive_data ที่เอาออก) ---
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
        prices = data.get('Close', pd.DataFrame())
        if prices.empty: return pd.DataFrame()
        if isinstance(prices, pd.Series): prices = prices.to_frame(name=list(tickers)[0])
        rolling_max = prices.cummax()
        return (prices / rolling_max) - 1
    except Exception as e:
        logging.error(f"Error in calculate_drawdown for {tickers}: {e}")
        return pd.DataFrame()

# --- [NEW] Logo Fetching Helper (with Rate Limit Handling) ---
# --- [MODIFIED] Logo Fetching Helper (with yfinance Fallback & using info dict) ---
# Function now accepts the info dictionary fetched by the caller
def _get_logo_url(ticker: str, info: Optional[dict] = None) -> Optional[str]:
    """
    Fetches the company logo URL.
    Tries Finnhub API first. If it fails or returns None,
    tries to get it from the yfinance info dictionary if provided.
    """
    finnhub_logo = None

    # 1. Try Finnhub first (เหมือนเดิม)
    if FINNHUB_API_KEY:
        try:
            url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            finnhub_logo = data.get('logo')

            if finnhub_logo:
                logging.info(f"Successfully fetched logo for {ticker} from Finnhub.")
                return finnhub_logo # Return immediately if found via Finnhub
            else:
                 logging.info(f"No logo URL found in Finnhub data for {ticker}. Will try yfinance fallback if info provided.")

        except requests.exceptions.HTTPError as http_err:
            if r.status_code == 429:
                logging.warning(f"Rate limit hit while fetching logo for {ticker} from Finnhub. Trying yfinance fallback.")
            else:
                logging.warning(f"HTTP error fetching logo from Finnhub for {ticker}: {http_err}. Trying yfinance fallback.")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request error fetching logo from Finnhub for {ticker}: {req_err}. Trying yfinance fallback.")
        except Exception as e:
            logging.error(f"Unexpected error fetching logo from Finnhub for {ticker}: {e}. Trying yfinance fallback.")
    else:
        logging.warning("FINNHUB_API_KEY not set. Skipping Finnhub, trying yfinance fallback.")

    # 2. Fallback to yfinance info dictionary (if info was passed)
    if info and isinstance(info, dict):
        yfinance_logo = info.get('logo_url')
        if yfinance_logo:
            logging.info(f"Found logo for {ticker} via yfinance info dictionary fallback.")
            return yfinance_logo
        else:
             logging.info(f"No logo_url key found in yfinance info dictionary for {ticker}.")

    # 3. If neither worked, return None
    logging.warning(f"Could not retrieve logo for {ticker} from any source.")
    return None

# --- [END Logo Helper] ---

@lru_cache(maxsize=10) # ใช้ cache เหมือนฟังก์ชันอื่นๆ
def get_market_cap_data(tickers: tuple) -> pd.DataFrame:
    """
    ฟังก์ชันน้ำหนักเบาที่ปรับให้เหมาะสมเพื่อดึง *เฉพาะ* Ticker และ Market Cap
    สำหรับใช้ในการจัดเรียง (ถูกเรียกใช้โดย update_constants.py)
    """
    all_data = []
    total = len(tickers)
    logging.info(f"Starting lightweight market cap fetch for {total} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info
            
            # ตรวจสอบข้อมูลพื้นฐาน
            if not info or info.get('quoteType') != 'EQUITY' or info.get('marketCap') is None:
                logging.warning(f"({i+1}/{total}) Skipping {ticker}: Invalid ticker or no market cap.")
                continue

            data_dict = {
                "Ticker": ticker,
                "Market Cap": info.get("marketCap")
            }
            all_data.append(data_dict)

            # หน่วงเวลาเล็กน้อยเพื่อไม่ให้ API ถูกใช้งานหนักเกินไป
            time.sleep(0.2) 

            if (i + 1) % 100 == 0: # Log ความคืบหน้าทุก 100 tickers
                 logging.info(f"({i+1}/{total}) Fetched market caps...")

        except Exception as e:
            # หากเกิดข้อผิดพลาด ให้ข้ามไปทำ Ticker ถัดไป
            logging.error(f"({i+1}/{total}) An error occurred processing market cap for {ticker}: {e}")
            time.sleep(0.5) # หน่วงเวลาเล็กน้อยเมื่อเกิด error
            continue
    
    logging.info(f"Finished lightweight fetch. Got data for {len(all_data)} out of {total} tickers.")
    return pd.DataFrame(all_data)

# --- [MODIFIED] get_competitor_data (เหมือนเดิม) ---
@lru_cache(maxsize=10)
def get_competitor_data(tickers: tuple) -> pd.DataFrame:
    all_data = []
    total = len(tickers)
    logging.info(f"Starting to fetch competitor data for {total} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info # Fetch info once
            if not info or info.get('quoteType') != 'EQUITY':
                logging.warning(f"({i+1}/{total}) Skipping {ticker}: Invalid ticker or no data.")
                continue # Skip to the next ticker if info is bad

            logging.info(f"({i+1}/{total}) Fetching data for {ticker}...")

            logo_url = _get_logo_url(ticker, info) # <<< Use the existing helper
            # Use financials/cashflow from the same tkr object to reduce API calls
            fin, cf = tkr.financials, tkr.cashflow

            revenue_series = _pick_row(fin, FIN_KEYS["revenue"]) if fin is not None and not fin.empty else pd.Series(dtype=float)
            if revenue_series is not None and not revenue_series.empty:
                revenue_series.index = pd.to_datetime(revenue_series.index)
                revenue_series = revenue_series.sort_index()
            cagr_3y = _cagr(revenue_series, 3) if revenue_series is not None and not revenue_series.empty else np.nan

            cfo_series = _pick_row(cf, CF_KEYS['cfo']) if cf is not None and not cf.empty else None
            ni_series = _pick_row(fin, FIN_KEYS['net_income']) if fin is not None and not fin.empty else None

            cash_conversion = np.nan
            if cfo_series is not None and ni_series is not None and not cfo_series.empty and not ni_series.empty:
                # Ensure indices are aligned if possible, or use latest available period
                aligned_cfo = cfo_series.reindex(ni_series.index, method='ffill').dropna()
                aligned_ni = ni_series.reindex(cfo_series.index, method='ffill').dropna()
                common_index = aligned_cfo.index.intersection(aligned_ni.index)
                if not common_index.empty:
                    latest_common_date = common_index.max()
                    latest_cfo = aligned_cfo.get(latest_common_date)
                    latest_ni = aligned_ni.get(latest_common_date)
                    if pd.notna(latest_cfo) and pd.notna(latest_ni) and latest_ni != 0:
                        cash_conversion = latest_cfo / latest_ni
                elif not cfo_series.empty and not ni_series.empty: # Fallback if no common index
                    latest_cfo, latest_ni = cfo_series.iloc[0], ni_series.iloc[0]
                    if pd.notna(latest_cfo) and pd.notna(latest_ni) and latest_ni != 0:
                        cash_conversion = latest_cfo / latest_ni


            data_dict = {
                "Ticker": ticker,
                "company_name": info.get('longName'), # <--- Added for DimCompany
                "sector": info.get('sector'), # <--- Added for DimCompany
                "logo_url": logo_url,
                "credit_rating": info.get('creditRating', None),
                "Price": info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'),
                "Market Cap": info.get("marketCap"),
                "Beta": info.get("beta"),
                "P/E": info.get('trailingPE'),
                "P/B": info.get('priceToBook'),
                "EV/EBITDA": info.get('enterpriseToEbitda'),
                "Revenue Growth (YoY)": info.get('revenueGrowth'),
                "Revenue CAGR (3Y)": cagr_3y,
                "Net Income Growth (YoY)": info.get('earningsGrowth'),
                "ROE": info.get('returnOnEquity'),
                "D/E Ratio": info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') is not None else np.nan,
                "Operating Margin": info.get('operatingMargins'),
                "Cash Conversion": cash_conversion,
                # --- [ข้อมูลใหม่ที่เพิ่มเข้ามา] ---
                "EBITDA Margin": info.get('ebitdaMargins'),
                "Trailing EPS": info.get('trailingEps'),
                "Forward P/E": info.get('forwardPE'),
                "Analyst Target Price": info.get('targetMeanPrice'),
                "Long Business Summary": info.get('longBusinessSummary')
            }
            all_data.append(data_dict)

            # --- [FIX 4] เพิ่มการหน่วงเวลา ---
            time.sleep(0.5) # Wait 0.5 second between tickers

        except Exception as e:
            # Log the specific error for the ticker but continue the loop
            logging.error(f"({i+1}/{total}) An error occurred processing summary for {ticker}: {e}")
            time.sleep(1) # General short sleep on error before trying next ticker
            continue # Continue to the next ticker

    logging.info(f"Finished fetching competitor data. Got data for {len(all_data)} out of {total} tickers.")
    return pd.DataFrame(all_data)
# --- [END OF MODIFICATION] ---

@lru_cache(maxsize=10)
def get_scatter_data(tickers: tuple) -> pd.DataFrame:
    # ... (โค้ดเดิม) ...
    scatter_data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if not info: continue
            scatter_data.append({'Ticker': ticker, 'EV/EBITDA': info.get('enterpriseToEbitda'), 'EBITDA Margin': info.get('ebitdaMargins')})
            time.sleep(0.2) # Added small delay
        except Exception as e:
             logging.warning(f"Could not fetch scatter data for {ticker}: {e}")
             time.sleep(5) # Longer delay on error
    return pd.DataFrame(scatter_data).dropna()

# --- [REMOVED] get_deep_dive_data function ---
# (ฟังก์ชันนี้ไม่จำเป็นแล้ว เพราะเราจะสร้างฟังก์ชันแยกสำหรับแต่ละส่วน)

def get_technical_analysis_data(price_history_df: pd.DataFrame) -> dict:
    # ... (ส่วนอื่น ๆ ของ data_handler.py) ...
    if not isinstance(price_history_df, pd.DataFrame) or price_history_df.empty: return {"error": "Price history data is missing."}
    df = price_history_df.copy()

    # --- [START FIX] Ensure column names are capitalized ---
    df.columns = [str(col).capitalize() for col in df.columns]
    # Check if required capitalized columns exist now
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df.columns]
         logging.error(f"Input DataFrame is missing required columns after capitalization attempt: {missing}")
         # <<< [แก้ไข] ส่งคืน Error ที่ชัดเจนขึ้น
         return {"error": f"Input price data missing required columns: {missing}"}
    # --- [END FIX] ---

    try:
        # Calculate indicators using capitalized column names
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
        # Handle potential division by zero in RSI calculation
        rs = gain / loss.replace(0, np.nan) # Replace 0 loss with NaN to avoid division error
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50) # Fill NaN RSI values (e.g., at the start or if loss is consistently 0)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # --- [แก้ไข] ตรวจสอบว่าคอลัมน์ Indicator ถูกสร้างขึ้นจริง ๆ ก่อน dropna ---
        indicator_cols = ['SMA50', 'SMA200', 'EMA20', 'BB_Upper', 'BB_Lower', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
        existing_indicator_cols = [col for col in indicator_cols if col in df.columns]

        # ถ้าไม่มี Indicator ไหนเลยคำนวณได้ ให้ส่งคืน Error
        if not existing_indicator_cols:
            logging.error("None of the required technical indicator columns could be calculated.")
            return {"error": "Could not calculate any technical indicators."}

        # Drop เฉพาะแถวที่ค่า Indicator ที่มีอยู่จริงเป็น NaN
        return {"data": df.dropna(subset=existing_indicator_cols)}

    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}", exc_info=True)
        # --- [แก้ไข] ส่งคืน Error ที่เฉพาะเจาะจงมากขึ้น ---
        return {"error": f"Calculation error: {type(e).__name__} - {e}"}


@lru_cache(maxsize=32)
def calculate_exit_multiple_valuation(ticker: str, forecast_years: int, eps_growth_rate: float, terminal_pe: float) -> dict:
    # ... (โค้ดเดิม) ...
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return {'error': 'Could not fetch company info from yfinance.'}
        current_price, trailing_eps = info.get('currentPrice') or info.get('previousClose'), info.get('trailingEps')
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


# --- [NEW HELPER FUNCTION FOR DCF BASE DATA FROM DB] ---
# --- [MODIFIED: 2025-11-01 - Added alternative metric names] ---
def _get_dcf_base_data_from_db(ticker: str) -> dict:
    """
    Queries the database (FactFinancialStatements & FactCompanySummary)
    to get the necessary base financial data for DCF analysis, including TTM metrics.
    FCFF = EBIT * (1 - TTM Tax Rate) + D&A - CapEx
    Also calculates WACC (WACC_calc) using Beta and Debt from DB.
    """
    with server.app_context():
        # 1. Find 4 latest unique report dates for the ticker
        # --- [START FIX 2] ---
        # Query TTM metrics (SUM over 4 quarters)
        # We need to look in *only* quarterly statements for TTM
        statement_types_income_cashflow = ['income_q', 'cashflow_q']
        statement_types_balance = ['balance_q'] # Prefer quarterly, but will check annual if needed
        
        latest_reports_query = db.session.query(
            distinct(FactFinancialStatements.report_date).label('report_date')
        ).filter(
            FactFinancialStatements.ticker == ticker,
            FactFinancialStatements.statement_type.in_(statement_types_income_cashflow) # <-- FIXED: Only _q
        ).order_by(
            FactFinancialStatements.report_date.desc()
        ).limit(4).subquery()
        # --- [END FIX 2] ---

        # 2. Query TTM metrics (SUM over 4 quarters)
        # --- [START FIX 1] ---
        required_ttm_metrics = [
            'EBIT', 'Operating Income', # <--- Added alternative
            'Depreciation And Amortization', 'Depreciation & Amortization', 'Depreciation', # <--- Added alternatives
            'Capital Expenditure', 'CapEx', # <--- Added alternative
            'Income Tax Expense',
            'Income Before Tax',
            'Interest Expense'
        ]
        # --- [END FIX 1] ---

        ttm_data = db.session.query(
            FactFinancialStatements.metric_name,
            func.sum(FactFinancialStatements.metric_value).label('ttm_value')
        ).filter(
            FactFinancialStatements.ticker == ticker,
            FactFinancialStatements.report_date.in_(latest_reports_query),
            FactFinancialStatements.metric_name.in_(required_ttm_metrics)
        ).group_by(FactFinancialStatements.metric_name).all()

        # --- [START FIX 1 (Normalization)] ---
        # Create a raw dictionary *without* default values first to check for presence
        ttm_dict_raw = {name: value for name, value in ttm_data}

        # Normalize TTM metrics from alternatives
        ebit_ttm_raw = ttm_dict_raw.get('EBIT', ttm_dict_raw.get('Operating Income'))
        da_ttm_raw = ttm_dict_raw.get('Depreciation And Amortization', ttm_dict_raw.get('Depreciation & Amortization', ttm_dict_raw.get('Depreciation')))
        capex_ttm_raw = ttm_dict_raw.get('Capital Expenditure', ttm_dict_raw.get('CapEx'))

        # Check if any of the *normalized* values are None (meaning *none* of their alternatives were found)
        if not all(v is not None for v in [ebit_ttm_raw, da_ttm_raw, capex_ttm_raw]):
            missing = []
            if ebit_ttm_raw is None: missing.append("EBIT/Operating Income")
            if da_ttm_raw is None: missing.append("D&A")
            if capex_ttm_raw is None: missing.append("CapEx")
            return {'error': f'Missing essential TTM financial data: {", ".join(missing)}. (Querying only _q statements)', 'success': False}
        
        # Now, create the final ttm_dict with defaults for calculation
        ttm_dict = {
            'EBIT': ebit_ttm_raw,
            'Depreciation And Amortization': da_ttm_raw,
            'Capital Expenditure': capex_ttm_raw,
            'Income Tax Expense': ttm_dict_raw.get('Income Tax Expense', 0.0),
            'Income Before Tax': ttm_dict_raw.get('Income Before Tax', 0.0),
            'Interest Expense': ttm_dict_raw.get('Interest Expense', 0.0)
        }
        # --- [END FIX 1 (Normalization)] ---

        # 3. Calculate TTM Tax Rate (Dynamic)
        ttm_tax = ttm_dict.get('Income Tax Expense', 0.0)
        ttm_pretax = ttm_dict.get('Income Before Tax', 0.0)

        # Use Dynamic TTM Rate, fallback to 21%
        tax_rate = (ttm_tax / ttm_pretax) if ttm_pretax and ttm_pretax != 0 else 0.21
        # Ensure tax rate is within a reasonable range (e.g., 0% to 50%)
        tax_rate = max(0.0, min(0.5, tax_rate))

        # 4. Query latest Balance Sheet items (Cash, Debt)
        # --- [START FIX 2] ---
        # Find the latest date from *any* balance sheet (quarterly preferred, annual fallback)
        latest_bs_date = db.session.query(
            func.max(FactFinancialStatements.report_date)
        ).filter(
            FactFinancialStatements.ticker == ticker,
            FactFinancialStatements.statement_type.in_(['balance_q', 'balance_a']) # <-- FIXED
        ).scalar()
        # --- [END FIX 2] ---

        bs_data = db.session.query(
            FactFinancialStatements.metric_name,
            FactFinancialStatements.metric_value
        ).filter(
            FactFinancialStatements.ticker == ticker,
            # --- [START FIX 2] ---
            FactFinancialStatements.statement_type.in_(['balance_q', 'balance_a']), # <-- FIXED
            # --- [END FIX 2] ---
            FactFinancialStatements.report_date == latest_bs_date,
            FactFinancialStatements.metric_name.in_(['Cash And Cash Equivalents', 'Total Debt'])
        ).all()

        bs_dict = {name: (value if value is not None else 0.0) for name, value in bs_data}

        # 5. Query latest price/market cap/beta from FactCompanySummary
        latest_summary = db.session.query(
            FactCompanySummary.price,
            FactCompanySummary.market_cap,
            FactCompanySummary.beta
        ).filter(
            FactCompanySummary.ticker == ticker
        ).order_by(FactCompanySummary.date_updated.desc()).first()

        if not latest_summary or latest_summary.price is None or latest_summary.market_cap is None or latest_summary.price == 0:
            return {'error': 'Missing latest summary data (Price, Market Cap).', 'success': False}

        price = latest_summary.price
        market_cap = latest_summary.market_cap
        beta = latest_summary.beta # <<< GET BETA

        # 6. Calculate Shares Outstanding (Market Cap / Price)
        shares_outstanding = market_cap / price

        if shares_outstanding == 0:
            return {'error': 'Could not calculate Shares Outstanding (Market Cap/Price is zero).', 'success': False}

        # 7. Calculate WACC (Dynamic Logic Restored)
        wacc_calc = None
        total_debt = bs_dict.get('Total Debt', 0.0)
        interest_expense = ttm_dict.get('Interest Expense', 0.0) # TTM Interest Expense
        total_capital = market_cap + total_debt

        if beta is not None and total_debt >= 0 and total_capital > 0:
             # Cost of Debt (Rd) - Uses TTM Interest Expense, falls back to R_f if no debt/interest or interest=0
            cost_of_debt_rd = (interest_expense / total_debt) if total_debt != 0 and interest_expense is not None and interest_expense != 0 else ASSUMED_RISK_FREE_RATE

             # Cost of Equity (Re)
            cost_of_equity_re = ASSUMED_RISK_FREE_RATE + beta * (ASSUMED_MARKET_RETURN - ASSUMED_RISK_FREE_RATE)

             # WACC Formula
            wacc_calc = (
                (market_cap / total_capital) * cost_of_equity_re
            ) + (
                (total_debt / total_capital) * cost_of_debt_rd * (1 - tax_rate)
            )

        # 8. Final Data Dict & Base FCFF Calculation
        dcf_base = {
            'success': True,
            'current_price': price,
            'shares_outstanding': shares_outstanding,
            # --- [START FIX 1 (Usage)] ---
            'ebit_ttm': ttm_dict['EBIT'], # This now holds the normalized value
            'tax_rate': tax_rate, # <<< Dynamic TTM Rate
            'd_and_a_ttm': ttm_dict['Depreciation And Amortization'], # This now holds the normalized value
            'capex_ttm': ttm_dict['Capital Expenditure'], # This now holds the normalized value
            # --- [END FIX 1 (Usage)] ---
            'cash': bs_dict.get('Cash And Cash Equivalents', 0.0),
            'total_debt': total_debt,
            'wacc_calculated': wacc_calc # <<< ADDED (for optional display/info)
        }

        dcf_base['net_debt'] = dcf_base['total_debt'] - bs_dict.get('Cash And Cash Equivalents', 0.0)
        dcf_base['base_fcff'] = (dcf_base['ebit_ttm'] * (1 - dcf_base['tax_rate'])) + dcf_base['d_and_a_ttm'] - dcf_base['capex_ttm']

        if dcf_base['base_fcff'] == 0:
             return {'error': 'Base FCFF is zero after calculation. Cannot run simulation.', 'success': False}

        return dcf_base
# --- [END NEW HELPER FUNCTION] ---


# ==============================================================================
# --- [NEW V-FINAL] SECTION: ML Risk Model V2.0 Data Fetching ---
# ==============================================================================

def _get_ml_risk_raw_data_base(tickers: Optional[List[str]] = None, start_year: int = 2010) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (V-FINAL)
    ดึงข้อมูลดิบ แต่ *ไม่* ทำการ merge_asof
    จะ return (df_merged_final, df_prices_final) ออกมาแทน
    """
    logging.info("Starting to fetch raw data (base) for ML Risk Model V2.0...")
    
    start_date = datetime(start_year, 1, 1).date() 
    
    with server.app_context():
        
        # ( ... โค้ดส่วนดึงข้อมูลจาก DB ทั้งหมดเหมือนเดิม ... )
        # --- [FIX 1: เพิ่ม Metrics ทั้งหมดที่ X และ Y ต้องการ] ---
        required_fin_metrics_alternatives = [
            'Accumulated Deficit', 'Accumulated_Deficit', 'Retained Earnings',
            'Current Ratio', 'Current_Ratio',
            'Total Equity', 'Stockholders Equity', 'Total Stockholders Equity',
            'Current Assets',
            'Current Liabilities',
            'Total Assets',
            'Inventory',
            'Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest',
            'EBIT', 'Operating Income',
            'Total Revenue', 'Sales', 'Revenues',
            'Net Income', 'Net Income Common Stockholders',
            'Operating Cash Flow', 'Total Cash Flow From Operating Activities',
            'Interest Expense',
            'Accounts Receivable',
            'Accounts Payable',
            'Cost Of Revenue', 'Cost Of Goods Sold',
        ]
        
        fin_query = db.session.query(
            FactFinancialStatements.ticker,
            FactFinancialStatements.report_date,
            FactFinancialStatements.metric_name,
            FactFinancialStatements.metric_value
        ).filter(
            FactFinancialStatements.metric_name.in_(required_fin_metrics_alternatives),
            FactFinancialStatements.statement_type.in_(['income_a', 'income_q', 'balance_a', 'balance_q', 'cashflow_a', 'cashflow_q']),
            FactFinancialStatements.report_date >= start_date
        )
        
        if tickers:
            fin_query = fin_query.filter(FactFinancialStatements.ticker.in_(tickers))

        df_financials_long = pd.read_sql(fin_query.statement, db.engine)
        
        if df_financials_long.empty:
            logging.warning("No financial statement data found for required metrics.")
            return pd.DataFrame(), pd.DataFrame()

        # --- [FIX 2: สร้าง Normalize Function ที่สมบูรณ์] ---
        def normalize_metric_name(name):
            if name in ['Accumulated Deficit', 'Accumulated_Deficit', 'Retained Earnings']:
                return 'Accumulated Deficit' 
            if name in ['Current Ratio', 'Current_Ratio']:
                return 'Current Ratio'
            if name in ['Total Equity', 'Stockholders Equity', 'Total Stockholders Equity']:
                return 'Stockholders Equity'
            if name in ['Current Assets']: return 'Current Assets'
            if name in ['Current Liabilities']: return 'Current Liabilities'
            if name in ['Total Assets']: return 'Total Assets'
            if name in ['Inventory']: return 'Inventory'
            if name in ['Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest']:
                return 'Total Liabilities'
            if name in ['EBIT', 'Operating Income']:
                return 'EBIT'
            if name in ['Total Revenue', 'Sales', 'Revenues']:
                return 'Total Revenue'
            if name in ['Net Income', 'Net Income Common Stockholders']:
                return 'Net Income'
            if name in ['Operating Cash Flow', 'Total Cash Flow From Operating Activities']:
                return 'Operating Cash Flow'
            if name in ['Interest Expense']: return 'Interest Expense'
            if name in ['Accounts Receivable']: return 'Accounts Receivable'
            if name in ['Accounts Payable']: return 'Accounts Payable'
            if name in ['Cost Of Revenue', 'Cost Of Goods Sold']:
                return 'Cost Of Revenue'
            return name
        
        df_financials_long['metric_name'] = df_financials_long['metric_name'].apply(normalize_metric_name)
        
        df_financials_long = df_financials_long.sort_values(by='report_date', ascending=True)
        df_financials_long = df_financials_long.drop_duplicates(subset=['ticker', 'report_date', 'metric_name'], keep='last')
        
        df_financials_wide = df_financials_long.pivot_table(
            index=['ticker', 'report_date'], 
            columns='metric_name', 
            values='metric_value'
        ).reset_index()
        
        df_financials_wide.columns.name = None
        
        # --- [FIX 1 (NEW): แก้ Warning 'Current Ratio' missing] ---
        if 'Current Ratio' not in df_financials_wide.columns:
            if 'Current Assets' in df_financials_wide.columns and 'Current Liabilities' in df_financials_wide.columns:
                logging.info("Manually calculating 'Current Ratio' in data_handler...")
                denom = df_financials_wide['Current Liabilities'].replace(0, np.nan)
                df_financials_wide['Current Ratio'] = df_financials_wide['Current Assets'] / denom
            else:
                logging.warning("Cannot calculate 'Current Ratio': Missing 'Current Assets' or 'Current Liabilities'. Filling with NaN.")
                df_financials_wide['Current Ratio'] = np.nan
        
        # 2. Fetch Credit Rating
        rating_query = db.session.query(DimCompany.ticker, DimCompany.credit_rating)
        if tickers:
            rating_query = rating_query.filter(DimCompany.ticker.in_(tickers))
        df_rating = pd.read_sql(rating_query.statement, db.engine)
        df_merged = pd.merge(df_financials_wide, df_rating, on='ticker', how='left')

        # 3. Fetch Market Data (Prices)
        # --- [START BATCH FIX - 2025-11-02] ---
        logging.info("  Fetching prices data (in yearly batches to prevent timeout)...")
        
        current_year = datetime.now().year
        all_price_dfs = [] # List สำหรับเก็บ DataFrame ของแต่ละปี
        
        # Loop from start_year (e.g., 2010) to the current year
        for year in range(start_year, current_year + 1):
            logging.info(f"    Fetching prices for year {year}...")
            
            year_start_date = datetime(year, 1, 1).date()
            year_end_date = datetime(year, 12, 31).date()

            price_query_year = db.session.query(
                FactDailyPrices.ticker,
                FactDailyPrices.date.label('price_date'), 
                FactDailyPrices.close.label('closing_price')
            ).filter(
                FactDailyPrices.date >= year_start_date,
                FactDailyPrices.date <= year_end_date  # <-- กรองทีละปี
            )
            
            # (สำคัญ) เรายังคง filter 'tickers' ถ้ามีการระบุมา
            if tickers:
                price_query_year = price_query_year.filter(FactDailyPrices.ticker.in_(tickers))

            # ดึงข้อมูลเฉพาะของปีนี้ (Query นี้จะเร็ว)
            df_prices_year = pd.read_sql(price_query_year.statement, db.engine)
            
            if not df_prices_year.empty:
                all_price_dfs.append(df_prices_year)
        
        # เมื่อ Loop ครบทุกปี, นำ DataFrame มารวมกัน
        if not all_price_dfs:
            logging.warning("  No price data found after batch fetching.")
            df_prices = pd.DataFrame(columns=['ticker', 'price_date', 'closing_price'])
        else:
            df_prices = pd.concat(all_price_dfs, ignore_index=True)
        
        logging.info(f"  Fetched {len(df_prices)} total price records from {start_year} to {current_year}.")
        # --- [END BATCH FIX - 2025-11-02] ---
        
        # 4. Fetch Market Cap
        summary_subquery = db.session.query(
            FactCompanySummary.ticker, func.max(FactCompanySummary.date_updated).label('max_date')
        ).group_by(FactCompanySummary.ticker).subquery()
        summary_query = db.session.query(
            FactCompanySummary.ticker, FactCompanySummary.price, FactCompanySummary.market_cap 
        ).join(
            summary_subquery,
            (FactCompanySummary.ticker == summary_subquery.c.ticker) & 
            (FactCompanySummary.date_updated == summary_subquery.c.max_date)
        )
        if tickers:
            summary_query = summary_query.filter(FactCompanySummary.ticker.in_(tickers))
        df_summary = pd.read_sql(summary_query.statement, db.engine)
        
        df_summary['shares_outstanding'] = df_summary['market_cap'] / df_summary['price'].replace(0, np.nan)
        df_summary.drop(columns=['price'], inplace=True) 
        df_merged = pd.merge(df_merged, df_summary[['ticker', 'market_cap', 'shares_outstanding']], on='ticker', how='left')
        
        # --- Post-Processing in Pandas for Price Matching ---
        
        df_merged.rename(columns={'report_date': 'date'}, inplace=True)
        df_prices.rename(columns={'price_date': 'date'}, inplace=True)

        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')
        df_prices['date'] = pd.to_datetime(df_prices['date'], errors='coerce')

        logging.info("Forcing 'ticker' columns to string type before processing.")
        df_merged['ticker'] = df_merged['ticker'].astype(str)
        df_prices['ticker'] = df_prices['ticker'].astype(str)
        
        df_merged = df_merged.dropna(subset=['date', 'ticker']) 
        df_prices = df_prices.dropna(subset=['date', 'ticker'])

        if df_merged.empty or df_prices.empty:
            logging.warning("No data left in financials or prices after date conversion/cleaning. Cannot merge.")
            return pd.DataFrame(), pd.DataFrame()

        # --- [START HOTFIX (V-FINAL Base): Simplified Sorting] ---
        
        logging.info("Dropping duplicates and forcing sort (V-FINAL) on left DataFrame (df_merged)...")
        df_merged = df_merged.drop_duplicates(subset=['ticker', 'date'], keep='last')
        df_merged = df_merged.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        logging.info("Dropping duplicates and forcing sort (V-FINAL) on right DataFrame (df_prices)...")
        df_prices = df_prices.drop_duplicates(subset=['ticker', 'date'], keep='last')
        df_prices = df_prices.sort_values(by=['ticker', 'date']).reset_index(drop=True)

        logging.info("Creating final (left) df_merged_final copy...")
        df_merged_final = df_merged.copy() 

        logging.info("Creating final (right) df_prices_final subset and copy...")
        df_prices_final = df_prices[['ticker', 'date', 'closing_price']].copy()
        # --- [END HOTFIX] ---
        
        logging.info("Base data fetch complete. Returning DFs for manual merge...")
        return df_merged_final, df_prices_final


def _merge_asof_by_group(df_merged: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    (V-FINAL) Performs pd.merge_asof by iterating through each 'ticker' group.
    This version keeps 'ticker' as a column throughout and is proven to work (V8).
    """
    logging.info("--- STARTING MERGE (V-FINAL - Groupby-Apply, Column-based) ---")
    
    if df_merged.empty or df_prices.empty:
        logging.warning("Empty input DFs, cannot merge.")
        return pd.DataFrame()

    # สร้าง GroupBy objects
    # (เรามั่นใจว่ามัน Sort มาแล้วจาก _get_ml_risk_raw_data_base)
    left_groups = df_merged.groupby('ticker')
    right_groups = df_prices.groupby('ticker')
    
    all_merged_groups = []
    
    logging.info(f"Found {len(left_groups)} unique tickers in LEFT frame to process...")
    processed_count = 0
    
    # วนลูปผ่านกลุ่ม Ticker ฝั่งซ้าย
    for ticker, left_group in left_groups:
        try:
            # พยายามดึงกลุ่ม Ticker ฝั่งขวา
            right_group = right_groups.get_group(ticker)
            
            # ทำ merge_asof *เฉพาะกลุ่มนี้*
            merged_group = pd.merge_asof(
                left_group,
                right_group,
                on='date',
                by='ticker',   # <-- **[V8 FIX]** เพิ่ม by='ticker' เข้าไปใน merge_asof
                direction='backward'
            )
            all_merged_groups.append(merged_group)
            
        except KeyError:
            # Ticker นี้ไม่มีข้อมูลราคา (ฝั่งขวา)
            left_group_with_nan = left_group.copy()
            left_group_with_nan['closing_price'] = np.nan
            all_merged_groups.append(left_group_with_nan)
            
        except Exception as e:
            logging.error(f"  ❌ FAILED to merge group for ticker: {ticker}. Error: {e}")
            continue
            
        processed_count += 1
        if processed_count % 500 == 0:
            logging.info(f"  Processed {processed_count}/{len(left_groups)} tickers...")

    logging.info(f"Finished processing all {len(left_groups)} tickers.")
    
    if not all_merged_groups:
        logging.warning("No groups were successfully merged.")
        return pd.DataFrame()
        
    # รวมทุกกลุ่มที่ merge เสร็จแล้วกลับเป็น DataFrame เดียว
    logging.info("Concatenating all merged groups...")
    df_final = pd.concat(all_merged_groups, ignore_index=True) 
    
    logging.info("--- MERGE (V-FINAL - Groupby-Apply) COMPLETE ---")
    return df_final


def get_ml_risk_raw_data(tickers: Optional[List[str]] = None, start_year: int = 2010) -> pd.DataFrame:
    """
    (V-FINAL) Fetches and processes the raw data for the ML Risk Model.
    This function now uses a robust Groupby-Apply merge method (from V8) 
    to bypass the pandas merge_asof sorting bug.
    """
    try:
        # 1. ดึงข้อมูลดิบ (ยังไม่ Merge)
        df_merged_final, df_prices_final = _get_ml_risk_raw_data_base(start_year=start_year)
        
        # 2. รันฟังก์ชัน Merge (V-FINAL/V8) ที่แยกออกมา
        df_final = _merge_asof_by_group(df_merged_final, df_prices_final)
        
        if df_final.empty:
            logging.warning("merge_asof (by group) resulted in an empty DataFrame.")
            return pd.DataFrame()

        # 3. ทำ Post-processing ที่เหลือ (ที่เคยอยู่ในฟังก์ชันเดิม)
        df_final.rename(columns={'date': 'report_date'}, inplace=True)
        
        if 'ticker' not in df_final.columns:
             logging.error("--- 💔 CRITICAL ERROR (V-FINAL) ---")
             logging.error("'ticker' column was lost during groupby-merge!")
             return pd.DataFrame()
        
        df_final.set_index(['ticker', 'report_date'], inplace=True)
        df_final.sort_index(inplace=True)
        
        logging.info(f"Finished fetching raw data (V-FINAL). Final Shape: {df_final.shape}")
        return df_final
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during get_ml_risk_raw_data (V-FINAL): {e}", exc_info=True)
        return pd.DataFrame()

# --- [MODIFIED] calculate_monte_carlo_dcf ---
@lru_cache(maxsize=32)
def calculate_monte_carlo_dcf(
    ticker: str,
    n_simulations: int,
    growth_min: float, growth_mode: float, growth_max: float,
    perpetual_min: float, perpetual_mode: float, perpetual_max: float,
    wacc_min: float, wacc_mode: float, wacc_max: float
) -> dict:
    """
    (MODIFIED: Fetches base data from DB for DCF calculation, WACC range still used for simulation)
    """
    try:
        # --- [MODIFICATION: Fetch base data from DB] ---
        dcf_data = _get_dcf_base_data_from_db( ticker)

        if not dcf_data.get('success'):
            # Propagate detailed error message
            return {'error': dcf_data.get('error', 'Missing essential data for DCF from DB.')}

        # Unpack base data
        base_fcff = dcf_data['base_fcff']
        net_debt = dcf_data['net_debt']
        shares_outstanding = dcf_data['shares_outstanding']
        current_price = dcf_data['current_price']
        # --- [END MODIFICATION] ---

        PROJECTION_YEARS = 5

        # --- Monte Carlo Simulation Logic (เหมือนเดิม) ---
        sim_growth = np.random.triangular(growth_min / 100.0, growth_mode / 100.0, growth_max / 100.0, n_simulations)
        sim_perpetual = np.random.triangular(perpetual_min / 100.0, perpetual_mode / 100.0, perpetual_max / 100.0, n_simulations)
        sim_wacc = np.random.triangular(wacc_min / 100.0, wacc_mode / 100.0, wacc_max / 100.0, n_simulations)

        # Handle cases where sim_wacc might be <= sim_perpetual
        valid_sim = sim_wacc > sim_perpetual
        if not np.any(valid_sim):
            return {'error': 'Invalid WACC/Perpetual Growth ranges resulting in WACC <= g for all simulations.'}

        # Filter out invalid simulations before calculations
        sim_growth = sim_growth[valid_sim]
        sim_perpetual = sim_perpetual[valid_sim]
        sim_wacc = sim_wacc[valid_sim]

        if len(sim_wacc) == 0:
             return {'error': 'No valid simulations after filtering WACC > g.'}

        n_valid_simulations = len(sim_wacc)

        # DCF Calculation
        future_fcffs = base_fcff * (1 + sim_growth) ** np.arange(1, PROJECTION_YEARS + 1)[:, np.newaxis]
        discounted_fcffs = future_fcffs / (1 + sim_wacc) ** np.arange(1, PROJECTION_YEARS + 1)[:, np.newaxis]
        final_year_fcff = future_fcffs[-1]

        terminal_value = (final_year_fcff * (1 + sim_perpetual)) / (sim_wacc - sim_perpetual)
        terminal_value[terminal_value < 0] = 0
        discounted_terminal_value = terminal_value / ((1 + sim_wacc) ** PROJECTION_YEARS)

        enterprise_value = np.sum(discounted_fcffs, axis=0) + discounted_terminal_value
        equity_value = enterprise_value - net_debt
        simulated_intrinsic_values = equity_value / shares_outstanding

        finite_values = simulated_intrinsic_values[np.isfinite(simulated_intrinsic_values)]
        if len(finite_values) == 0:
             return {'error': 'All simulations resulted in non-finite intrinsic values.'}

        return {
            'simulated_values': finite_values.tolist(),
            'current_price': current_price,
            'mean': np.mean(finite_values),
            'median': np.median(finite_values),
            'p10': np.percentile(finite_values, 10),
            'p90': np.percentile(finite_values, 90),
            'success': True,
            'num_valid_simulations': len(finite_values)
        }

    except Exception as e:
        logging.error(f"Monte Carlo DCF failed for {ticker}: {e}", exc_info=True)
        return {'error': str(e)}
# --- [END MODIFIED] calculate_monte_carlo_dcf ---


# ==============================================================================
# --- [NEW] SECTION: Fallback Data Fetching Functions ---
# ==============================================================================

@lru_cache(maxsize=128) # Cache results to avoid repeated API calls
def get_deep_dive_header_data(ticker: str) -> dict:
    """
    Fetches header data for the deep dive page.
    Tries database first, falls back to live yfinance fetch.
    Returns a dictionary containing header info or an error message.
    """
    logging.info(f"[Fallback] Trying to fetch header data for {ticker} from DB...")
    header_data = {'source': 'database', 'error': None}
    try:
        with server.app_context():
            # Query DB for latest DimCompany and FactCompanySummary
            db_result = db.session.query(
                DimCompany.company_name, DimCompany.logo_url, DimCompany.sector,
                FactCompanySummary.price, FactCompanySummary.market_cap,
                FactCompanySummary.analyst_target_price, FactCompanySummary.pe_ratio,
                FactCompanySummary.forward_pe, FactCompanySummary.long_business_summary,
                FactCompanySummary.beta,
                FactCompanySummary.peer_cluster_id,
                FactCompanySummary.predicted_default_prob
            ).outerjoin( # Use outerjoin in case DimCompany exists but summary doesn't yet
                FactCompanySummary,
                (DimCompany.ticker == FactCompanySummary.ticker) &
                (FactCompanySummary.date_updated == db.session.query(func.max(FactCompanySummary.date_updated)).filter(FactCompanySummary.ticker == ticker).scalar_subquery())
            ).filter(
                DimCompany.ticker == ticker
            ).first()

            # Query DB for last two closing prices for daily change calculation
            price_history_db = db.session.query(FactDailyPrices.close) \
                                        .filter(FactDailyPrices.ticker == ticker) \
                                        .order_by(FactDailyPrices.date.desc()) \
                                        .limit(2).all()

        if db_result:
            logging.info(f"[Fallback] Found header data for {ticker} in DB.")
            current_price = db_result.price
            if len(price_history_db) >= 2:
                previous_close = price_history_db[1][0]
            elif len(price_history_db) == 1:
                previous_close = price_history_db[0][0]
            elif current_price is not None: # Fallback if only summary price exists
                previous_close = current_price
            else: # Cannot determine previous close
                 previous_close = None
                 logging.warning(f"[Fallback] Could not determine previous close for {ticker} from DB.")


            # Populate dict from DB result
            header_data.update({
                'company_name': db_result.company_name,
                'logo_url': db_result.logo_url,
                'sector': db_result.sector,
                'current_price': current_price,
                'previous_close': previous_close, # Needed for daily change calculation
                'market_cap': db_result.market_cap,
                'analyst_target_price': db_result.analyst_target_price,
                'pe_ratio': db_result.pe_ratio,
                'forward_pe': db_result.forward_pe,
                'long_business_summary': db_result.long_business_summary,
                'beta': db_result.beta,
                'peer_cluster_id': db_result.peer_cluster_id,
                'predicted_default_prob': db_result.predicted_default_prob,
                # Add other necessary fields if needed
            })
            # Check if essential data like price is present
            if header_data.get('current_price') is not None:
                return header_data # Return data from DB
            else:
                 logging.warning(f"[Fallback] Data found for {ticker} in DB, but essential info (like price) is missing. Will try live fetch.")

        else:
             logging.info(f"[Fallback] Header data for {ticker} not found in DB. Trying live fetch...")

        # --- Fallback to Live Fetch ---
        header_data['source'] = 'live'
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info
            hist = tkr.history(period="2d") # Get last 2 days for price change

            if not info or info.get('quoteType') != 'EQUITY':
                header_data['error'] = "Invalid ticker or no data available from yfinance."
                return header_data

            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            previous_close = None
            if not hist.empty and len(hist) > 0:
                 # Try getting 'previousClose' from info first, then from history
                 previous_close = info.get('previousClose')
                 if previous_close is None and len(hist) >= 1 :
                      # If only one day in history use its close, otherwise use the second to last day
                      previous_close_index = -1 if len(hist) == 1 else -2
                      previous_close = hist['Close'].iloc[previous_close_index]


            # Get logo using the helper function
            logo_url = _get_logo_url(info)

            header_data.update({
                'company_name': info.get('longName'),
                'logo_url': logo_url,
                'sector': info.get('sector'),
                'current_price': current_price,
                'previous_close': previous_close,
                'market_cap': info.get('marketCap'),
                'analyst_target_price': info.get('targetMeanPrice'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'long_business_summary': info.get('longBusinessSummary'),
                'beta': info.get('beta'),
                'predicted_default_prob': None,
                # Map other yfinance info keys as needed
            })
            logging.info(f"[Fallback] Successfully fetched live header data for {ticker}.")
            # Ensure essential data exists after live fetch
            if header_data.get('current_price') is None:
                 header_data['error'] = "Live fetch successful but essential data (price) is still missing."


        except Exception as e:
            logging.error(f"[Fallback] Live fetch failed for {ticker}: {e}", exc_info=True)
            header_data['error'] = f"Live yfinance fetch failed: {e}"

        return header_data

    except Exception as db_e:
        logging.error(f"[Fallback] Database query failed for {ticker} header: {db_e}", exc_info=True)
        header_data['error'] = f"Database query failed: {db_e}"
        # Optionally, you could still attempt a live fetch here if the DB query itself fails
        # but for simplicity, we return the DB error first.
        return header_data

# --- [NEW] Helper: Convert period string to start date ---
def _period_to_start_date(period: str) -> datetime.date:
    """Converts yfinance period string (e.g., '5y', '1y') to a start date."""
    today = datetime.utcnow().date()
    if period == '5y':
        return today - timedelta(days=5*365)
    elif period == '1y':
        return today - timedelta(days=365)
    elif period == 'ytd':
         return datetime(today.year, 1, 1).date()
    # Add more periods as needed
    else: # Default to 5 years if unrecognized
        logging.warning(f"Unrecognized period '{period}'. Defaulting to 5 years.")
        return today - timedelta(days=5*365)


@lru_cache(maxsize=64)
def get_historical_prices(ticker: str, period: str = "5y") -> Tuple[pd.DataFrame, str]:
    """
    Fetches historical OHLCV data for the deep dive technical chart.
    Tries database first, falls back to live yfinance fetch.
    Returns a tuple: (DataFrame, source_string)
    """
    logging.info(f"[Fallback] Trying to fetch '{period}' price history for {ticker} from DB...")
    source = 'database'
    start_date = _period_to_start_date(period)
    end_date = datetime.utcnow().date() # Fetch up to today

    try:
        with server.app_context():
            query = db.session.query(
                FactDailyPrices.date, FactDailyPrices.open, FactDailyPrices.high,
                FactDailyPrices.low, FactDailyPrices.close, FactDailyPrices.volume
            ).filter(
                FactDailyPrices.ticker == ticker,
                FactDailyPrices.date >= start_date
            ).order_by(FactDailyPrices.date.asc())

            # Fetch results using pandas read_sql for efficiency
            df_prices = pd.read_sql(query.statement, db.engine, index_col='date')

        if not df_prices.empty:
            # --- Check data recency ---
            last_date_in_db = df_prices.index.max()
            days_diff = (end_date - last_date_in_db).days

            # If data is reasonably recent (e.g., within 3 days, accounting for weekends), use DB data
            if days_diff <= 3:
                logging.info(f"[Fallback] Found recent enough price history for {ticker} in DB (Last: {last_date_in_db}).")
                df_prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                return df_prices, source
            else:
                logging.warning(f"[Fallback] Price history for {ticker} in DB is outdated (Last: {last_date_in_db.date()}). Will try live fetch.")

        else:
             logging.info(f"[Fallback] Price history for {ticker} not found in DB for period '{period}'. Trying live fetch...")

        # --- Fallback to Live Fetch ---
        source = 'live'
        try:
            # Use yfinance download
            df_prices = yf.download([ticker], start=start_date, end=end_date, auto_adjust=True, progress=False)

            if df_prices.empty:
                logging.warning(f"[Fallback] Live fetch (yf.download) returned empty for {ticker} and period '{period}'.")
                return pd.DataFrame(), source # Return empty DataFrame

            # --- [START FIX] Handle potential MultiIndex for single ticker ---
            if isinstance(df_prices.columns, pd.MultiIndex):
                logging.info(f"[Fallback] Flattening MultiIndex columns for {ticker}.")
                # Flatten MultiIndex (e.g., ('Close', '')) -> 'Close'
                df_prices.columns = df_prices.columns.get_level_values(0)
            # --- [END FIX] ---

            # Rename columns if needed (yf.download might return slightly different names)
            df_prices.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True, errors='ignore')

            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_prices.columns for col in required_cols):
                 logging.error(f"[Fallback] Live fetched data for {ticker} is missing required OHLCV columns.")
                 return pd.DataFrame(), source


            logging.info(f"[Fallback] Successfully fetched live price history for {ticker}.")
            return df_prices, source

        except Exception as e:
            logging.error(f"[Fallback] Live price fetch failed for {ticker}: {e}", exc_info=True)
            return pd.DataFrame({"error": f"Live yfinance fetch failed: {e}"}), source # Return DataFrame with error


    except Exception as db_e:
        logging.error(f"[Fallback] Database query failed for {ticker} prices: {db_e}", exc_info=True)
        # Attempt live fetch even if DB query fails
        logging.info("[Fallback] DB query failed, attempting live fetch as fallback...")
        source = 'live_fallback_db_error'
        try:
            df_prices = yf.download([ticker], start=start_date, end=end_date, auto_adjust=True, progress=False)
            if df_prices.empty:
                logging.warning(f"[Fallback] Live fetch (after DB error) returned empty for {ticker}.")
                return pd.DataFrame({"error": "DB query failed and live fetch returned no data."}), source
            # --- [START FIX] Handle potential MultiIndex for single ticker ---
            if isinstance(df_prices.columns, pd.MultiIndex):
                logging.info(f"[Fallback] Flattening MultiIndex columns for {ticker} (after DB error).")
                df_prices.columns = df_prices.columns.get_level_values(0)
            # --- [END FIX] ---
            df_prices.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True, errors='ignore')
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_prices.columns for col in required_cols):
                 logging.error(f"[Fallback] Live fetched data (after DB error) for {ticker} is missing required OHLCV columns.")
                 return pd.DataFrame({"error": "DB query failed and live fetch data is incomplete."}), source
            logging.info(f"[Fallback] Successfully fetched live price history for {ticker} after DB error.")
            return df_prices, source
        except Exception as live_e:
             logging.error(f"[Fallback] Live price fetch failed for {ticker} after DB error: {live_e}", exc_info=True)
             return pd.DataFrame({"error": f"DB query failed ({db_e}) AND live yfinance fetch failed ({live_e})."}), source

@lru_cache(maxsize=128)
def get_quarterly_financials(ticker: str, statement_type: str) -> Tuple[pd.DataFrame, str]:
    """
    Fetches quarterly financial statement data (Income, Balance Sheet, Cash Flow).
    Tries database first, falls back to live yfinance fetch.
    Returns a tuple: (DataFrame in WIDE format, source_string)
    """
    logging.info(f"[Fallback] Trying to fetch '{statement_type}' statement for {ticker} from DB...")
    source = 'database'
    valid_statement_types = ['income', 'balance', 'cashflow']
    if statement_type not in valid_statement_types:
        logging.error(f"Invalid statement type requested: {statement_type}")
        return pd.DataFrame({"error": f"Invalid statement type: {statement_type}"}), 'error'

    try:
        with server.app_context():
            # Query the long-format data from DB
            query = db.session.query(
                FactFinancialStatements.report_date,
                FactFinancialStatements.metric_name,
                FactFinancialStatements.metric_value
            ).filter(
                FactFinancialStatements.ticker == ticker,
                FactFinancialStatements.statement_type == statement_type
            ).order_by(
                FactFinancialStatements.report_date.desc()
            ) # Fetch all available quarterly data

            df_long = pd.read_sql(query.statement, db.engine)

        if not df_long.empty:
            logging.info(f"[Fallback] Found '{statement_type}' data for {ticker} in DB.")
            # Pivot to wide format for display consistency
            df_wide = df_long.pivot(index='metric_name', columns='report_date', values='metric_value')
            # Sort columns (dates) from newest to oldest
            df_wide = df_wide[sorted(df_wide.columns, reverse=True)]
            # Limit to latest 16 quarters if more exist
            df_wide = df_wide.iloc[:, :16]
            return df_wide, source

        else:
             logging.info(f"[Fallback] '{statement_type}' data for {ticker} not found in DB. Trying live fetch...")

        # --- Fallback to Live Fetch ---
        source = 'live'
        try:
            tkr = yf.Ticker(ticker)
            df_live = None
            if statement_type == 'income':
                df_live = tkr.quarterly_financials
            elif statement_type == 'balance':
                df_live = tkr.quarterly_balance_sheet
            elif statement_type == 'cashflow':
                df_live = tkr.quarterly_cashflow

            if df_live is None or df_live.empty:
                logging.warning(f"[Fallback] Live fetch (yfinance) returned empty for {ticker} '{statement_type}'.")
                return pd.DataFrame(), source # Return empty DataFrame

            # Limit to latest 16 quarters from live data
            df_live = df_live.iloc[:, :16]

            logging.info(f"[Fallback] Successfully fetched live '{statement_type}' data for {ticker}.")
            return df_live, source

        except Exception as e:
            logging.error(f"Live financial statement fetch failed for {ticker} '{statement_type}': {e}", exc_info=True)
            return pd.DataFrame({"error": f"Live yfinance fetch failed: {e}"}), source

    except Exception as db_e:
        logging.error(f"[Fallback] Database query failed for {ticker} '{statement_type}': {db_e}", exc_info=True)
         # Attempt live fetch even if DB query fails
        logging.info(f"[Fallback] DB query failed for {statement_type}, attempting live fetch...")
        source = 'live_fallback_db_error'
        try:
            tkr = yf.Ticker(ticker)
            df_live = None
            if statement_type == 'income': df_live = tkr.quarterly_financials
            elif statement_type == 'balance': df_live = tkr.quarterly_balance_sheet
            elif statement_type == 'cashflow': df_live = tkr.quarterly_cashflow

            if df_live is None or df_live.empty:
                logging.warning(f"[Fallback] Live fetch (after DB error) returned empty for {ticker} '{statement_type}'.")
                return pd.DataFrame({"error": "DB query failed and live fetch returned no data."}), source

            df_live = df_live.iloc[:, :16]
            logging.info(f"[Fallback] Successfully fetched live '{statement_type}' data for {ticker} after DB error.")
            return df_live, source
        except Exception as live_e:
             logging.error(f"[Fallback] Live financial fetch failed for {ticker} '{statement_type}' after DB error: {live_e}", exc_info=True)
             return pd.DataFrame({"error": f"DB query failed ({db_e}) AND live yfinance fetch failed ({live_e})."}), source
        
# --- [NEW] Growth Fetching Helper ---
@lru_cache(maxsize=10) 
def get_yoy_growth_data(tickers: tuple) -> pd.DataFrame:
    """
    Fetches the latest Revenue Growth (YoY) from the database for sorting.
    """
    all_data = []
    total = len(tickers)
    logging.info(f"Starting lightweight growth fetch for {total} tickers from DB...")

    try:
        with server.app_context():
            # 1. Find the latest date for each ticker in FactCompanySummary
            latest_date_sq = db.session.query(
                FactCompanySummary.ticker,
                func.max(FactCompanySummary.date_updated).label('max_date')
            ).filter(
                FactCompanySummary.ticker.in_(tickers)
            ).group_by(FactCompanySummary.ticker).subquery()

            # 2. Query the latest data for Ticker and revenue_growth_yoy
            query = db.session.query(
                FactCompanySummary.ticker,
                FactCompanySummary.revenue_growth_yoy
            ).join(
                latest_date_sq,
                (FactCompanySummary.ticker == latest_date_sq.c.ticker) &
                (FactCompanySummary.date_updated == latest_date_sq.c.max_date)
            ).filter(
                FactCompanySummary.revenue_growth_yoy.isnot(None) # Filter out None values
            )

            df = pd.read_sql(query.statement, db.engine)
            df.rename(columns={'ticker': 'Ticker', 'revenue_growth_yoy': 'Revenue Growth (YoY)'}, inplace=True)
            
            logging.info(f"Finished growth fetch. Got data for {len(df)} out of {total} tickers.")
            return df
    except Exception as e:
        logging.error(f"An error occurred processing growth data from DB: {e}", exc_info=True)
        return pd.DataFrame()
# --- [END NEW] ---