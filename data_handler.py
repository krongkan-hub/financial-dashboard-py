# data_handler.py (FIXED - Rate Limiting & Optimized)

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
import time # Import time for delays

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from newsapi import NewsApiClient

# --- Initialize NewsAPI (เหมือนเดิม) ---
if Config.NEWS_API_KEY:
    newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
else:
    newsapi = None
    logging.warning("NEWS_API_KEY not found in config. News fetching will be disabled.")

# --- Hugging Face API Function (เหมือนเดิม) ---
def analyze_sentiment_batch(texts_to_analyze: List[str]) -> List[dict]:
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

# --- Core Data Fetching Functions ---
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

# --- [MODIFIED] get_competitor_data (added time.sleep, company_name, sector) ---
@lru_cache(maxsize=10)
def get_competitor_data(tickers: tuple) -> pd.DataFrame:
    """
    ดึงข้อมูลสรุปสำหรับ ETL Job 1
    เพิ่ม: Company Name, Sector และ Time Delay
    """
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

            logo_url = _get_logo_url(info)
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
            # Optional: Add a longer sleep if a specific error (like rate limit) occurs
            # if "RateLimitException" in str(e): # Example check (adjust based on actual error)
            #    logging.warning(f"Rate limit hit for {ticker}. Sleeping for 60 seconds...")
            #    time.sleep(60)
            time.sleep(1) # General short sleep on error before trying next ticker
            continue # Continue to the next ticker

    logging.info(f"Finished fetching competitor data. Got data for {len(all_data)} out of {total} tickers.")
    return pd.DataFrame(all_data)
# --- [END OF MODIFICATION] ---


@lru_cache(maxsize=10)
def get_scatter_data(tickers: tuple) -> pd.DataFrame:
    # (ฟังก์ชันนี้ไม่จำเป็นต้องแก้ เพราะจะถูกเลิกใช้ใน callbacks.py)
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

@lru_cache(maxsize=32)
def get_deep_dive_data(ticker: str) -> dict:
    """
    (แก้ไข: ลด API Calls ที่ไม่จำเป็นสำหรับ ETL Job 3)
    ดึงเฉพาะข้อมูลที่จำเป็นสำหรับ Job 3 (Financials) และ Job DCF
    """
    try:
        tkr = yf.Ticker(ticker)
        # Fetch data needed for Financials and potentially DCF
        info = tkr.info # Still needed for DCF potentially
        if not info or info.get('quoteType') != 'EQUITY': return {"error": "Invalid ticker or no data available."}
        
        # Only fetch financials if needed (ETL Job 3 will call this)
        # Let's keep fetching them for now as DCF might need them too,
        # but be aware this is an area for potential optimization if DCF is also moved to DB.
        qf = tkr.quarterly_financials
        qbs = tkr.quarterly_balance_sheet
        qcf = tkr.quarterly_cashflow
        
        result = {"info": info} # Keep info for DCF or other uses
        
        result["financial_statements"] = {
            "income": qf.iloc[:,:16].dropna(how='all', axis=1) if qf is not None else pd.DataFrame(),
            "balance": qbs.iloc[:,:16].dropna(how='all', axis=1) if qbs is not None else pd.DataFrame(),
            "cashflow": qcf.iloc[:,:16].dropna(how='all', axis=1) if qcf is not None else pd.DataFrame()
        }

        # Removed price history, margin trends, financial trends as they are derived/stored elsewhere now
        # result["price_history"] = tkr.history(period="5y") # Removed

        return result
    except Exception as e:
        logging.error(f"Critical failure in get_deep_dive_data for {ticker}: {e}", exc_info=True)
        return {"error": str(e)}

@lru_cache(maxsize=32)
def calculate_dcf_intrinsic_value(ticker: str, forecast_growth_rate: float, perpetual_growth_rate: float, wacc_override: Optional[float]) -> dict:
    # (ฟังก์ชันนี้ยังคงเหมือนเดิม - ใช้ yfinance สด)
    # ... (โค้ดเดิม) ...
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info: return {'Ticker': ticker, 'error': 'Could not fetch company info from yfinance.'}
        
        ASSUMED_PERPETUAL_GROWTH, PROJECTION_YEARS, ASSUMED_MARKET_RETURN = perpetual_growth_rate, 5, 0.08
        income_stmt, balance_sheet, cashflow = ticker_obj.financials, ticker_obj.balance_sheet, ticker_obj.cashflow
        if any(df.empty for df in [income_stmt, balance_sheet, cashflow]): return {'Ticker': ticker, 'error': 'Financial statements missing.'}
        last_year_income, last_year_balance, last_year_cashflow = income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0], cashflow.iloc[:, 0]
        ebit, tax_provision, pretax_income = get_financial_data(last_year_income, ['EBIT', 'Ebit']), get_financial_data(last_year_income, ['Tax Provision', 'Income Tax Expense']), get_financial_data(last_year_income, ['Pretax Income', 'Income Before Tax'])
        d_and_a, capex = get_financial_data(last_year_cashflow, ['Depreciation And Amortization', 'Depreciation']), get_financial_data(last_year_cashflow, ['Capital Expenditure', 'CapEx'])
        cash, total_debt = get_financial_data(last_year_balance, ['Cash And Cash Equivalents', 'Cash']), get_financial_data(last_year_balance, ['Total Debt'])
        interest_expense = get_financial_data(last_year_income, ['Interest Expense', 'Interest Expense Net'])
        market_cap, beta, shares_outstanding, current_price = info.get('marketCap'), info.get('beta'), info.get('sharesOutstanding'), info.get('currentPrice') or info.get('previousClose')
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


@lru_cache(maxsize=32)
def calculate_monte_carlo_dcf(
    ticker: str,
    n_simulations: int,
    growth_min: float, growth_mode: float, growth_max: float,
    perpetual_min: float, perpetual_mode: float, perpetual_max: float,
    wacc_min: float, wacc_mode: float, wacc_max: float
) -> dict:
    """
    (ฟังก์ชันนี้ยังคงเหมือนเดิม - ใช้ yfinance สด)
    """
    try:
        ticker_obj, info = yf.Ticker(ticker), yf.Ticker(ticker).info
        if not info or not info.get('sharesOutstanding'):
            return {'error': 'Missing shares outstanding or basic info.'}

        income_stmt, balance_sheet, cashflow = ticker_obj.financials, ticker_obj.balance_sheet, ticker_obj.cashflow
        last_year_income, last_year_balance, last_year_cashflow = income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0], cashflow.iloc[:, 0]

        ebit = get_financial_data(last_year_income, ['EBIT', 'Ebit'])
        tax_provision = get_financial_data(last_year_income, ['Tax Provision', 'Income Tax Expense'])
        pretax_income = get_financial_data(last_year_income, ['Pretax Income', 'Income Before Tax'])
        d_and_a = get_financial_data(last_year_cashflow, ['Depreciation And Amortization', 'Depreciation'])
        capex = get_financial_data(last_year_cashflow, ['Capital Expenditure', 'CapEx'])
        cash_and_equivalents = get_financial_data(last_year_balance, ['Cash And Cash Equivalents', 'Cash'])
        total_debt = get_financial_data(last_year_balance, ['Total Debt'])
        shares_outstanding = info.get('sharesOutstanding')
        current_price = info.get('currentPrice') or info.get('previousClose')

        if any(v is None for v in [ebit, tax_provision, pretax_income, d_and_a, capex, cash_and_equivalents, shares_outstanding, current_price]):
            return {'error': 'Missing essential data for DCF.'}

        tax_rate = (tax_provision / pretax_income) if pretax_income != 0 else 0.21
        base_fcff = (ebit * (1 - tax_rate)) + d_and_a - capex
        net_debt = total_debt - cash_and_equivalents if total_debt is not None else -cash_and_equivalents
        PROJECTION_YEARS = 5

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
        
        if len(sim_wacc) == 0: # Check again after filtering
             return {'error': 'No valid simulations after filtering WACC > g.'}
        
        n_valid_simulations = len(sim_wacc) # Update simulation count

        future_fcffs = base_fcff * (1 + sim_growth) ** np.arange(1, PROJECTION_YEARS + 1)[:, np.newaxis]
        discounted_fcffs = future_fcffs / (1 + sim_wacc) ** np.arange(1, PROJECTION_YEARS + 1)[:, np.newaxis]
        final_year_fcff = future_fcffs[-1]
        
        # Calculate terminal value only for valid simulations
        terminal_value = (final_year_fcff * (1 + sim_perpetual)) / (sim_wacc - sim_perpetual)
        
        # Ensure terminal value is not negative (or handle appropriately)
        terminal_value[terminal_value < 0] = 0 
        
        discounted_terminal_value = terminal_value / ((1 + sim_wacc) ** PROJECTION_YEARS)

        enterprise_value = np.sum(discounted_fcffs, axis=0) + discounted_terminal_value
        equity_value = enterprise_value - net_debt
        simulated_intrinsic_values = equity_value / shares_outstanding

        # Filter out potential non-finite values before calculating statistics
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
            'num_valid_simulations': len(finite_values) # Add info about valid simulations
        }

    except Exception as e:
        logging.error(f"Monte Carlo DCF failed for {ticker}: {e}", exc_info=True)
        return {'error': str(e)}


def get_technical_analysis_data(price_history_df: pd.DataFrame) -> dict:
    """
    (ฟังก์ชันนี้ไม่ถูกแก้ไข - คำนวณ Indicators จาก DataFrame ที่ส่งเข้ามา)
    """
    if not isinstance(price_history_df, pd.DataFrame) or price_history_df.empty: return {"error": "Price history data is missing."}
    df = price_history_df.copy()
    try:
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
        ema12, ema26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return {"data": df.dropna(subset=['SMA50', 'SMA200', 'EMA20', 'BB_Upper', 'BB_Lower', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'])} # Drop rows where indicators couldn't be calculated initially
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}", exc_info=True)
        return {"error": f"Calculation error: {e}"}


@lru_cache(maxsize=32)
def calculate_exit_multiple_valuation(ticker: str, forecast_years: int, eps_growth_rate: float, terminal_pe: float) -> dict:
    """
    (ฟังก์ชันนี้จะถูกเลิกใช้ใน callbacks.py และย้าย Logic ไปคำนวณจาก DB)
    (แต่ยังคงไว้ก่อน เผื่อมีการเรียกใช้จากที่อื่น หรือเพื่อเปรียบเทียบ)
    """
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