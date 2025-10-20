# data_handler.py (Final Version with Hugging Face API)

import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from functools import lru_cache
import logging
import warnings
from datetime import datetime, timedelta
from urllib.parse import urlparse
import requests # <-- Import ที่จำเป็น
import os       # <-- Import ที่จำเป็น

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

# --- ส่วนที่ 1: การตั้งค่า NewsAPI (เหมือนเดิม) ---
from config import Config
from newsapi import NewsApiClient

if Config.NEWS_API_KEY:
    newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY)
else:
    newsapi = None
    logging.warning("NEWS_API_KEY not found in config. News fetching will be disabled.")

# --- ส่วนที่ 2: ฟังก์ชันใหม่สำหรับเรียกใช้ Hugging Face API ---
def analyze_sentiment_via_api(text_to_analyze: str) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    hf_token = os.environ.get('HUGGING_FACE_TOKEN')

    if not hf_token:
        logging.error("HUGGING_FACE_TOKEN environment variable not set.")
        return {'label': 'neutral', 'score': 0.0}
        
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text_to_analyze[:512]} # ตัดข้อความไม่ให้เกิน 512 ตัวอักษร
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status() # เช็คว่า request สำเร็จหรือไม่ (เช่น 200 OK)
        result = response.json()
        
        if result and isinstance(result, list) and isinstance(result[0], list):
            best_prediction = max(result[0], key=lambda x: x['score'])
            return {'label': best_prediction['label'].lower(), 'score': best_prediction['score']}

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
    except (KeyError, IndexError, TypeError):
        # logging.error(f"Could not parse API response. Response: {result}, Error: {e}")
        pass
        
    return {'label': 'neutral', 'score': 0.0} # คืนค่า default ถ้ามีปัญหา

# --- ส่วนที่ 3: ฟังก์ชันดึงข่าวที่ถูกแก้ไขให้เรียกใช้ API ---
@lru_cache(maxsize=32)
def get_news_and_sentiment(company_name: str) -> dict:
    if not newsapi or not os.environ.get('HUGGING_FACE_TOKEN'):
        return {"error": "News service or Hugging Face Token is not configured."}

    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        all_articles = newsapi.get_everything(
            q=f'"{company_name}"', language='en', sort_by='relevancy',
            from_param=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'),
            page_size=20
        )

        if all_articles['status'] != 'ok' or all_articles['totalResults'] == 0:
            return {"articles": [], "summary": {}}

        analyzed_articles = []
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for article in all_articles['articles']:
            if not article.get('description'): continue
            
            text_to_analyze = article['title'] + ". " + article['description']
            result = analyze_sentiment_via_api(text_to_analyze) # <-- เรียกใช้ API แทนการโหลดโมเดล
            
            article['sentiment'] = result['label']
            article['sentiment_score'] = result['score']
            analyzed_articles.append(article)
            sentiment_scores[result['label']] += 1

        total_analyzed = len(analyzed_articles)
        summary = {
            'positive_count': sentiment_scores['positive'], 'neutral_count': sentiment_scores['neutral'],
            'negative_count': sentiment_scores['negative'], 'total_count': total_analyzed,
            'positive_pct': (sentiment_scores['positive'] / total_analyzed) * 100 if total_analyzed > 0 else 0,
            'neutral_pct': (sentiment_scores['neutral'] / total_analyzed) * 100 if total_analyzed > 0 else 0,
            'negative_pct': (sentiment_scores['negative'] / total_analyzed) * 100 if total_analyzed > 0 else 0,
        }
        return {"articles": analyzed_articles, "summary": summary}
        
    except Exception as e:
        logging.error(f"Error in get_news_and_sentiment for {company_name}: {e}", exc_info=True)
        return {"error": str(e)}

# --- โค้ดส่วนที่เหลือ (ตั้งแต่บรรทัดนี้ลงไป) ไม่มีการเปลี่ยนแปลง ---

FIN_KEYS = {
    "revenue": ["Total Revenue", "Revenues", "Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold"],
    "gross_profit": ["Gross Profit"],
    "op_income": ["Operating Income", "Operating Income or Loss", "Ebit", "Earnings Before Interest and Taxes"],
    "net_income": ["Net Income", "Net Income From Continuing Ops", "Net Income Applicable To Common Shares"],
}
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
        except Exception:
            continue
    return None

def _cagr(series: pd.Series, years: int) -> float:
    series = series.dropna().sort_index().tail(years + 1)
    if len(series) < 2: return np.nan
    num_years = len(series) - 1
    if pd.api.types.is_datetime64_any_dtype(series.index):
        num_years = (series.index[-1].year - series.index[0].year)
    start_val, end_val = series.iloc[0], series.iloc[-1]
    if pd.notna(start_val) and pd.notna(end_val) and start_val > 0 and num_years > 0:
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
    logo_url = info.get('logo_url')
    if not logo_url:
        website = info.get('website')
        if website:
            try:
                domain = urlparse(website).netloc.replace('www.', '')
                if domain: logo_url = f"https://logo.clearbit.com/{domain}"
            except Exception: logo_url = None
    return logo_url

@lru_cache(maxsize=32)
def get_revenue_series(ticker: str) -> pd.Series:
    try:
        tkr = yfinance.Ticker(ticker)
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
    if not tickers: return pd.DataFrame()
    try:
        data = yfinance.download(list(tickers), period=period, auto_adjust=True, progress=False)
        if data.empty: return pd.DataFrame()
        prices = data['Close']
        if isinstance(prices, pd.Series): prices = prices.to_frame(name=tickers[0])
        rolling_max = prices.cummax()
        return (prices / rolling_max) - 1
    except Exception as e: logging.error(f"Error in calculate_drawdown for {tickers}: {e}")
    return pd.DataFrame()

@lru_cache(maxsize=10)
def get_competitor_data(tickers: tuple) -> pd.DataFrame:
    all_data = []
    for ticker in tickers:
        try:
            tkr = yfinance.Ticker(ticker)
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
                "Market Cap": info.get("marketCap"), "Beta": info.get("beta"),
                "P/E": info.get('trailingPE'), "P/B": info.get('priceToBook'),
                "EV/EBITDA": info.get('enterpriseToEbitda'), "Revenue Growth (YoY)": info.get('revenueGrowth'), "Revenue CAGR (3Y)": cagr_3y,
                "Net Income Growth (YoY)": info.get('earningsGrowth'), "ROE": info.get('returnOnEquity'),
                "D/E Ratio": info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') is not None else np.nan,
                "Operating Margin": info.get('operatingMargins'), "Cash Conversion": cash_conversion
            })
        except Exception as e: logging.error(f"An error occurred processing summary for {ticker}: {e}")
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

@lru_cache(maxsize=10)
def get_scatter_data(tickers: tuple) -> pd.DataFrame:
    scatter_data = []
    for ticker in tickers:
        try:
            info = yfinance.Ticker(ticker).info
            scatter_data.append({'Ticker': ticker, 'EV/EBITDA': info.get('enterpriseToEbitda'), 'EBITDA Margin': info.get('ebitdaMargins')})
        except Exception as e: logging.warning(f"Could not fetch scatter data for {ticker}: {e}")
    return pd.DataFrame(scatter_data).dropna()

@lru_cache(maxsize=32)
def get_deep_dive_data(ticker: str) -> dict:
    try:
        tkr = yfinance.Ticker(ticker)
        info = tkr.info
        if not info or info.get('quoteType') != 'EQUITY': return {"error": "Invalid ticker or no data available."}

        company_name = info.get('longName', ticker)
        sentiment_data = get_news_and_sentiment(company_name)

        logo_url = _get_logo_url(info)
        def format_large_number(n):
            if pd.isna(n) or n is None: return "N/A"
            if abs(n) >= 1e12: return f'${n/1e12:,.2f}T'
            if abs(n) >= 1e9: return f'${n/1e9:,.2f}B'
            if abs(n) >= 1e6: return f'${n/1e6:,.2f}M'
            return f'${n:,.0f}'
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', 1)
        daily_change, daily_change_pct = current_price - previous_close, ((current_price - previous_close) / previous_close) * 100
        
        result = {
            "company_name": company_name,
            "exchange": info.get('exchange', 'N/A'), "logo_url": logo_url,
            "business_summary": info.get('longBusinessSummary', 'Business summary not available.'), "current_price": current_price,
            "daily_change_str": f"{'+' if daily_change >= 0 else ''}{daily_change:,.2f}", "daily_change_pct_str": f"{'+' if daily_change_pct >= 0 else ''}{daily_change_pct:.2f}%",
            "market_cap_str": format_large_number(info.get('marketCap')), "target_mean_price": info.get('targetMeanPrice'),
            "recommendation_key": info.get('recommendationKey', 'N/A').replace('_', ' ').title(),
            "key_stats": { "P/E Ratio": f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A", "Forward P/E": f"{info.get('forwardPE'):.2f}" if info.get('forwardPE') else "N/A", "PEG Ratio": f"{info.get('pegRatio'):.2f}" if info.get('pegRatio') else "N/A", "Dividend Yield": f"{info.get('dividendYield')*100:.2f}%" if info.get('dividendYield') else "N/A", },
            "sentiment_data": sentiment_data
        }
        
        income_stmt_quarterly = tkr.quarterly_financials.iloc[:, :16]

        revenue = _pick_row(income_stmt_quarterly, FIN_KEYS['revenue'])
        gross_profit = _pick_row(income_stmt_quarterly, FIN_KEYS['gross_profit'])
        op_income = _pick_row(income_stmt_quarterly, FIN_KEYS['op_income'])
        net_income = _pick_row(income_stmt_quarterly, FIN_KEYS['net_income'])
        if gross_profit is None and revenue is not None:
            cost_of_revenue = _pick_row(income_stmt_quarterly, FIN_KEYS['cost_of_revenue'])
            if cost_of_revenue is not None: gross_profit = revenue - cost_of_revenue

        financial_trends = pd.DataFrame({'Revenue': revenue, 'Net Income': net_income})
        financial_trends['Revenue'] = pd.to_numeric(financial_trends['Revenue'], errors='coerce')
        financial_trends['Net Income'] = pd.to_numeric(financial_trends['Net Income'], errors='coerce')

        margin_trends = pd.DataFrame(index=financial_trends.index)
        if revenue is not None and not revenue.empty:
            revenue_safe = revenue.replace(0, np.nan)
            if gross_profit is not None: margin_trends['Gross Margin'] = gross_profit / revenue_safe
            if op_income is not None: margin_trends['Operating Margin'] = op_income / revenue_safe
            if net_income is not None: margin_trends['Net Margin'] = net_income / revenue_safe

        if not financial_trends.empty and pd.api.types.is_datetime64_any_dtype(financial_trends.index):
            financial_trends.index = financial_trends.index.to_period('Q').strftime('%Y-Q%q')
        if not margin_trends.empty and pd.api.types.is_datetime64_any_dtype(margin_trends.index):
            margin_trends.index = margin_trends.index.to_period('Q').strftime('%Y-Q%q')

        result["financial_trends"] = financial_trends.sort_index().dropna(how='all')
        result["margin_trends"] = margin_trends.sort_index().dropna(how='all')

        result["financial_statements"] = {
            "income": tkr.quarterly_financials.iloc[:, :16].dropna(how='all', axis=1),
            "balance": tkr.quarterly_balance_sheet.iloc[:, :16].dropna(how='all', axis=1),
            "cashflow": tkr.quarterly_cashflow.iloc[:, :16].dropna(how='all', axis=1)
        }

        result["price_history"] = tkr.history(period="5y")
        
        return result
    except Exception as e:
        logging.error(f"Critical failure in get_deep_dive_data for {ticker}: {e}", exc_info=True)
        return {"error": str(e)}

@lru_cache(maxsize=32)
def calculate_dcf_intrinsic_value(
    ticker: str,
    forecast_growth_rate: float,
    perpetual_growth_rate: Optional[float] = None,
    wacc_override: Optional[float] = None
) -> dict:
    ASSUMED_PERPETUAL_GROWTH = perpetual_growth_rate if perpetual_growth_rate is not None else 0.025
    PROJECTION_YEARS, ASSUMED_MARKET_RETURN = 5, 0.08
    try:
        ticker_obj, info = yfinance.Ticker(ticker), yfinance.Ticker(ticker).info
        if not info or not info.get('sharesOutstanding'): return {'Ticker': ticker, 'error': 'Missing shares outstanding or basic info.'}
        income_stmt, balance_sheet, cashflow = ticker_obj.financials, ticker_obj.balance_sheet, ticker_obj.cashflow
        if any(df.empty for df in [income_stmt, balance_sheet, cashflow]): return {'Ticker': ticker, 'error': 'One or more financial statements are empty.'}
        last_year_income, last_year_balance, last_year_cashflow = income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0], cashflow.iloc[:, 0]
        
        ebit = get_financial_data(last_year_income, ['EBIT', 'Ebit'])
        tax_provision = get_financial_data(last_year_income, ['Tax Provision', 'Income Tax Expense'])
        pretax_income = get_financial_data(last_year_income, ['Pretax Income', 'Income Before Tax'])
        d_and_a = get_financial_data(last_year_cashflow, ['Depreciation And Amortization', 'Depreciation'])
        capex = get_financial_data(last_year_cashflow, ['Capital Expenditure', 'CapEx'])
        cash_and_equivalents = get_financial_data(last_year_balance, ['Cash And Cash Equivalents', 'Cash'])
        total_debt = get_financial_data(last_year_balance, ['Total Debt', 'Total Debt Net Minority Interest'])
        interest_expense = get_financial_data(last_year_income, ['Interest Expense', 'Interest Expense Net'])
        
        market_cap = info.get('marketCap')
        beta = info.get('beta')
        shares_outstanding = info.get('sharesOutstanding')
        current_price = info.get('currentPrice') or info.get('previousClose')

        required_components = [ebit, tax_provision, pretax_income, d_and_a, capex, cash_and_equivalents, shares_outstanding, current_price]
        if any(v is None for v in required_components): return {'Ticker': ticker, 'error': 'Missing essential data for DCF.'}
        
        tax_rate = (tax_provision / pretax_income) if pretax_income and pretax_income != 0 else 0.21
        base_fcff = (ebit * (1 - tax_rate)) + d_and_a - capex

        wacc = None
        if wacc_override is not None:
            wacc = wacc_override
        else:
            if any(v is None for v in [beta, market_cap, total_debt, interest_expense]):
                return {'Ticker': ticker, 'error': 'Missing data for automatic WACC calculation.'}
            cost_of_debt_rd = (interest_expense / total_debt) if total_debt != 0 else 0.05
            tnx_history = yfinance.Ticker('^TNX').history(period='1d')
            risk_free_rate = (tnx_history['Close'].iloc[0] / 100) if not tnx_history.empty else 0.04
            cost_of_equity_re = risk_free_rate + beta * (ASSUMED_MARKET_RETURN - risk_free_rate)
            total_capital = market_cap + total_debt
            if total_capital == 0: return {'Ticker': ticker, 'error': 'Total capital is zero.'}
            equity_weight, debt_weight = market_cap / total_capital, total_debt / total_capital
            wacc = (equity_weight * cost_of_equity_re) + (debt_weight * cost_of_debt_rd * (1 - tax_rate))

        if wacc is None or wacc <= ASSUMED_PERPETUAL_GROWTH:
             return {'Ticker': ticker, 'error': f'WACC ({wacc:.2%}) is invalid or less than perpetual growth ({ASSUMED_PERPETUAL_GROWTH:.2%}).'}
        
        future_fcffs = [base_fcff * ((1 + forecast_growth_rate) ** year) for year in range(1, PROJECTION_YEARS + 1)]
        discounted_fcffs = [fcff / ((1 + wacc) ** (year + 1)) for year, fcff in enumerate(future_fcffs)]
        terminal_value = (future_fcffs[-1] * (1 + ASSUMED_PERPETUAL_GROWTH)) / (wacc - ASSUMED_PERPETUAL_GROWTH)
        discounted_terminal_value = terminal_value / ((1 + wacc) ** PROJECTION_YEARS)
        enterprise_value = sum(discounted_fcffs) + discounted_terminal_value
        net_debt = total_debt - cash_and_equivalents if total_debt is not None else -cash_and_equivalents
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding

        return {'Ticker': ticker, 'intrinsic_value': intrinsic_value_per_share, 'current_price': current_price, 'wacc': wacc}
    except Exception as e:
        logging.error(f"Critical failure in DCF calculation for {ticker}: {e}", exc_info=True)
        return {'Ticker': ticker, 'error': 'An unexpected error occurred during calculation.'}

def get_technical_analysis_data(price_history_df: pd.DataFrame) -> dict:
    if not isinstance(price_history_df, pd.DataFrame) or price_history_df.empty:
        return {"error": "Price history data is missing or invalid."}
    
    df = price_history_df.copy()
    
    # Technical Indicators
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
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
    return {"data": df}

@lru_cache(maxsize=32)
def calculate_exit_multiple_valuation(ticker: str, forecast_years: int, eps_growth_rate: float, terminal_pe: float) -> dict:
    """
    Calculates future valuation metrics based on an exit multiple approach.
    Returns a dictionary with the results for a single ticker.
    """
    try:
        eps_growth_rate_decimal = eps_growth_rate / 100.0
        tkr = yfinance.Ticker(ticker)
        info = tkr.info
        current_price = info.get('currentPrice') or info.get('previousClose')
        trailing_eps = info.get('trailingEps')
        if not all([current_price, trailing_eps, trailing_eps > 0]):
            return {'error': 'Missing data'}
        future_eps = trailing_eps * ((1 + eps_growth_rate_decimal) ** forecast_years)
        target_price = future_eps * terminal_pe
        target_upside = (target_price / current_price) - 1 if current_price > 0 else 0
        irr = ((target_price / current_price) ** (1 / forecast_years)) - 1 if current_price > 0 and forecast_years > 0 else 0
        return {
            'Target Price': target_price,
            'Target Upside': target_upside,
            'IRR %': irr
        }
    except Exception:
        return {'error': 'Calculation failed'}