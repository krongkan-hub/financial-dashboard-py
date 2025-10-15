# data_handler.py (Fully Updated for New Deep Dive Design)

import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from functools import lru_cache
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ... (Helper functions like _pick_row, _cagr, etc. remain unchanged) ...
CONFIG = {
    "DAILY_RECENT_DAYS": 365, "PNG_DPI": 130, "FIG_SIZE": (7.5, 4.0),
    "TZ": "Asia/Bangkok", "FUND_YEARS": 5, "MONEY_DIV": 1_000_000,
}
FIN_KEYS = {
    "revenue": ["Total Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "op_income": ["Operating Income", "Operating Income or Loss", "Ebit", "Earnings Before Interest and Taxes"],
    "net_income": ["Net Income", "Net Income Applicable To Common Shares"],
    "ebitda": ["EBITDA", "Ebitda"]
}
CF_KEYS = {"cfo": ["Total Cash From Operating Activities", "Operating Cash Flow"],"capex": ["Capital Expenditures"]}

def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    original_index, df.index = df.index, df.index.str.lower()
    for k in candidates:
        if k.lower() in df.index:
            series = df.loc[k.lower()]
            series.name = original_index[df.index.get_loc(k.lower())]
            df.index = original_index
            return series
    df.index = original_index
    return None

def _cagr(series: pd.Series, years: int) -> float:
    series = series.dropna().sort_index().tail(years + 1)
    if len(series) < 2: return np.nan
    start_val, end_val = series.iloc[0], series.iloc[-1]
    num_years = (series.index[-1].year - series.index[0].year)
    if pd.notna(start_val) and pd.notna(end_val) and start_val > 0 and num_years > 0:
        return (end_val / start_val)**(1.0 / num_years) - 1.0
    return np.nan

def get_financial_data(statement_series, possible_keys):
    for key in possible_keys:
        if key in statement_series.index:
            return abs(statement_series[key])
    return None

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
    except Exception: pass
    return pd.Series(dtype=float)

# ... (calculate_drawdown, get_competitor_data, get_scatter_data remain unchanged) ...
@lru_cache(maxsize=20)
def calculate_drawdown(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    if not tickers: return pd.DataFrame()
    prices = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)['Close']
    if prices.empty: return pd.DataFrame()
    if isinstance(prices, pd.Series): prices = prices.to_frame(name=tickers[0])
    rolling_max = prices.cummax()
    return (prices / rolling_max) - 1

@lru_cache(maxsize=10)
def get_competitor_data(tickers: tuple) -> pd.DataFrame:
    logging.info(f"--- FETCHING COMPETITOR DATA for {tickers} ---")
    all_data = []
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info
            revenue_series = get_revenue_series(ticker)
            cagr_3y = _cagr(revenue_series, 3) if not revenue_series.empty else np.nan
            cfo_series = _pick_row(tkr.cashflow, CF_KEYS['cfo'])
            ni_series = _pick_row(tkr.financials, FIN_KEYS['net_income'])
            cash_conversion = np.nan
            if cfo_series is not None and ni_series is not None and not cfo_series.empty and not ni_series.empty:
                latest_cfo = cfo_series.iloc[0]
                latest_ni = ni_series.iloc[0]
                if pd.notna(latest_cfo) and pd.notna(latest_ni) and latest_ni != 0: cash_conversion = latest_cfo / latest_ni
            all_data.append({
                "Ticker": ticker, "Price": info.get('currentPrice') or info.get('previousClose'),
                "Market Cap": info.get("marketCap"), "P/E": info.get('trailingPE'), "P/B": info.get('priceToBook'),
                "EV/EBITDA": info.get('enterpriseToEbitda'), "Revenue Growth (YoY)": info.get('revenueGrowth'),
                "Revenue CAGR (3Y)": cagr_3y, "Net Income Growth (YoY)": info.get('earningsGrowth'),
                "ROE": info.get('returnOnEquity'), "D/E Ratio": info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') is not None else np.nan,
                "Operating Margin": info.get('operatingMargins'), "Cash Conversion": cash_conversion
            })
        except Exception as e: logging.error(f"An error occurred processing summary for {ticker}: {e}")
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

@lru_cache(maxsize=10)
def get_scatter_data(tickers: tuple) -> pd.DataFrame:
    scatter_data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            scatter_data.append({'Ticker': ticker, 'EV/EBITDA': info.get('enterpriseToEbitda'), 'EBITDA Margin': info.get('ebitdaMargins')})
        except Exception as e: logging.warning(f"Could not fetch scatter data for {ticker}: {e}")
    return pd.DataFrame(scatter_data).dropna()


# ==================================================================
# ส่วนที่ 3: ฟังก์ชันสำหรับ DEEP DIVE (ส่วนที่เพิ่มใหม่และอัปเดต)
# ==================================================================
@lru_cache(maxsize=32)
def get_deep_dive_data(ticker: str) -> dict:
    """
    ดึงข้อมูลทั้งหมดที่จำเป็นสำหรับหน้า Deep Dive ในครั้งเดียว
    """
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        if not info or info.get('quoteType') != 'EQUITY':
            return {"error": "Invalid ticker or no data available."}

        # --- Helper functions for formatting ---
        def format_pct(val):
            return f"{val*100:.2f}%" if pd.notna(val) else "N/A"
        def format_large_number(n):
            if pd.isna(n): return "N/A"
            if n >= 1e12: return f'${n/1e12:,.2f}T'
            if n >= 1e9: return f'${n/1e9:,.2f}B'
            if n >= 1e6: return f'${n/1e6:,.2f}M'
            return f'${n:,.0f}'

        # --- 1. Marquee and Key Stats Data ---
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        previous_close = info.get('previousClose')
        daily_change = current_price - previous_close if current_price and previous_close else 0
        daily_change_pct = (daily_change / previous_close) * 100 if previous_close else 0

        result = {
            "company_name": info.get('longName', ticker),
            "exchange": info.get('exchange', 'N/A'),
            "logo_url": info.get('logo_url'),
            "business_summary": info.get('longBusinessSummary'),
            "current_price": current_price,
            "daily_change": daily_change,
            "daily_change_pct": daily_change_pct,
            "daily_change_str": f"{'+' if daily_change >= 0 else ''}{daily_change:,.2f}",
            "daily_change_pct_str": f"{'+' if daily_change >= 0 else ''}{daily_change_pct:.2f}%",
            "market_cap_str": format_large_number(info.get('marketCap')),
            "key_stats": {
                "P/E Ratio": f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A",
                "Forward P/E": f"{info.get('forwardPE'):.2f}" if info.get('forwardPE') else "N/A",
                "PEG Ratio": f"{info.get('pegRatio'):.2f}" if info.get('pegRatio') else "N/A",
                "Dividend Yield": format_pct(info.get('dividendYield')),
            }
        }

        # --- 2. Financial Trends ---
        income_stmt_annual = tkr.financials.iloc[:, :4] # Get last 4 years
        revenue = _pick_row(income_stmt_annual, FIN_KEYS['revenue'])
        net_income = _pick_row(income_stmt_annual, FIN_KEYS['net_income'])
        financial_trends = pd.DataFrame({'Revenue': revenue, 'Net Income': net_income})
        financial_trends.index = pd.to_datetime(financial_trends.index).year
        result["financial_trends"] = financial_trends.sort_index().dropna(how='all')

        # --- 3. Financial Statements (Annual) ---
        result["financial_statements"] = {
            "income": tkr.financials.iloc[:, :4].dropna(how='all', axis=1),
            "balance": tkr.balance_sheet.iloc[:, :4].dropna(how='all', axis=1),
            "cashflow": tkr.cashflow.iloc[:, :4].dropna(how='all', axis=1)
        }
        
        # --- 4. Price History ---
        result["price_history"] = tkr.history(period="5y")

        return result

    except Exception as e:
        logging.error(f"Failed to get deep dive data for {ticker}: {e}")
        return {"error": str(e)}

# --- DCF Function (unchanged) ---
@lru_cache(maxsize=32)
def calculate_dcf_intrinsic_value(ticker: str, forecast_growth_rate: float) -> dict:
    PROJECTION_YEARS, ASSUMED_MARKET_RETURN, ASSUMED_PERPETUAL_GROWTH = 5, 0.08, 0.025
    try:
        ticker_obj, info = yf.Ticker(ticker), yf.Ticker(ticker).info
        income_stmt, balance_sheet, cashflow = ticker_obj.income_stmt, ticker_obj.balance_sheet, ticker_obj.cashflow
        last_year_income, last_year_balance, last_year_cashflow = income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0], cashflow.iloc[:, 0]
        ebit, tax_provision, pretax_income = get_financial_data(last_year_income, ['EBIT', 'Ebit']), get_financial_data(last_year_income, ['Tax Provision', 'Income Tax Expense']), get_financial_data(last_year_income, ['Pretax Income', 'Income Before Tax'])
        d_and_a, capex = get_financial_data(last_year_cashflow, ['Depreciation And Amortization', 'Depreciation']), get_financial_data(last_year_cashflow, ['Capital Expenditure', 'CapEx'])
        market_cap, total_debt, interest_expense = info.get('marketCap'), get_financial_data(last_year_balance, ['Total Debt', 'Total Debt Net Minority Interest']), get_financial_data(last_year_income, ['Interest Expense', 'Interest Expense Net'])
        beta, cash_and_equivalents, shares_outstanding = info.get('beta'), get_financial_data(last_year_balance, ['Cash And Cash Equivalents', 'Cash']), info.get('sharesOutstanding')
        current_price = info.get('currentPrice') or info.get('previousClose')
        required_components = [ebit, tax_provision, pretax_income, d_and_a, capex, market_cap, total_debt, interest_expense, beta, cash_and_equivalents, shares_outstanding]
        if any(v is None for v in required_components): return {'Ticker': ticker, 'error': 'Missing essential data for DCF.'}
        tax_rate = tax_provision / pretax_income if pretax_income != 0 else 0.21
        base_fcff = (ebit * (1 - tax_rate)) + d_and_a - capex
        cost_of_debt_rd = interest_expense / total_debt if total_debt != 0 else 0.05
        tnx = yf.Ticker('^TNX'); risk_free_rate = (tnx.history(period='1d')['Close'].iloc[0]) / 100 if not tnx.history(period='1d').empty else 0.04
        cost_of_equity_re = risk_free_rate + beta * (ASSUMED_MARKET_RETURN - risk_free_rate)
        equity_weight, debt_weight = market_cap / (market_cap + total_debt), total_debt / (market_cap + total_debt)
        wacc = (equity_weight * cost_of_equity_re) + (debt_weight * cost_of_debt_rd * (1 - tax_rate))
        if (wacc - ASSUMED_PERPETUAL_GROWTH) <= 0: return {'Ticker': ticker, 'error': 'WACC <= perpetual growth rate.'}
        future_fcffs = [base_fcff * ((1 + forecast_growth_rate) ** year) for year in range(1, PROJECTION_YEARS + 1)]
        discounted_fcffs = [fcff / ((1 + wacc) ** (year + 1)) for year, fcff in enumerate(future_fcffs)]
        terminal_value = (future_fcffs[-1] * (1 + ASSUMED_PERPETUAL_GROWTH)) / (wacc - ASSUMED_PERPETUAL_GROWTH)
        discounted_terminal_value = terminal_value / ((1 + wacc) ** PROJECTION_YEARS)
        enterprise_value = sum(discounted_fcffs) + discounted_terminal_value
        net_debt = total_debt - cash_and_equivalents
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding
        return {'Ticker': ticker, 'intrinsic_value': intrinsic_value_per_share, 'current_price': current_price, 'wacc': wacc}
    except Exception as e:
        logging.error(f"DCF calculation failed for {ticker}: {e}")
        return {'Ticker': ticker, 'error': str(e)}