# data_handler.py (Definitive Corrected Version)

import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from functools import lru_cache
import logging
import warnings
from datetime import datetime
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

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
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
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
                "Market Cap": info.get("marketCap"), "P/E": info.get('trailingPE'), "P/B": info.get('priceToBook'),
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
            info = yf.Ticker(ticker).info
            scatter_data.append({'Ticker': ticker, 'EV/EBITDA': info.get('enterpriseToEbitda'), 'EBITDA Margin': info.get('ebitdaMargins')})
        except Exception as e: logging.warning(f"Could not fetch scatter data for {ticker}: {e}")
    return pd.DataFrame(scatter_data).dropna()

@lru_cache(maxsize=32)
def get_deep_dive_data(ticker: str) -> dict:
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        if not info or info.get('quoteType') != 'EQUITY': return {"error": "Invalid ticker or no data available."}
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
            "company_name": info.get('longName', ticker), "exchange": info.get('exchange', 'N/A'), "logo_url": logo_url,
            "business_summary": info.get('longBusinessSummary', 'Business summary not available.'), "current_price": current_price,
            "daily_change_str": f"{'+' if daily_change >= 0 else ''}{daily_change:,.2f}", "daily_change_pct_str": f"{'+' if daily_change_pct >= 0 else ''}{daily_change_pct:.2f}%",
            "market_cap_str": format_large_number(info.get('marketCap')), "target_mean_price": info.get('targetMeanPrice'),
            "recommendation_key": info.get('recommendationKey', 'N/A').replace('_', ' ').title(),
            "key_stats": { "P/E Ratio": f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A", "Forward P/E": f"{info.get('forwardPE'):.2f}" if info.get('forwardPE') else "N/A", "PEG Ratio": f"{info.get('pegRatio'):.2f}" if info.get('pegRatio') else "N/A", "Dividend Yield": f"{info.get('dividendYield')*100:.2f}%" if info.get('dividendYield') else "N/A", }
        }
        income_stmt_annual = tkr.financials.iloc[:, :4]
        revenue = _pick_row(income_stmt_annual, FIN_KEYS['revenue'])
        gross_profit = _pick_row(income_stmt_annual, FIN_KEYS['gross_profit'])
        op_income = _pick_row(income_stmt_annual, FIN_KEYS['op_income'])
        net_income = _pick_row(income_stmt_annual, FIN_KEYS['net_income'])
        if gross_profit is None and revenue is not None:
            cost_of_revenue = _pick_row(income_stmt_annual, FIN_KEYS['cost_of_revenue'])
            if cost_of_revenue is not None: gross_profit = revenue - cost_of_revenue
        
        # --- START FIX ---
        financial_trends = pd.DataFrame({'Revenue': revenue, 'Net Income': net_income})
        financial_trends['Revenue'] = pd.to_numeric(financial_trends['Revenue'], errors='coerce')
        financial_trends['Net Income'] = pd.to_numeric(financial_trends['Net Income'], errors='coerce')
        
        margin_trends = pd.DataFrame(index=financial_trends.index)
        if revenue is not None and not revenue.empty:
            revenue_safe = revenue.replace(0, np.nan)
            if gross_profit is not None: margin_trends['Gross Margin'] = gross_profit / revenue_safe
            if op_income is not None: margin_trends['Operating Margin'] = op_income / revenue_safe
            if net_income is not None: margin_trends['Net Margin'] = net_income / revenue_safe
        
        # Convert index to year for BOTH dataframes AFTER calculations are complete
        if not financial_trends.empty and pd.api.types.is_datetime64_any_dtype(financial_trends.index):
            financial_trends.index = financial_trends.index.year
        if not margin_trends.empty and pd.api.types.is_datetime64_any_dtype(margin_trends.index):
            margin_trends.index = margin_trends.index.year

        result["financial_trends"] = financial_trends.sort_index().dropna(how='all')
        result["margin_trends"] = margin_trends.sort_index().dropna(how='all')
        # --- END FIX ---
        
        result["financial_statements"] = {"income": tkr.financials.iloc[:, :4].dropna(how='all', axis=1), "balance": tkr.balance_sheet.iloc[:, :4].dropna(how='all', axis=1), "cashflow": tkr.cashflow.iloc[:, :4].dropna(how='all', axis=1)}
        result["price_history"] = tkr.history(period="5y")
        news_raw = tkr.news or []
        processed_news = [item for item in news_raw if 'title' in item and 'link' in item]
        for item in processed_news:
            item['providerPublishTime'] = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M') if 'providerPublishTime' in item else 'N/A'
        result["news"] = processed_news[:8]
        return result
    except Exception as e:
        logging.error(f"Critical failure in get_deep_dive_data for {ticker}: {e}", exc_info=True)
        return {"error": str(e)}

@lru_cache(maxsize=32)
def calculate_dcf_intrinsic_value(ticker: str, forecast_growth_rate: float) -> dict:
    PROJECTION_YEARS, ASSUMED_MARKET_RETURN, ASSUMED_PERPETUAL_GROWTH = 5, 0.08, 0.025
    try:
        ticker_obj, info = yf.Ticker(ticker), yf.Ticker(ticker).info
        if not info or not info.get('sharesOutstanding'): return {'Ticker': ticker, 'error': 'Missing shares outstanding or basic info.'}
        income_stmt, balance_sheet, cashflow = ticker_obj.financials, ticker_obj.balance_sheet, ticker_obj.cashflow
        if any(df.empty for df in [income_stmt, balance_sheet, cashflow]): return {'Ticker': ticker, 'error': 'One or more financial statements are empty.'}
        last_year_income, last_year_balance, last_year_cashflow = income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0], cashflow.iloc[:, 0]
        ebit = get_financial_data(last_year_income, ['EBIT', 'Ebit'])
        tax_provision, pretax_income = get_financial_data(last_year_income, ['Tax Provision', 'Income Tax Expense']), get_financial_data(last_year_income, ['Pretax Income', 'Income Before Tax'])
        d_and_a, capex = get_financial_data(last_year_cashflow, ['Depreciation And Amortization', 'Depreciation']), get_financial_data(last_year_cashflow, ['Capital Expenditure', 'CapEx'])
        market_cap, total_debt = info.get('marketCap'), get_financial_data(last_year_balance, ['Total Debt', 'Total Debt Net Minority Interest'])
        interest_expense, beta = get_financial_data(last_year_income, ['Interest Expense', 'Interest Expense Net']), info.get('beta')
        cash_and_equivalents = get_financial_data(last_year_balance, ['Cash And Cash Equivalents', 'Cash'])
        shares_outstanding, current_price = info.get('sharesOutstanding'), info.get('currentPrice') or info.get('previousClose')
        required_components = [ebit, tax_provision, pretax_income, d_and_a, capex, market_cap, total_debt, interest_expense, beta, cash_and_equivalents, shares_outstanding, current_price]
        if any(v is None for v in required_components): return {'Ticker': ticker, 'error': 'Missing essential data for DCF.'}
        tax_rate = (tax_provision / pretax_income) if pretax_income != 0 else 0.21
        base_fcff = (ebit * (1 - tax_rate)) + d_and_a - capex
        cost_of_debt_rd = (interest_expense / total_debt) if total_debt != 0 else 0.05
        tnx_history = yf.Ticker('^TNX').history(period='1d')
        risk_free_rate = (tnx_history['Close'].iloc[0] / 100) if not tnx_history.empty else 0.04
        cost_of_equity_re = risk_free_rate + beta * (ASSUMED_MARKET_RETURN - risk_free_rate)
        total_capital = market_cap + total_debt
        if total_capital == 0: return {'Ticker': ticker, 'error': 'Total capital is zero.'}
        equity_weight, debt_weight = market_cap / total_capital, total_debt / total_capital
        wacc = (equity_weight * cost_of_equity_re) + (debt_weight * cost_of_debt_rd * (1 - tax_rate))
        if wacc is None or wacc <= ASSUMED_PERPETUAL_GROWTH: return {'Ticker': ticker, 'error': 'WACC is invalid or less than perpetual growth rate.'}
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
        logging.error(f"Critical failure in DCF calculation for {ticker}: {e}", exc_info=True)
        return {'Ticker': ticker, 'error': 'An unexpected error occurred during calculation.'}