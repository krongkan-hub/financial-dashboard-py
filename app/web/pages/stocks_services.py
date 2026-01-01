
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html 
from datetime import datetime
from app.data_handler import fetch_prices_with_fallback

from app.constants import COLOR_DISCRETE_MAP, INDEX_TICKER_TO_NAME

def generate_ytd_performance_figure(tickers, indices, start_date=None):
    """
    Generates the 'YTD Perforance Comparison' Plotly figure.
    This logic is shared between server-side initial render and client-side callbacks.
    """
    store_data = {'tickers': tickers, 'indices': indices}
    tickers = tuple(store_data.get('tickers', []))
    indices = tuple(store_data.get('indices', []))
    all_symbols = tuple(set(tickers + indices))

    if not all_symbols:
        # Fallback or empty state fig
        return None

    if start_date is None:
        start_date = datetime(datetime.now().year, 1, 1).date()

    try:
        raw_data = fetch_prices_with_fallback(all_symbols, start_date)

        if raw_data.empty: 
            return None

        ytd_data = raw_data.pivot(index='date', columns='ticker', values='close').sort_index().ffill()
        if ytd_data.empty or len(ytd_data) < 2: 
             return None

        ytd_perf = (ytd_data / ytd_data.iloc[0]) - 1
        
        fig = px.line(ytd_perf, title='YTD Performance Comparison', 
                        color_discrete_map=COLOR_DISCRETE_MAP, 
                        labels=INDEX_TICKER_TO_NAME,
                        template='plotly_dark')
        
        fig.update_layout(
            yaxis_tickformat=".2%", 
            legend_title_text='Symbol', 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    except Exception as e:
        # In a real service we might log this, but for now return None to degrade gracefully
        return None

# --- NEW SERVICES FOR DASHBOARD ENHANCEMENTS ---

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fetch_financial_health_data(tickers):
    """
    Fetches balance sheet and income statement data to calculate health metrics.
    Returns a DataFrame with columns: [Ticker, Net Debt/EBITDA, Interest Coverage, Current Ratio, D/E Ratio]
    """
    data_list = []
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            # 1. Net Debt / EBITDA
            total_debt = info.get('totalDebt')
            total_cash = info.get('totalCash')
            ebitda = info.get('ebitda')
            
            net_debt_ebitda = None
            if total_debt is not None and total_cash is not None and ebitda:
                net_debt = total_debt - total_cash
                net_debt_ebitda = net_debt / ebitda
            
            # 2. Interest Coverage (EBIT / Interest Expense)
            interest_coverage = info.get('interestCoverage')
            
            if interest_coverage is None:
                fin = yf_ticker.financials
                if not fin.empty and 'EBIT' in fin.index and 'Interest Expense' in fin.index:
                    ebit = fin.loc['EBIT'].iloc[0]
                    int_exp = fin.loc['Interest Expense'].iloc[0]
                    if int_exp != 0:
                        interest_coverage = abs(ebit / int_exp)
            
            # 3. Current Ratio
            current_ratio = info.get('currentRatio')
            
            # 4. Debt to Equity
            debt_to_equity = info.get('debtToEquity') 
            
            data_list.append({
                'Ticker': ticker,
                'Net Debt/EBITDA': net_debt_ebitda,
                'Interest Coverage': interest_coverage,
                'Current Ratio': current_ratio,
                'D/E Ratio': debt_to_equity
            })
        except Exception as e:
            print(f"Error fetching health data for {ticker}: {e}")
            data_list.append({'Ticker': ticker})
            
    return pd.DataFrame(data_list)

def fetch_analyst_data(tickers):
    """
    Fetches analyst recommendations and target prices.
    Returns DataFrame: [Ticker, Consensus, Target Price, Target Low, Target High, Current Price, Upside %]
    """
    data_list = []
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            # Analyst Consensus: e.g. "strong_buy" -> "Strong Buy"
            rec_key = info.get('recommendationKey', 'none').replace('_', ' ').title()
            
            # Target Prices
            target_mean = info.get('targetMeanPrice')
            target_low = info.get('targetLowPrice')
            target_high = info.get('targetHighPrice')
            current_price = info.get('currentPrice') or info.get('previousClose')
            
            # Upside calculation
            upside = None
            if target_mean and current_price:
                upside = ((target_mean / current_price) - 1) * 100
            
            data_list.append({
                'Ticker': ticker,
                'Consensus': rec_key,
                'Target Price': target_mean,
                'Target Low': target_low,
                'Target High': target_high,
                'Current Price': current_price,
                'Upside %': upside
            })
        except Exception as e:
             print(f"Error fetching analyst data for {ticker}: {e}")
             data_list.append({'Ticker': ticker})

    return pd.DataFrame(data_list)

def generate_historical_valuation_figure(tickers):
    """
    Generates a single multi-line chart comparing the Historical P/E Ratio of selected stocks.
    X-Axis: Date (5 Years)
    Y-Axis: P/E Ratio
    Reference Lines: 15x, 20x, 25x.
    """
    if not tickers: return None
    
    fig = go.Figure()

    has_data = False
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # 1. Fetch Price (5 years)
            hist = yf_ticker.history(period="5y")
            if hist.empty: continue
            
            # 2. Fetch EPS Data (Robust Strategy)
            eps_daily = None
            
            # Strategy A: Annual Income Statement
            try:
                stmts = [yf_ticker.income_stmt, yf_ticker.financials] # Try both properties
                for stmt in stmts:
                    if stmt is not None and not stmt.empty:
                        # Try finding a valid EPS row
                        eps_row = None
                        if 'Basic EPS' in stmt.index: eps_row = stmt.loc['Basic EPS']
                        elif 'Diluted EPS' in stmt.index: eps_row = stmt.loc['Diluted EPS']
                        elif 'BasicEPS' in stmt.index: eps_row = stmt.loc['BasicEPS']
                        
                        if eps_row is not None:
                            # Use annual EPS
                            eps_annual = eps_row.sort_index()
                            # Reindex to daily (step function forward fill)
                            eps_daily = eps_annual.reindex(hist.index, method='ffill')
                            break # Found it
            except Exception as e:
                print(f"Strategy A failed for {ticker}: {e}")

            # Strategy B: Quarterly Income Statement (if Annual failed)
            if eps_daily is None:
                try:
                    q_stmt = yf_ticker.quarterly_income_stmt
                    if q_stmt is not None and not q_stmt.empty:
                        eps_row = None
                        if 'Basic EPS' in q_stmt.index: eps_row = q_stmt.loc['Basic EPS']
                        elif 'Diluted EPS' in q_stmt.index: eps_row = q_stmt.loc['Diluted EPS']
                        
                        if eps_row is not None:
                            # Quarterly EPS needs to be TTM-ized? Or just annualized?
                            # For simplicity in this trends graph, we can just use the TTM sum or 4x?
                            # Actually, plotting Price / Quarterly EPS * 4 is a decent approximation of P/E.
                            # Or usually P/E is Price / TTM EPS.
                            # Let's simple forward fill the quarterly value * 4 (crude annualized).
                            eps_q = eps_row.sort_index()
                            # Taking rolling sum of last 4 quarters would be better for TTM
                            eps_ttm = eps_q.rolling(window=4, min_periods=1).sum()
                            eps_daily = eps_ttm.reindex(hist.index, method='ffill')
                except Exception as e:
                    print(f"Strategy B failed for {ticker}: {e}")

            # Strategy C: Use TTM EPS from info (Constant Line - Last Resort)
            if eps_daily is None:
                ttm_eps = yf_ticker.info.get('trailingEps')
                if ttm_eps:
                    # Create a constant series (Last resort, allows plotting but flat earnings)
                    eps_daily = pd.Series(ttm_eps, index=hist.index)

            
            # 3. Calculate P/E Ratio Trend
            if eps_daily is not None:
                # Forward fill any remaining gaps
                eps_daily = eps_daily.ffill()
                
                # Align indices
                common_index = hist.index.intersection(eps_daily.index)
                if common_index.empty: continue
                
                hist_aligned = hist.loc[common_index]
                eps_aligned = eps_daily.loc[common_index]

                pe_series = hist_aligned['Close'] / eps_aligned
                
                # Filter out negative or extreme P/Es to keep chart readable? 
                # User wants to see the data, let's keep it but maybe clip visualization? 
                # No, raw data is better.
                
                # Plot P/E Line
                color = COLOR_DISCRETE_MAP.get(ticker, None) # Use consistent colors if available
                fig.add_trace(go.Scatter(
                    x=pe_series.index, 
                    y=pe_series, 
                    name=ticker, 
                    mode='lines',
                    line=dict(color=color, width=2)
                ))
                has_data = True
            
        except Exception as e:
            print(f"Error gen historical P/E for {ticker}: {e}")

    if not has_data:
        return None

    # Add Reference Lines (15x, 20x, 25x)
    fig.add_hline(y=15, line_width=1, line_dash="dash", line_color="green", opacity=0.5, annotation_text="15x (Value)", annotation_position="bottom right")
    fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="yellow", opacity=0.5, annotation_text="20x (Moderate)", annotation_position="bottom right")
    fig.add_hline(y=25, line_width=1, line_dash="dash", line_color="red", opacity=0.5, annotation_text="25x (Premium)", annotation_position="bottom right")

    fig.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        title_text="Historical P/E Ratio Comparison (5 Years)",
        yaxis_title="P/E Ratio",
        legend_title_text='Ticker'
    )
    return fig
