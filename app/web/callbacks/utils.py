import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf
from dash.dash_table.Format import Format, Scheme
from ... import db, server
from ...models import FactDailyPrices

# ==================================================================
# Table Helpers & Config
# ==================================================================
TABS_CONFIG = {
    "tab-performance": { "tab_name": "Performance" },
    "tab-drawdown": { "tab_name": "Drawdown" },
    "tab-scatter": { "tab_name": "Valuation vs. Quality" },
    "tab-dcf": { "tab_name": "Margin of Safety (DCF)" },
    "tab-valuation": { "columns": ["Ticker", "Market Cap", "Company Size", "Price", "P/E", "P/B", "EV/EBITDA"], "higher_is_better": {"P/E": False, "P/B": False, "EV/EBITDA": False}, "tab_name": "Valuation" },
    "tab-growth": { "columns": ["Ticker", "Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"], "higher_is_better": {k: True for k in ["Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"]}, "tab_name": "Growth" },
    "tab-fundamentals": { "columns": ["Ticker", "Operating Margin", "ROE", "D/E Ratio", "Cash Conversion"], "higher_is_better": {"Operating Margin": True, "ROE": True, "D/E Ratio": False, "Cash Conversion": True}, "tab_name": "Fundamentals" },
    "tab-forecast": {
        "columns": ["Ticker", "Target Price", "Target Upside", "IRR %", "Volatility Level", "Valuation Model", "Stock Profile"],
        "higher_is_better": {"Target Upside": True, "IRR %": True},
        "tab_name": "Target"
    }
}

def _format_market_cap(n):
    if pd.isna(n): return '-'
    return f'${n/1e9:,.2f}B' if abs(n) >= 1e9 else f'${n/1e6:,.2f}M'

def _prepare_display_dataframe(df_raw):
    df_display = df_raw.copy()
    def create_ticker_cell(row):
        logo_url, ticker = row.get('logo_url'), row['Ticker']
        logo_html = f'<img src="{logo_url}" style="height: 22px; width: 22px; margin-right: 8px; border-radius: 4px;" onerror="this.style.display=\'none\'">' if logo_url and pd.notna(logo_url) else ''
        return f'''<a href="/deepdive/{ticker}" style="text-decoration: none; color: inherit; font-weight: 600; display: flex; align-items: center;">{logo_html}<span>{ticker}</span></a>'''

    df_display['Ticker'] = df_display.apply(create_ticker_cell, axis=1)

    if "Market Cap" in df_display.columns:
        df_display["Market Cap"] = df_raw["market_cap_raw"].apply(_format_market_cap)

    scoring_cols = ['Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile']
    for col in scoring_cols:
        if col in df_display.columns:
            df_display[col] = df_raw[col].apply(lambda x: x if pd.notna(x) else "")
    return df_display

def _generate_datatable_columns(tab_config):
    columns = []
    for col in tab_config["columns"]:
        col_def = {"name": col.replace(" (3Y)", ""), "id": col}
        if col == 'Ticker': col_def.update({"type": "text", "presentation": "markdown"})
        elif col in ['Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile', 'Market Cap']: col_def.update({"type": "text"})
        elif col in ['Target Upside', 'IRR %']: col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        elif col == 'Target Price': col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        elif any(kw in col for kw in ['Growth', 'Margin', 'ROE', 'Conversion']): col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        else: col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        columns.append(col_def)
    return columns

def _generate_datatable_style_conditionals(tab_config):
    style_data_conditional, style_cell_conditional = [], []
    displayed_columns = tab_config["columns"]
    other_cols = [c for c in displayed_columns if c != 'Ticker']
    ticker_width_percent = 15
    style_cell_conditional.append({'if': {'column_id': 'Ticker'}, 'width': f'{ticker_width_percent}%'})
    if other_cols:
        other_col_width_percent = (100 - ticker_width_percent) / len(other_cols)
        for col in other_cols:
            style_cell_conditional.append({'if': {'column_id': col}, 'width': f'{other_col_width_percent}%'})
    style_cell_conditional.append({'if': {'column_id': 'Ticker'}, 'textAlign': 'left'})
    return style_data_conditional, style_cell_conditional

def apply_custom_scoring(df):
    if df.empty: return df

    bins = [0, 10e9, 100e9, 1000e9, float('inf')]
    labels = ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    df['Company Size'] = pd.cut(df['market_cap'], bins=bins, labels=labels, right=False)

    conditions_volatility = [df['beta'].isnull(), df['beta'] < 0.5, df['beta'] <= 2, df['beta'] > 2]
    choices_volatility = ["N/A", "Core", "Growth", "Hyper Growth"]
    df['Volatility Level'] = np.select(conditions_volatility, choices_volatility, default='N/A')

    conditions_valuation = [df['ev_ebitda'].isnull(), df['ev_ebitda'] < 10, df['ev_ebitda'] <= 25, df['ev_ebitda'] > 25]
    choices_valuation = ["N/A", "Cheap", "Fair Value", "Expensive"]
    df['Valuation Model'] = np.select(conditions_valuation, choices_valuation, default='N/A')

    def categorize_stock(row):
        vol, val = row['Volatility Level'], row['Valuation Model']
        if vol == 'Hyper Growth' and val == 'Fair Value': return "Promising Growth"
        if vol == 'Core' and val == 'Cheap': return "Hidden Value Gem"
        if vol == 'Hyper Growth' and val == 'Expensive': return "High-Flyer"
        if vol == 'Core' and val == 'Fair Value': return "Stable Compounder"
        if vol == 'Growth' and val == 'Fair Value': return "Core Holding"
        if vol == 'Growth' and val == 'Cheap': return "Value Opportunity"
        return "Needs Review"
    df['Stock Profile'] = df.apply(categorize_stock, axis=1)
    return df

def fetch_prices_with_fallback(symbols: tuple, start_date: datetime.date) -> pd.DataFrame:
    if not symbols: return pd.DataFrame()
    
    symbols_list = list(symbols)
    today = datetime.utcnow().date()
    
    df_db = pd.DataFrame()
    live_fetch_needed = symbols_list
    
    try:
        with server.app_context():
            query = db.session.query(FactDailyPrices.date, FactDailyPrices.ticker, FactDailyPrices.close) \
                              .filter(FactDailyPrices.ticker.in_(symbols_list),
                                      FactDailyPrices.date >= start_date)
            df_db = pd.read_sql(query.statement, db.engine)
        
        df_all = df_db.copy()
        db_tickers = set(df_db['ticker'].unique())
        
        missing_symbols = [s for s in symbols_list if s not in db_tickers]
        stale_symbols = []
        
        if not df_db.empty:
            for ticker in db_tickers:
                last_db_date = df_db[df_db['ticker'] == ticker]['date'].max()
                if (today - last_db_date.date()).days > 3: 
                    stale_symbols.append(ticker)
                    
        live_fetch_needed = list(set(missing_symbols + stale_symbols))
        
        if live_fetch_needed:
            df_all = df_db[~df_db['ticker'].isin(live_fetch_needed)].copy()
            
    except Exception as e:
        logging.warning(f"DB Query failed for Dashboard prices, forcing all to live fetch: {e}")
        df_all = pd.DataFrame()
        live_fetch_needed = symbols_list
        
    if live_fetch_needed:
        logging.info(f"Performing LIVE fallback fetch for {len(live_fetch_needed)} symbols: {live_fetch_needed}")
        
        try:
            df_live = yf.download(live_fetch_needed, start=start_date, end=today, auto_adjust=True, progress=False)
        except Exception as e:
            logging.error(f"yfinance live fetch failed: {e}")
            return df_all 
            
        if df_live.empty:
            logging.warning("Live fetch returned empty data.")
            return df_all 
            
        if isinstance(df_live.columns, pd.MultiIndex):
            df_live = df_live['Close'].rename_axis(['Date']).reset_index().rename(columns={'Date': 'date'})
            df_live = df_live.melt(id_vars='date', var_name='ticker', value_name='close')
        elif isinstance(df_live, pd.DataFrame) and 'Close' in df_live.columns:
            df_live = df_live.rename_axis(['Date']).reset_index().rename(columns={'Date': 'date', 'Close': 'close'})
            if len(live_fetch_needed) == 1:
                df_live['ticker'] = live_fetch_needed[0]
            else:
                 return df_all
        else:
            logging.error("Live fetched data format is unexpected.")
            return df_all
            
        df_live = df_live[['date', 'ticker', 'close']].dropna(subset=['close'])
        df_combined = pd.concat([df_all, df_live]).drop_duplicates(subset=['date', 'ticker'], keep='last')
        return df_combined

    return df_all
