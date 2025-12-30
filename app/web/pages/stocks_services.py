
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
