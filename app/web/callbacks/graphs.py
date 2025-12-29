import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import logging
from sqlalchemy import func
from ... import db, server
from ...models import FactCompanySummary
from ...constants import COLOR_DISCRETE_MAP, INDEX_TICKER_TO_NAME
from ...data_handler import calculate_monte_carlo_dcf
from .utils import fetch_prices_with_fallback, TABS_CONFIG

def register_graph_callbacks(app):

    @app.callback(
        Output('analysis-pane-content', 'children'),
        [Input('analysis-tabs', 'active_tab'),
         Input('user-selections-store', 'data'),
         Input('dcf-assumptions-store', 'data'),
         Input('table-pane-content', 'children')]
    )
    def render_graph_content(active_tab, store_data, dcf_data, table_content):
        store_data = store_data or {'tickers': [], 'indices': []}
        tickers = tuple(store_data.get('tickers', []))
        indices = tuple(store_data.get('indices', []))
        all_symbols = tuple(set(tickers + indices))

        if not all_symbols:
            return dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")

        try:
            if active_tab == "tab-performance":
                start_of_year = datetime(datetime.now().year, 1, 1).date()
                raw_data = fetch_prices_with_fallback(all_symbols, start_of_year)

                if raw_data.empty: raise ValueError("No price data found (DB or Live) for YTD performance.")

                ytd_data = raw_data.pivot(index='date', columns='ticker', values='close').sort_index().ffill()
                if ytd_data.empty or len(ytd_data) < 2: raise ValueError("Not enough data after pivot.")

                ytd_perf = (ytd_data / ytd_data.iloc[0]) - 1
                
                fig = px.line(ytd_perf, title='YTD Performance Comparison', 
                              color_discrete_map=COLOR_DISCRETE_MAP, 
                              labels=INDEX_TICKER_TO_NAME,
                              template='plotly_dark') # [MODIFIED] Added Dark Template
                
                # [MODIFIED] Added Transparent Background
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            if active_tab == "tab-drawdown":
                one_year_ago = datetime.utcnow().date() - timedelta(days=365)
                raw_data = fetch_prices_with_fallback(all_symbols, one_year_ago)

                if raw_data.empty: raise ValueError("No price data found (DB or Live) for 1-Year Drawdown.")

                prices = raw_data.pivot(index='date', columns='ticker', values='close').sort_index().ffill()
                if prices.empty: raise ValueError("Not enough data after pivot.")

                rolling_max = prices.cummax()
                drawdown_data = (prices / rolling_max) - 1
                
                fig = px.line(drawdown_data, title='1-Year Drawdown Comparison', 
                              color_discrete_map=COLOR_DISCRETE_MAP, 
                              labels=INDEX_TICKER_TO_NAME,
                              template='plotly_dark') # [MODIFIED] Added Dark Template
                
                # [MODIFIED] Added Transparent Background
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            if active_tab == "tab-scatter":
                if not tickers: return dbc.Alert("Please select stocks to display the chart.", color="info", className="mt-3 text-center")

                with server.app_context():
                    latest_date_sq = db.session.query(
                        FactCompanySummary.ticker,
                        func.max(FactCompanySummary.date_updated).label('max_date')
                    ).filter(FactCompanySummary.ticker.in_(tickers)).group_by(FactCompanySummary.ticker).subquery()

                    query = db.session.query(
                        FactCompanySummary.ticker,
                        FactCompanySummary.ev_ebitda,
                        FactCompanySummary.ebitda_margin
                    ).join(
                        latest_date_sq,
                        (FactCompanySummary.ticker == latest_date_sq.c.ticker) &
                        (FactCompanySummary.date_updated == latest_date_sq.c.max_date)
                    )
                    df_scatter = pd.read_sql(query.statement, db.engine)
                    df_scatter.rename(columns={'ticker': 'Ticker', 'ev_ebitda': 'EV/EBITDA', 'ebitda_margin': 'EBITDA Margin'}, inplace=True)

                if df_scatter.empty: return dbc.Alert("Could not fetch scatter data from DB.", color="warning")

                df_scatter = df_scatter.dropna()
                if df_scatter.empty: return dbc.Alert("No valid scatter data points to plot.", color="warning")

                fig = px.scatter(df_scatter, x="EBITDA Margin", y="EV/EBITDA", text="Ticker", title="Valuation vs. Quality Analysis", template='plotly_dark') # [MODIFIED] Added Dark Template
                fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='lightgrey'))) # [MODIFIED] lighter border
                # [MODIFIED] Added Transparent Background & Grid adjustments
                fig.update_layout(xaxis_tickformat=".2%", yaxis_title="EV / EBITDA (Valuation)", xaxis_title="EBITDA Margin (Quality)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                x_avg, y_avg = df_scatter["EBITDA Margin"].mean(), df_scatter["EV/EBITDA"].mean()
                fig.add_vline(x=x_avg, line_width=1, line_dash="dash", line_color="grey"); fig.add_hline(y=y_avg, line_width=1, line_dash="dash", line_color="grey")
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            if active_tab == "tab-dcf":
                if not tickers: return dbc.Alert("Please select stocks for DCF simulation.", color="info", className="mt-3 text-center")
                if not dcf_data: return dbc.Alert("Please set simulation assumptions using the gear icon.", color="info", className="mt-3 text-center")

                all_results = []
                for ticker in tickers:
                    result = calculate_monte_carlo_dcf(
                        ticker=ticker,
                        n_simulations=dcf_data.get('simulations', 10000),
                        growth_min=dcf_data.get('growth_min', 3.0),
                        growth_mode=dcf_data.get('growth_mode', 5.0),
                        growth_max=dcf_data.get('growth_max', 8.0),
                        perpetual_min=dcf_data.get('perpetual_min', 1.5),
                        perpetual_mode=dcf_data.get('perpetual_mode', 2.5),
                        perpetual_max=dcf_data.get('perpetual_max', 3.0),
                        wacc_min=dcf_data.get('wacc_min', 7.0),
                        wacc_mode=dcf_data.get('wacc_mode', 8.0),
                        wacc_max=dcf_data.get('wacc_max', 10.0)
                    )

                    if 'error' not in result:
                        result['Ticker'] = ticker
                        all_results.append(result)
                    else:
                         logging.warning(f"DCF simulation failed for {ticker}: {result['error']}")

                if not all_results:
                    return dbc.Alert("Could not run simulation for any selected stocks. Check logs for details (e.g., missing financial data).", color="danger")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                for res in all_results: fig.add_trace(go.Histogram(x=res['simulated_values'], name=res['Ticker'], opacity=0.6, nbinsx=100), row=1, col=1)
                mos_data = [{'Ticker': r['Ticker'], 'current_price': r['current_price'], 'intrinsic_value': r['mean']} for r in all_results]; df_mos = pd.DataFrame(mos_data)
                fig.add_trace(go.Scatter(x=df_mos['current_price'], y=df_mos['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_mos['intrinsic_value'], y=df_mos['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Mean Intrinsic Value'), row=2, col=1)
                for i, row in df_mos.iterrows(): fig.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'], line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3), row=2, col=1)
                # [MODIFIED] Added Dark Template & Transparent Background
                fig.update_layout(title_text='Monte Carlo DCF Analysis', barmode='overlay', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig.update_yaxes(title_text="Frequency", row=1, col=1); fig.update_yaxes(title_text="Ticker", row=2, col=1); fig.update_xaxes(title_text="Share Price ($)", row=2, col=1)
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            return html.P("This is an empty tab!")

        except Exception as e:
            tab_name = TABS_CONFIG.get(active_tab, {}).get('tab_name', 'Graph')
            logging.error(f"Error rendering graph content for tab {tab_name} ({active_tab}): {e}", exc_info=True)
            return dbc.Alert(f"An error occurred while rendering '{tab_name}': {type(e).__name__} - {e}", color="danger")
