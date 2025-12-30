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
from ...data_handler import calculate_monte_carlo_dcf, fetch_prices_with_fallback
from .utils import TABS_CONFIG
from app.web.pages.stocks_services import generate_ytd_performance_figure # [NEW] Import Service

def register_graph_callbacks(app):

    app.clientside_callback(
        """
        function(active_tab) {
            const styles = {'display': 'none'};
            const visible = {'display': 'block'};
            return [
                active_tab === 'tab-performance' ? visible : styles,
                active_tab === 'tab-drawdown' ? visible : styles,
                active_tab === 'tab-scatter' ? visible : styles,
                active_tab === 'tab-dcf' ? visible : styles
            ];
        }
        """,
        [Output('content-performance', 'style'),
         Output('content-drawdown', 'style'),
         Output('content-scatter', 'style'),
         Output('content-dcf', 'style')],
        Input('analysis-tabs', 'active_tab')
    )

    @app.callback(
        [Output('content-performance', 'children'),
         Output('content-drawdown', 'children'),
         Output('content-scatter', 'children'),
         Output('content-dcf', 'children')],
        [Input('user-selections-store', 'data'),
         Input('dcf-assumptions-store', 'data')]
    )
    def update_all_graphs(store_data, dcf_data):
        store_data = store_data or {'tickers': [], 'indices': []}
        tickers = tuple(store_data.get('tickers', []))
        indices = tuple(store_data.get('indices', []))
        all_symbols = tuple(set(tickers + indices))

        # --- 1. PERFORMANCE TAB ---
        perf_content = html.Div() # Default empty
        if all_symbols:
            fig_perf = generate_ytd_performance_figure(tickers, indices)
            if fig_perf:
                perf_content = dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_perf)))
            else:
                 perf_content = dbc.Alert("No data available for selected instruments.", color="warning", className="mt-3 text-center")
        else:
            perf_content = dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")

        # --- 2. DRAWDOWN TAB ---
        drawdown_content = html.Div()
        if all_symbols:
            try:
                one_year_ago = datetime.utcnow().date() - timedelta(days=365)
                raw_data = fetch_prices_with_fallback(all_symbols, one_year_ago)
                if not raw_data.empty:
                    prices = raw_data.pivot(index='date', columns='ticker', values='close').sort_index().ffill()
                    if not prices.empty:
                        rolling_max = prices.cummax()
                        drawdown_data = (prices / rolling_max) - 1
                        fig_dd = px.line(drawdown_data, title='1-Year Drawdown Comparison', 
                                      color_discrete_map=COLOR_DISCRETE_MAP, 
                                      labels=INDEX_TICKER_TO_NAME,
                                      template='plotly_dark')
                        fig_dd.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        drawdown_content = dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_dd)))
                    else:
                         drawdown_content = dbc.Alert("Not enough data for Drawdown.", color="warning")
                else:
                    drawdown_content = dbc.Alert("No price data found for Drawdown.", color="warning")
            except Exception as e:
                drawdown_content = dbc.Alert(f"Error rendering Drawdown: {e}", color="danger")
        else:
             drawdown_content = dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")

        # --- 3. SCATTER TAB ---
        scatter_content = html.Div()
        if tickers:
            try:
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

                if not df_scatter.empty:
                    df_scatter = df_scatter.dropna()
                    if not df_scatter.empty:
                        fig_sc = px.scatter(df_scatter, x="EBITDA Margin", y="EV/EBITDA", text="Ticker", title="Valuation vs. Quality Analysis", template='plotly_dark')
                        fig_sc.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='lightgrey')))
                        fig_sc.update_layout(xaxis_tickformat=".2%", yaxis_title="EV / EBITDA (Valuation)", xaxis_title="EBITDA Margin (Quality)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        x_avg, y_avg = df_scatter["EBITDA Margin"].mean(), df_scatter["EV/EBITDA"].mean()
                        fig_sc.add_vline(x=x_avg, line_width=1, line_dash="dash", line_color="grey"); fig_sc.add_hline(y=y_avg, line_width=1, line_dash="dash", line_color="grey")
                        scatter_content = dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_sc)))
                    else:
                        scatter_content = dbc.Alert("No valid scatter data points.", color="warning")
                else:
                    scatter_content = dbc.Alert("Could not fetch scatter data from DB.", color="warning")
            except Exception as e:
                scatter_content = dbc.Alert(f"Error rendering Scatter: {e}", color="danger")
        else:
             scatter_content = dbc.Alert("Please select stocks to display the chart.", color="info", className="mt-3 text-center")

        # --- 4. DCF TAB ---
        dcf_content = html.Div()
        if tickers:
            if dcf_data:
                try:
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

                    if all_results:
                        fig_dcf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                        for res in all_results: fig_dcf.add_trace(go.Histogram(x=res['simulated_values'], name=res['Ticker'], opacity=0.6, nbinsx=100), row=1, col=1)
                        mos_data = [{'Ticker': r['Ticker'], 'current_price': r['current_price'], 'intrinsic_value': r['mean']} for r in all_results]; df_mos = pd.DataFrame(mos_data)
                        fig_dcf.add_trace(go.Scatter(x=df_mos['current_price'], y=df_mos['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'), row=2, col=1)
                        fig_dcf.add_trace(go.Scatter(x=df_mos['intrinsic_value'], y=df_mos['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Mean Intrinsic Value'), row=2, col=1)
                        for i, row in df_mos.iterrows(): fig_dcf.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'], line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3), row=2, col=1)
                        fig_dcf.update_layout(title_text='Monte Carlo DCF Analysis', barmode='overlay', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        fig_dcf.update_yaxes(title_text="Frequency", row=1, col=1); fig_dcf.update_yaxes(title_text="Ticker", row=2, col=1); fig_dcf.update_xaxes(title_text="Share Price ($)", row=2, col=1)
                        dcf_content = dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_dcf)))
                    else:
                        dcf_content = dbc.Alert("Could not run simulation. Check missing data.", color="danger")
                except Exception as e:
                    dcf_content = dbc.Alert(f"Error rendering DCF: {e}", color="danger")
            else:
                 dcf_content = dbc.Alert("Please set simulation assumptions using the gear icon.", color="info", className="mt-3 text-center")
        else:
             dcf_content = dbc.Alert("Please select stocks for DCF simulation.", color="info", className="mt-3 text-center")

        return perf_content, drawdown_content, scatter_content, dcf_content
