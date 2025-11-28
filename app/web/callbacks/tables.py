import dash_bootstrap_components as dbc
from dash import dash_table, html, Input, Output
import pandas as pd
import logging
from sqlalchemy import func
from ... import db, server
from ...models import FactCompanySummary, DimCompany
from .utils import (
    apply_custom_scoring, _prepare_display_dataframe, 
    _generate_datatable_columns, _generate_datatable_style_conditionals, 
    TABS_CONFIG
)

def register_table_callbacks(app):

    @app.callback(
        Output('table-pane-content', 'children'),
        Output('sort-by-dropdown', 'options'),
        Output('sort-by-dropdown', 'value'),
        [Input('table-tabs', 'active_tab'),
         Input('user-selections-store', 'data'),
         Input('sort-by-dropdown', 'value'),
         Input('forecast-assumptions-store', 'data')]
    )
    def render_table_content(active_tab, store_data, sort_by_column, forecast_data):
        try:
            if not store_data or not store_data.get('tickers'):
                return dbc.Alert("Please select stocks to view the comparison table.", color="info", className="mt-3 text-center"), [], None

            tickers = tuple(store_data.get('tickers'))

            df_full = pd.DataFrame()
            with server.app_context():
                latest_date_sq = db.session.query(
                    FactCompanySummary.ticker,
                    func.max(FactCompanySummary.date_updated).label('max_date')
                ).group_by(FactCompanySummary.ticker).subquery()

                query = db.session.query(
                    FactCompanySummary,
                    DimCompany.logo_url
                ).join(
                    latest_date_sq,
                    (FactCompanySummary.ticker == latest_date_sq.c.ticker) &
                    (FactCompanySummary.date_updated == latest_date_sq.c.max_date)
                ).join(
                    DimCompany,
                    FactCompanySummary.ticker == DimCompany.ticker
                ).filter(
                    FactCompanySummary.ticker.in_(tickers)
                )

                results = query.all()

                if results:
                    data_list = []
                    for summary_obj, logo_url_val in results:
                        data_dict = {c.name: getattr(summary_obj, c.name, None) for c in summary_obj.__table__.columns}
                        data_dict['logo_url'] = logo_url_val
                        data_dict['market_cap_raw'] = data_dict.get('market_cap')
                        data_list.append(data_dict)
                    df_full = pd.DataFrame(data_list)
                else:
                     try:
                        df_full = pd.read_sql(query.statement, db.engine)
                        summary_cols = [c.name for c in FactCompanySummary.__table__.columns]
                        new_cols = summary_cols + ['logo_url']
                        if not df_full.empty:
                            df_full.columns = new_cols
                            df_full['market_cap_raw'] = df_full['market_cap']
                     except Exception as sql_err:
                         logging.warning(f"Failed direct read_sql fallback: {sql_err}")

            if df_full.empty:
                return dbc.Alert(f"No summary data found in the warehouse for: {', '.join(tickers)}. Please wait for the next ETL run or check ETL logs.", color="warning", className="mt-3 text-center"), [], None

            df_full = apply_custom_scoring(df_full)

            column_mapping = {
                'ticker': 'Ticker', 'price': 'Price',
                'market_cap': 'Market Cap',
                'beta': 'Beta', 'pe_ratio': 'P/E', 'pb_ratio': 'P/B', 'ev_ebitda': 'EV/EBITDA',
                'revenue_growth_yoy': 'Revenue Growth (YoY)', 'revenue_cagr_3y': 'Revenue CAGR (3Y)',
                'net_income_growth_yoy': 'Net Income Growth (YoY)',
                'roe': 'ROE', 'de_ratio': 'D/E Ratio',
                'operating_margin': 'Operating Margin', 'cash_conversion': 'Cash Conversion',
                'logo_url': 'logo_url',
                'trailing_eps': 'Trailing EPS',
                'ebitda_margin': 'EBITDA Margin',
            }
            df_full.rename(columns={k: v for k, v in column_mapping.items() if k in df_full.columns}, inplace=True)

            if active_tab == 'tab-forecast':
                forecast_years, eps_growth, terminal_pe = forecast_data.get('years'), forecast_data.get('growth'), forecast_data.get('pe')
                if all(v is not None for v in [forecast_years, eps_growth, terminal_pe]):

                    df_full['Trailing EPS'] = pd.to_numeric(df_full.get('Trailing EPS'), errors='coerce')
                    df_full['Price'] = pd.to_numeric(df_full.get('Price'), errors='coerce')

                    eps_growth_decimal = eps_growth / 100.0

                    def calc_target(row):
                        if pd.isna(row['Trailing EPS']) or row['Trailing EPS'] <= 0 or pd.isna(row['Price']) or row['Price'] <= 0:
                            return pd.NA, pd.NA, pd.NA

                        try:
                            future_eps = row['Trailing EPS'] * ((1 + eps_growth_decimal) ** forecast_years)
                            target_price = future_eps * terminal_pe
                            upside = ((target_price / row['Price']) - 1) * 100
                            irr = (((target_price / row['Price']) ** (1 / forecast_years)) - 1) * 100
                            return target_price, upside, irr
                        except Exception:
                            return pd.NA, pd.NA, pd.NA

                    df_full[['Target Price', 'Target Upside', 'IRR %']] = df_full.apply(calc_target, axis=1, result_type='expand')

            tab_config = TABS_CONFIG.get(active_tab, {})
            display_cols = tab_config.get("columns", [])
            higher_is_better = tab_config.get("higher_is_better", {})

            df_display = _prepare_display_dataframe(df_full)

            missing_cols = [c for c in display_cols if c not in df_display.columns]
            for c in missing_cols: df_display[c] = pd.NA

            df_display = df_display[display_cols]

            sort_options = [{'label': col, 'value': col} for col in display_cols if col != 'Ticker']
            
            if sort_by_column and sort_by_column in df_display.columns:
                ascending = not higher_is_better.get(sort_by_column, True)
                df_display = df_display.sort_values(by=sort_by_column, ascending=ascending)

            columns = _generate_datatable_columns(tab_config)
            style_data_conditional, style_cell_conditional = _generate_datatable_style_conditionals(tab_config)

            table = dash_table.DataTable(
                data=df_display.to_dict('records'),
                columns=columns,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'right', 'padding': '10px', 'fontFamily': 'sans-serif'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center'},
                style_data_conditional=style_data_conditional,
                style_cell_conditional=style_cell_conditional,
                markdown_options={"html": True},
                sort_action="native",
            )

            return table, sort_options, sort_by_column

        except Exception as e:
            logging.error(f"Error rendering table content: {e}", exc_info=True)
            return dbc.Alert(f"An error occurred while rendering the table: {e}", color="danger"), [], None
