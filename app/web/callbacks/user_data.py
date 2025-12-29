import dash
from dash import html, dcc, callback_context, Output, Input, State, ALL
import dash_bootstrap_components as dbc
import json
import logging
from flask_login import current_user
from ... import db, server
from ...models import UserSelection, UserAssumptions, DimCompany
from ...constants import (
    TOP_5_DEFAULT_TICKERS, INDEX_TICKER_TO_NAME, SECTORS, 
    ALL_TICKERS_SORTED_BY_GROWTH, SECTOR_TO_INDEX_MAPPING
)

def register_user_data_callbacks(app):
    
    def save_selections_to_db(user_id, symbols, symbol_type):
        UserSelection.query.filter_by(user_id=user_id, symbol_type=symbol_type).delete(synchronize_session=False)
        for symbol in symbols: db.session.add(UserSelection(user_id=user_id, symbol_type=symbol_type, symbol=symbol))
        db.session.commit()

    @app.callback(
        Output('user-selections-store', 'data'),
        Output('forecast-assumptions-store', 'data'),
        Output('dcf-assumptions-store', 'data'),
        Input('url', 'pathname')
    )
    def load_user_data_to_store(pathname):
        default_dcf_data = {
            'simulations': 10000, 'growth_min': 3.0, 'growth_mode': 5.0, 'growth_max': 8.0,
            'perpetual_min': 1.5, 'perpetual_mode': 2.5, 'perpetual_max': 3.0,
            'wacc_min': 7.0, 'wacc_mode': 8.0, 'wacc_max': 10.0
        }
        default_forecast_data = {'years': 5, 'growth': 10, 'pe': 20}
        default_selections_data = {'tickers': TOP_5_DEFAULT_TICKERS, 'indices': ['^GSPC']}

        if pathname != '/register' and current_user.is_authenticated:
            with server.app_context():
                user_id = current_user.id
                stocks = UserSelection.query.filter_by(user_id=user_id, symbol_type='stock').all()
                indices = UserSelection.query.filter_by(user_id=user_id, symbol_type='index').all()
                if not stocks and not indices: selections_data = default_selections_data
                else: selections_data = {'tickers': [s.symbol for s in stocks], 'indices': [i.symbol for i in indices]}
                assumptions = UserAssumptions.query.filter_by(user_id=user_id).first()
                if assumptions:
                    forecast_data = {'years': assumptions.forecast_years, 'growth': assumptions.eps_growth, 'pe': assumptions.terminal_pe}
                    dcf_data = { 'simulations': assumptions.dcf_simulations, 'growth_min': assumptions.dcf_growth_min, 'growth_mode': assumptions.dcf_growth_mode, 'growth_max': assumptions.dcf_growth_max, 'perpetual_min': assumptions.dcf_perpetual_min, 'perpetual_mode': assumptions.dcf_perpetual_mode, 'perpetual_max': assumptions.dcf_perpetual_max, 'wacc_min': assumptions.dcf_wacc_min, 'wacc_mode': assumptions.dcf_wacc_mode, 'wacc_max': assumptions.dcf_wacc_max }
                else: forecast_data, dcf_data = default_forecast_data, default_dcf_data
                return selections_data, forecast_data, dcf_data
        return default_selections_data, default_forecast_data, default_dcf_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-ticker-button', 'n_clicks'), [State('ticker-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
    def add_ticker_to_store(n_clicks, selected_tickers, store_data):
        store_data = store_data or {'tickers': [], 'indices': []}; updated = False
        if selected_tickers:
            for ticker in selected_tickers:
                if ticker not in store_data['tickers']: store_data['tickers'].append(ticker); updated = True
            if updated and current_user.is_authenticated:
                with server.app_context(): save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input({'type': 'remove-stock', 'index': ALL}, 'n_clicks'), State('user-selections-store', 'data'), prevent_initial_call=True)
    def remove_ticker_from_store(n_clicks, store_data):
        if not any(n_clicks): return dash.no_update
        store_data = store_data or {'tickers': [], 'indices': []}; triggered_id_str = callback_context.triggered[0]['prop_id'].split('.')[0]
        try:
            triggered_id = json.loads(triggered_id_str)['index']
            if triggered_id in store_data['tickers']:
                store_data['tickers'].remove(triggered_id)
                if current_user.is_authenticated:
                    with server.app_context(): save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
                return store_data
        except (json.JSONDecodeError, KeyError): logging.warning(f"Could not parse triggered ID for remove-stock: {triggered_id_str}")
        return dash.no_update

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-index-button', 'n_clicks'), [State('index-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
    def add_index_to_store(n_clicks, selected_indices, store_data):
        store_data = store_data or {'tickers': [], 'indices': []}; updated = False
        if selected_indices:
            for index in selected_indices:
                if index not in store_data['indices']: store_data['indices'].append(index); updated = True
            if updated and current_user.is_authenticated:
                with server.app_context(): save_selections_to_db(current_user.id, store_data['indices'], 'index')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input({'type': 'remove-index', 'index': ALL}, 'n_clicks'), State('user-selections-store', 'data'), prevent_initial_call=True)
    def remove_index_from_store(n_clicks, store_data):
        if not any(n_clicks): return dash.no_update
        store_data = store_data or {'tickers': [], 'indices': []}; triggered_id_str = callback_context.triggered[0]['prop_id'].split('.')[0]
        try:
            triggered_id = json.loads(triggered_id_str)['index']
            if triggered_id in store_data['indices']:
                store_data['indices'].remove(triggered_id)
                if current_user.is_authenticated:
                    with server.app_context(): save_selections_to_db(current_user.id, store_data['indices'], 'index')
                return store_data
        except (json.JSONDecodeError, KeyError): logging.warning(f"Could not parse triggered ID for remove-index: {triggered_id_str}")
        return dash.no_update

    @app.callback(Output('ticker-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_ticker_summary_display(store_data):
        tickers = store_data.get('tickers', []) if store_data else []
        if not tickers: return html.Div([html.Span("No stocks selected.", className="text-white-50 fst-italic")])
        return [html.Label("Selected Stocks:", className="text-white small fw-bold")] + [dbc.Badge([t, html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-stock', 'index': t})], color="light", className="m-1 p-2 text-dark border") for t in tickers]

    @app.callback(Output('index-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_index_summary_display(store_data):
        indices = store_data.get('indices', []) if store_data else []
        if not indices: return html.Span("No indices selected.", className="text-white-50 fst-italic")
        return [html.Label("Selected Indices:", className="text-white small fw-bold")] + [
            dbc.Badge([
                INDEX_TICKER_TO_NAME.get(t, t),
                html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-index', 'index': t})
            ], color="light", className="m-1 p-2 text-dark border")
            for t in indices
        ]

    @app.callback(
        Output('ticker-select-dropdown', 'options'),
        [Input('sector-dropdown', 'value'), Input('user-selections-store', 'data')]
    )
    def update_ticker_options(selected_sector, store_data):
        if not selected_sector: return []
        selected_tickers = set(store_data.get('tickers', [])) if store_data else set()
        if selected_sector == 'All': tickers_to_display_list = ALL_TICKERS_SORTED_BY_GROWTH
        else: tickers_to_display_list = SECTORS.get(selected_sector, [])
        tickers_to_query = [t for t in tickers_to_display_list if t not in selected_tickers]
        if not tickers_to_query: return []

        with server.app_context():
            try:
                results = db.session.query(DimCompany.ticker, DimCompany.company_name) \
                                    .filter(DimCompany.ticker.in_(tickers_to_query)) \
                                    .all()
                ticker_name_map = {ticker: name for ticker, name in results if name}
            except Exception as e:
                logging.error(f"Error querying company names for dropdown: {e}")
                return [{'label': t, 'value': t} for t in tickers_to_query]

        options = []
        for ticker in tickers_to_query:
            company_name = ticker_name_map.get(ticker)
            # Use company name if available, otherwise fallback to ticker
            label = f"{company_name} ({ticker})" if company_name else ticker
            options.append({'label': label, 'value': ticker})
        return options

    @app.callback(Output('index-select-dropdown', 'options'), Input('user-selections-store', 'data'))
    def update_index_options(store_data):
        if not store_data or not store_data.get('tickers'): return []
        selected_tickers, selected_indices = store_data.get('tickers', []), store_data.get('indices', [])
        ticker_to_sector = {t: s for s, stocks in SECTORS.items() for t in stocks}
        active_sectors = {ticker_to_sector.get(t) for t in selected_tickers if ticker_to_sector.get(t)}
        relevant_indices = {idx for sec in active_sectors for idx in SECTOR_TO_INDEX_MAPPING.get(sec, [])} | {'^GSPC', '^NDX'}
        return [{'label': INDEX_TICKER_TO_NAME.get(i, i), 'value': i} for i in sorted(list(relevant_indices)) if i not in selected_indices]
