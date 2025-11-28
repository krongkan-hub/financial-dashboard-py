import dash
from dash import html, callback_context, Output, Input, State
import dash_bootstrap_components as dbc
from flask_login import current_user
from ... import db, server
from ...models import UserAssumptions
from ..layout import METRIC_DEFINITIONS
from .utils import TABS_CONFIG

def register_modal_callbacks(app):

    @app.callback(Output('forecast-assumptions-modal', 'is_open'), Input('open-forecast-modal-btn', 'n_clicks'), State('forecast-assumptions-modal', 'is_open'), prevent_initial_call=True)
    def toggle_forecast_modal(n_clicks, is_open): return not is_open if n_clicks else is_open

    @app.callback(Output('open-forecast-modal-btn', 'style'), Input('table-tabs', 'active_tab'))
    def toggle_gear_button_visibility(active_tab): return {'display': 'inline-block'} if active_tab == 'tab-forecast' else {'display': 'none'}

    @app.callback(Output('forecast-assumptions-store', 'data', allow_duplicate=True), Output('forecast-assumptions-modal', 'is_open', allow_duplicate=True), Input('apply-forecast-changes-btn', 'n_clicks'), [State('modal-forecast-years-input', 'value'), State('modal-forecast-eps-growth-input', 'value'), State('modal-forecast-terminal-pe-input', 'value')], prevent_initial_call=True)
    def save_forecast_assumptions(n_clicks, years, growth, pe):
        if n_clicks:
            new_data = {'years': years, 'growth': growth, 'pe': pe}
            if current_user.is_authenticated:
                with server.app_context():
                    user_id = current_user.id; assumptions = UserAssumptions.query.filter_by(user_id=user_id).first()
                    if not assumptions: assumptions = UserAssumptions(user_id=user_id); db.session.add(assumptions)
                    assumptions.forecast_years, assumptions.eps_growth, assumptions.terminal_pe = years, growth, pe
                    db.session.commit()
            return new_data, False
        return dash.no_update, dash.no_update

    @app.callback(Output("definitions-modal", "is_open"), Output("definitions-modal-title", "children"), Output("definitions-modal-body", "children"), [Input("open-definitions-modal-btn-graphs", "n_clicks"), Input("open-definitions-modal-btn-tables", "n_clicks"), Input("close-definitions-modal-btn", "n_clicks")], [State("definitions-modal", "is_open"), State("analysis-tabs", "active_tab"), State("table-tabs", "active_tab")], prevent_initial_call=True)
    def toggle_definitions_modal(graphs_clicks, tables_clicks, close_clicks, is_open, analysis_tab, table_tab):
        ctx = callback_context;
        if not ctx.triggered or ctx.triggered_id == "close-definitions-modal-btn": return False, dash.no_update, dash.no_update
        tab_id = analysis_tab if ctx.triggered_id == "open-definitions-modal-btn-graphs" else table_tab
        tab_config = TABS_CONFIG.get(tab_id, {}); tab_name = tab_config.get('tab_name', 'Metric'); title = f"{tab_name.upper()} DEFINITION"; body_content = []
        if ctx.triggered_id == "open-definitions-modal-btn-graphs": body_content = METRIC_DEFINITIONS.get(tab_id, html.P("No definition available."))
        else:
            columns_in_tab = tab_config.get('columns', [])
            for col in columns_in_tab:
                if col in METRIC_DEFINITIONS: body_content.append(METRIC_DEFINITIONS[col]); body_content.append(html.Hr())
            if body_content: body_content.pop()
            else: body_content = [html.P("No definitions for this tab.")]
        return True, title, body_content

    @app.callback(Output('open-dcf-modal-btn', 'style'), Input('analysis-tabs', 'active_tab'))
    def toggle_dcf_gear_button_visibility(active_tab): return {'display': 'inline-block'} if active_tab == 'tab-dcf' else {'display': 'none'}

    @app.callback(Output('dcf-assumptions-modal', 'is_open'), Input('open-dcf-modal-btn', 'n_clicks'), State('dcf-assumptions-modal', 'is_open'), prevent_initial_call=True)
    def toggle_dcf_modal(n_clicks, is_open): return not is_open if n_clicks else is_open

    @app.callback(Output('dcf-assumptions-store', 'data', allow_duplicate=True), Output('dcf-assumptions-modal', 'is_open', allow_duplicate=True), Input('apply-dcf-changes-btn', 'n_clicks'), [State('mc-dcf-simulations-input', 'value'), State('mc-dcf-growth-min', 'value'), State('mc-dcf-growth-mode', 'value'), State('mc-dcf-growth-max', 'value'), State('mc-dcf-perpetual-min', 'value'), State('mc-dcf-perpetual-mode', 'value'), State('mc-dcf-perpetual-max', 'value'), State('mc-dcf-wacc-min', 'value'), State('mc-dcf-wacc-mode', 'value'), State('mc-dcf-wacc-max', 'value')], prevent_initial_call=True)
    def save_dcf_assumptions(n_clicks, sims, g_min, g_mode, g_max, p_min, p_mode, p_max, w_min, w_mode, w_max):
        if n_clicks:
            new_data = {'simulations': sims, 'growth_min': g_min, 'growth_mode': g_mode, 'growth_max': g_max, 'perpetual_min': p_min, 'perpetual_mode': p_mode, 'perpetual_max': p_max, 'wacc_min': w_min, 'wacc_mode': w_mode, 'wacc_max': w_max}
            if current_user.is_authenticated:
                with server.app_context():
                    user_id = current_user.id; assumptions = UserAssumptions.query.filter_by(user_id=user_id).first()
                    if not assumptions: assumptions = UserAssumptions(user_id=user_id); db.session.add(assumptions)
                    assumptions.dcf_simulations, assumptions.dcf_growth_min, assumptions.dcf_growth_mode, assumptions.dcf_growth_max = sims, g_min, g_mode, g_max
                    assumptions.dcf_perpetual_min, assumptions.dcf_perpetual_mode, assumptions.dcf_perpetual_max = p_min, p_mode, p_max
                    assumptions.dcf_wacc_min, assumptions.dcf_wacc_mode, assumptions.dcf_wacc_max = w_min, w_mode, w_max
                    db.session.commit()
            return new_data, False
        return dash.no_update, dash.no_update

    @app.callback([Output('mc-dcf-simulations-input', 'value'), Output('mc-dcf-growth-min', 'value'), Output('mc-dcf-growth-mode', 'value'), Output('mc-dcf-growth-max', 'value'), Output('mc-dcf-perpetual-min', 'value'), Output('mc-dcf-perpetual-mode', 'value'), Output('mc-dcf-perpetual-max', 'value'), Output('mc-dcf-wacc-min', 'value'), Output('mc-dcf-wacc-mode', 'value'), Output('mc-dcf-wacc-max', 'value')], Input('dcf-assumptions-store', 'data'))
    def sync_dcf_modal_inputs(dcf_data):
        if not dcf_data: return 10000, 3.0, 5.0, 8.0, 1.5, 2.5, 3.0, 7.0, 8.0, 10.0
        return (dcf_data.get('simulations', 10000), dcf_data.get('growth_min', 3.0), dcf_data.get('growth_mode', 5.0), dcf_data.get('growth_max', 8.0), dcf_data.get('perpetual_min', 1.5), dcf_data.get('perpetual_mode', 2.5), dcf_data.get('perpetual_max', 3.0), dcf_data.get('wacc_min', 7.0), dcf_data.get('wacc_mode', 8.0), dcf_data.get('wacc_max', 10.0))

    @app.callback([Output('modal-forecast-years-input', 'value'), Output('modal-forecast-eps-growth-input', 'value'), Output('modal-forecast-terminal-pe-input', 'value')], Input('forecast-assumptions-store', 'data'))
    def sync_forecast_modal_inputs(forecast_data):
        if not forecast_data: return 5, 10, 20
        return forecast_data.get('years', 5), forecast_data.get('growth', 10), forecast_data.get('pe', 20)
