from dash import html, Output, Input
from ..pages import deep_dive, bonds
from ..layout import create_navbar, build_layout

def register_navigation_callbacks(app):
    @app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
    def display_page(pathname):
        if pathname == '/':
            return build_layout()
        elif pathname == '/bonds':
            return bonds.create_bonds_layout()
        elif pathname == '/derivatives':
            return html.Div([
                html.H1("Derivatives Analysis (Coming Soon) 🚧", className="mt-5"),
                html.P("หน้านี้กำลังอยู่ระหว่างการพัฒนา")
            ])
        else:
            return html.Div([
                html.H1("404: Not found 😔", className="mt-5"),
                html.P(f"ไม่พบเส้นทางที่ร้องขอ: {pathname}")
            ], style={'textAlign': 'center'})

    @app.callback(Output('navbar-container', 'children'), Input('url', 'pathname'))
    def update_navbar_callback(pathname):
        if pathname != '/register' and not pathname.startswith('/deepdive/'):
            return create_navbar()
        return None
