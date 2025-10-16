# auth.py (Refactored version - Step 2)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user

# ==================================================================
# ส่วนที่ 1: Layout Components (Unchanged)
# ==================================================================
def create_register_layout():
    """Creates the layout for the registration page."""
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Create a New Account", className="text-center"), width=12), className="mt-5"),
        dbc.Row(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dbc.Alert(id='register-alert', color="danger", is_open=False),
                    dbc.Label("Username"),
                    dbc.Input(id="reg-username-input", type="text", placeholder="Choose a username"),
                    dbc.Label("Password", className="mt-3"),
                    dbc.Input(id="reg-password-input", type="password", placeholder="Enter a password (min. 8 characters)"), # <-- Added placeholder text
                    dbc.Label("Confirm Password", className="mt-3"),
                    dbc.Input(id="reg-password-confirm-input", type="password", placeholder="Enter password again"),
                    dbc.Button("Create Account", id="create-account-button", color="primary", className="mt-4 w-100"),
                    html.Hr(),
                    dcc.Link("Already have an account? Log in", href="/", className="text-center d-block"),
                ])
            ]), width={"size": 6, "offset": 3}, md={"size": 4, "offset": 4}
        ), className="mt-4")
    ], fluid=True)


def create_login_modal():
    """Creates the layout for the login modal."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Login")),
            dbc.ModalBody([
                dbc.Alert(id='login-alert-modal', color="danger", is_open=False),
                dbc.Label("Username"),
                dbc.Input(id="username-input-modal", type="text", placeholder="Enter username"),
                dbc.Label("Password", className="mt-3"),
                dbc.Input(id="password-input-modal", type="password", placeholder="Enter password"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Create an Account", href="/register", color="secondary", className="me-auto"),
                dbc.Button("Login", id="login-button-modal", color="primary")
            ]),
        ],
        id="login-modal",
        is_open=False,
    )

# ==================================================================
# ส่วนที่ 2: Register Callbacks
# ==================================================================
def register_auth_callbacks(app, db, User):
    """Registers all authentication-related callbacks."""

    @app.callback(
        Output("login-modal", "is_open"),
        Input("open-login-modal-button", "n_clicks"),
        State("login-modal", "is_open"),
        prevent_initial_call=True,
    )
    def open_modal(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        Output('login-alert-modal', 'children'),
        Output('login-alert-modal', 'is_open'),
        Output("login-modal", "is_open", allow_duplicate=True),
        Input('login-button-modal', 'n_clicks'),
        State('username-input-modal', 'value'),
        State('password-input-modal', 'value'),
        prevent_initial_call=True
    )
    def login_user_modal(n_clicks, username, password):
        if n_clicks is None:
            return dash.no_update, "", False, False

        with app.server.app_context():
            if username and password:
                user = User.query.filter_by(username=username).first()
                if user and check_password_hash(user.password, password):
                    login_user(user, remember=True)
                    return '/', "", False, False
                return dash.no_update, "Invalid username or password.", True, True
        
        return dash.no_update, "Please enter username and password.", True, True

    @app.callback(
        Output('register-alert', 'children'),
        Output('register-alert', 'is_open'),
        Output('url', 'pathname', allow_duplicate=True),
        Input('create-account-button', 'n_clicks'),
        State('reg-username-input', 'value'),
        State('reg-password-input', 'value'),
        State('reg-password-confirm-input', 'value'),
        prevent_initial_call=True
    )
    def register_user_page(n_clicks, username, password, password_confirm):
        if n_clicks and username and password and password_confirm:
            with app.server.app_context():
                # --- [STEP 2 REFACTOR] ---
                # Add a simple password policy check for minimum length.
                if len(password) < 8:
                    return "Password must be at least 8 characters long.", True, dash.no_update
                # --- [END STEP 2 REFACTOR] ---

                if password != password_confirm:
                    return "Passwords do not match.", True, dash.no_update
                
                if User.query.filter_by(username=username).first():
                    return "Username already exists.", True, dash.no_update
                
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username=username, password=hashed_password)
                db.session.add(new_user)
                db.session.commit()
            
            # Use a success alert and redirect the user
            # This is a better user experience than just changing the URL.
            # (Note: For a true success message on the login page, more state management would be needed)
            return "Registration successful! Please log in.", False, "/" # Change alert to success color if desired
        
        # This alert is for when fields are missing.
        if n_clicks:
             return "Please fill in all fields.", True, dash.no_update
        
        return dash.no_update, False, dash.no_update