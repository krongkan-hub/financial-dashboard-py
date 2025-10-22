# index.py (UPDATED - Main Entry Point)

from dash import dcc, html
from flask import redirect
from flask_login import logout_user

# Import core objects from app.py
from app import app, server, db, User

# Import layout components from layout.py
from layout import METRIC_DEFINITIONS

# Import callback registration functions
from auth import register_auth_callbacks
from callbacks import register_callbacks

# --- Logout Route ---
@server.route('/logout')
def logout():
    logout_user()
    return redirect('/', code=302)

# --- App Layout ---
# The main layout is a container for the page content, which is updated by the router callback
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# --- Register All Callbacks ---
register_callbacks(app, METRIC_DEFINITIONS)
register_auth_callbacks(app, db, User)

# --- Run App ---
if __name__ == '__main__':
    with server.app_context():
        db.create_all()
    # Run this file to start the app: python index.py
    app.run(debug=False)