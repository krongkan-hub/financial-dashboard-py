# index.py (CLEANED - AND UPDATED FOR NEW STRUCTURE)

import os 
from dash import dcc, html
from flask import redirect
from flask_login import logout_user
import logging 

# --- [CHANGED] Import core objects from app/ package ---
from app import app, server, db
from app.models import User  # <-- Import User จาก models

# --- [CHANGED] Import layout components from app/web/ ---
from app.web.layout import METRIC_DEFINITIONS

# --- [CHANGED] Import callback registration functions from app/web/ ---
from app.web.auth import register_auth_callbacks
from app.web.callbacks import register_callbacks

# --- Logout Route (เหมือนเดิม) ---
@server.route('/logout')
def logout():
    logout_user()
    return redirect('/', code=302)

# --- App Layout (เหมือนเดิม) ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# --- Register All Callbacks (เหมือนเดิม) ---
register_callbacks(app, METRIC_DEFINITIONS)
register_auth_callbacks(app, db, User)

# --- Run App (เหมือนเดิม) ---
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)