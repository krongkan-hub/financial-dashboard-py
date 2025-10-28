# index.py (CLEANED - No Scheduler)

import os 
from dash import dcc, html
from flask import redirect
from flask_login import logout_user
import logging 

# Import core objects from app.py
from app import app, server, db, User

# Import layout components from layout.py
from layout import METRIC_DEFINITIONS

# Import callback registration functions
from auth import register_auth_callbacks
from callbacks import register_callbacks

# --- [REMOVED] APScheduler and ETL Job imports (ไม่จำเป็นต้องใช้ในไฟล์นี้แล้ว) ---
# from apscheduler.schedulers.background import BackgroundScheduler
# from etl import update_company_summaries, update_daily_prices, update_financial_statements, update_news_sentiment
# from pytz import utc 

# --- Logout Route ---
@server.route('/logout')
def logout():
    logout_user()
    return redirect('/', code=302)

# --- App Layout ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# --- Register All Callbacks ---
register_callbacks(app, METRIC_DEFINITIONS)
register_auth_callbacks(app, db, User)

# --- [REMOVED] Function to start the ETL Scheduler (start_scheduler) ---
# --- [REMOVED] Block ที่ใช้เรียก start_scheduler() ---

# --- Run App (For Local Development Only) ---
if __name__ == '__main__':
    # Run the Flask development server (for local testing)
    # Gunicorn runs 'server' object directly in production
    app.run(debug=False, use_reloader=False)