# index.py (UPDATED - แก้ไข NameError และ Timezone)

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

# --- [NEW] Import APScheduler และ ETL Jobs ---
from apscheduler.schedulers.background import BackgroundScheduler
# [FIX]: เพิ่ม update_financial_statements และ update_news_sentiment
from etl import update_company_summaries, update_daily_prices, update_financial_statements, update_news_sentiment
from pytz import utc # Import timezone

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

# --- [MODIFIED] Function to start the ETL Scheduler ---
def start_scheduler():
    scheduler = BackgroundScheduler(daemon=True, timezone=utc) 
    logging.info("Initializing APScheduler with UTC timezone...")

    # --- Job 1: Update Company Summaries (18:00 ICT -> 11:00 UTC) ---
    scheduler.add_job(
        update_company_summaries,
        'cron',
        hour=11,  # 11:00 UTC (18:00 ICT)
        minute=0,
        misfire_grace_time=3600, 
        id='update_summaries_job' 
    )
    logging.info("Scheduled job: update_company_summaries (Daily @ 11:00 UTC / 18:00 ICT)")

    # --- Job 3: Update Financial Statements (18:00 ICT -> 11:00 UTC) ---
    scheduler.add_job(
        update_financial_statements,
        'cron',
        hour=11,  # 11:00 UTC (18:00 ICT)
        minute=0,
        misfire_grace_time=3600*3, 
        id='update_financials_job'
    )
    logging.info("Scheduled job: update_financial_statements (Daily @ 11:00 UTC / 18:00 ICT)")

    # --- Job 2: Update Daily Prices (19:00 ICT -> 12:00 UTC) ---
    scheduler.add_job(
        update_daily_prices,
        'cron',
        hour=12, # 12:00 UTC (19:00 ICT)
        minute=0,
        misfire_grace_time=3600*3, 
        id='update_prices_job' 
    )
    logging.info("Scheduled job: update_daily_prices (Daily @ 12:00 UTC / 19:00 ICT)")

    # --- Job 4: Update News Sentiment (20:00 ICT -> 13:00 UTC) ---
    scheduler.add_job(
        update_news_sentiment,
        'cron',
        hour=13, # 13:00 UTC (20:00 ICT)
        minute=0,
        misfire_grace_time=3600*3, 
        id='update_news_sentiment_job'
    )
    logging.info("Scheduled job: update_news_sentiment (Daily @ 13:00 UTC / 20:00 ICT)")


    # --- Start Scheduler ---
    try:
        scheduler.start()
        logging.info("APScheduler has started successfully. ETL jobs are scheduled.")
    except Exception as e:
         logging.error(f"Failed to start APScheduler: {e}", exc_info=True)
# --- [END MODIFIED] ---

# --- [ย้ายมาตรงนี้] Start the scheduler when the module is imported ---
# Ensure it runs only once, especially in development with auto-reloading
if os.environ.get('WERKZEUG_RUN_MAIN') != 'true': 
    start_scheduler()

# --- Run App (For Local Development Only) ---
if __name__ == '__main__':
    # Run the Flask development server (for local testing)
    # Gunicorn runs 'server' object directly in production
    app.run(debug=False, use_reloader=False)