# index.py (UPDATED - Main Entry Point with Job 2 Schedule)

from dash import dcc, html
from flask import redirect
from flask_login import logout_user
import logging # เพิ่ม logging

# Import core objects from app.py
from app import app, server, db, User

# Import layout components from layout.py
from layout import METRIC_DEFINITIONS

# Import callback registration functions
from auth import register_auth_callbacks
from callbacks import register_callbacks

# --- [NEW] Import APScheduler และ ETL Jobs ---
from apscheduler.schedulers.background import BackgroundScheduler
from etl import update_company_summaries, update_daily_prices # Import ทั้งสอง Job
# --- [END NEW] ---

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
    scheduler = BackgroundScheduler(daemon=True)
    logging.info("Initializing APScheduler...")

    # --- Job 1: Update Company Summaries ---
    scheduler.add_job(
        update_company_summaries,
        'cron',
        hour=8,  # 8:00 UTC = 15:00 Bangkok
        minute=0,
        misfire_grace_time=3600, # ถ้าพลาด ให้ลองใหม่ภายใน 1 ชม.
        id='update_summaries_job' # ตั้ง ID ให้ Job
    )
    logging.info("Scheduled job: update_company_summaries (Daily @ 08:00 UTC / 15:00 Bangkok)")

    # --- Job 2: Update Daily Prices ---
    scheduler.add_job(
        update_daily_prices,
        'cron',
        hour=9, # 9:00 UTC = 16:00 Bangkok - ให้ทำงานหลัง Job 1
        minute=0,
        misfire_grace_time=3600*3, # ถ้าพลาด ให้ลองใหม่ภายใน 3 ชม.
        id='update_prices_job' # ตั้ง ID ให้ Job
    )
    logging.info("Scheduled job: update_daily_prices (Daily @ 09:00 UTC / 16:00 Bangkok)")


    # --- Start Scheduler ---
    try:
        scheduler.start()
        logging.info("APScheduler has started successfully. ETL jobs are scheduled.")
    except Exception as e:
         logging.error(f"Failed to start APScheduler: {e}", exc_info=True)
# --- [END MODIFIED] ---

# --- [ย้ายมาตรงนี้] Start the scheduler when the module is imported ---
# Ensure it runs only once, especially in development with auto-reloading
if os.environ.get('WERKZEUG_RUN_MAIN') != 'true': # Check added for local dev server
    start_scheduler()

# --- Run App (For Local Development Only) ---
if __name__ == '__main__':
    # Run the Flask development server (for local testing)
    # Gunicorn runs 'server' object directly in production
    app.run(debug=False, use_reloader=False) # Add use_reloader=False