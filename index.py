# index.py (UPDATED - แก้ไข NameError และ Timezone)

import os # <--- [FIX #1] เพิ่ม import os
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
from etl import update_company_summaries, update_daily_prices 
from pytz import utc # <--- [FIX #2] Import timezone

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
    # [FIX #2] เพิ่ม timezone=utc เพื่อความชัดเจน
    scheduler = BackgroundScheduler(daemon=True, timezone=utc) 
    logging.info("Initializing APScheduler with UTC timezone...")

    # # --- Job 1: Update Company Summaries ---
    # scheduler.add_job(
    #     update_company_summaries,
    #     'cron',
    #     hour=14,  # 8:00 UTC (21:00 Bangkok)
    #     minute=0,
    #     misfire_grace_time=3600, # ถ้าพลาด ให้ลองใหม่ภายใน 1 ชม.
    #     id='update_summaries_job' # ตั้ง ID ให้ Job
    # )
    # logging.info("Scheduled job: update_company_summaries (Daily @ 08:00 UTC)")

    # --- Job 2: Update Daily Prices ---
    scheduler.add_job(
        update_daily_prices,
        'cron',
        hour=15, # 9:00 UTC (22:00 Bangkok)
        minute=0,
        misfire_grace_time=3600*3, # ถ้าพลาด ให้ลองใหม่ภายใน 3 ชม.
        id='update_prices_job' # ตั้ง ID ให้ Job
    )
    logging.info("Scheduled job: update_daily_prices (Daily @ 09:00 UTC)")


    # --- Start Scheduler ---
    try:
        scheduler.start()
        logging.info("APScheduler has started successfully. ETL jobs are scheduled.")
    except Exception as e:
         logging.error(f"Failed to start APScheduler: {e}", exc_info=True)
# --- [END MODIFIED] ---

# --- [ย้ายมาตรงนี้] Start the scheduler when the module is imported ---
# Ensure it runs only once, especially in development with auto-reloading
# [FIX #1] ตอนนี้ 'os' ถูก import แล้ว บรรทัดนี้จะทำงานได้
if os.environ.get('WERKZEUG_RUN_MAIN') != 'true': # Check added for local dev server
    start_scheduler()

# --- Run App (For Local Development Only) ---
if __name__ == '__main__':
    # Run the Flask development server (for local testing)
    # Gunicorn runs 'server' object directly in production
    app.run(debug=False, use_reloader=False) # Add use_reloader=False