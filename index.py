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
    # ทำทุกวัน เวลาตี 1 UTC (8 โมงเช้าไทย)
    scheduler.add_job(
        update_company_summaries,
        'cron',
        hour=1,
        minute=0,
        misfire_grace_time=3600, # ถ้าพลาด ให้ลองใหม่ภายใน 1 ชม.
        id='update_summaries_job' # ตั้ง ID ให้ Job
    )
    logging.info("Scheduled job: update_company_summaries (Daily @ 01:00 UTC)")

    # --- Job 2: Update Daily Prices ---
    # ทำทุกวันเสาร์ เวลาตี 3 UTC (10 โมงเช้าไทย)
    # (หรือจะเปลี่ยนเป็น 'cron', hour=2 เพื่อให้ทำงานทุกวันก็ได้ ถ้าต้องการข้อมูลราคาอัปเดตทุกวัน)
    scheduler.add_job(
        update_daily_prices,
        'cron',
        # day_of_week='sat', # ทำเฉพาะวันเสาร์
        hour=2,            # ตี 2 UTC (9 โมงเช้าไทย) - ให้ทำงานหลัง Job 1
        minute=0,
        misfire_grace_time=3600*3, # ถ้าพลาด ให้ลองใหม่ภายใน 3 ชม.
        id='update_prices_job' # ตั้ง ID ให้ Job
    )
    # logging.info("Scheduled job: update_daily_prices (Weekly on Sat @ 03:00 UTC)")
    logging.info("Scheduled job: update_daily_prices (Daily @ 02:00 UTC)")


    # --- Start Scheduler ---
    try:
        scheduler.start()
        logging.info("APScheduler has started successfully. ETL jobs are scheduled.")
    except Exception as e:
         logging.error(f"Failed to start APScheduler: {e}", exc_info=True)
# --- [END MODIFIED] ---


# --- Run App ---
if __name__ == '__main__':
    # ไม่ต้องใช้ db.create_all() แล้ว
    # Start the scheduler when the app starts
    start_scheduler()

    # Run the Flask development server (for local testing)
    # For production (like Render), Gunicorn runs this via 'index:server' in Procfile
    app.run(debug=False) # debug=False สำคัญมากสำหรับ APScheduler