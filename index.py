# index.py (UPDATED - Main Entry Point)

from dash import dcc, html
from flask import redirect
from flask_login import logout_user
import logging # <-- [NEW] เพิ่ม logging

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
# --- [END NEW] ---

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

# --- [NEW] Function to start the ETL Scheduler ---
def start_scheduler():
    scheduler = BackgroundScheduler(daemon=True)
    
    # สั่งให้ Job 1 (update_company_summaries) ทำงานทุกวัน
    # เวลาตี 1 (01:00) ตามเวลา UTC
    scheduler.add_job(
        update_company_summaries, 
        'cron', 
        hour=1, 
        minute=0, 
        misfire_grace_time=3600 # ถ้าพลาด ให้ลองใหม่ภายใน 1 ชม.
    )
    
    # (ในอนาคต เราสามารถเพิ่ม Job 2 ที่นี่)
    # scheduler.add_job(update_daily_prices, 'cron', day_of_week='sat', hour=3)
    
    scheduler.start()
    logging.info("APScheduler has started... ETL jobs are scheduled.")
# --- [END NEW] ---


# --- Run App ---
if __name__ == '__main__':
    with server.app_context():
        # db.create_all() # <-- [REMOVED] เราใช้ flask db upgrade แล้ว ไม่จำเป็นต้องใช้ create_all() อีก
        pass # เรายังคงต้องการ app_context เพื่อสตาร์ท scheduler

    # --- [NEW] สั่งให้ Scheduler เริ่มทำงาน ---
    start_scheduler() 
    # --- [END NEW] ---
    
    # Run this file to start the app: python index.py
    app.run(debug=False)