# app.py (FIXED - Core App Initialization with Absolute Path & Celery Integration)

import os
import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from config import Config
from celery_worker import celery # <-- 1. Import celery instance ที่เราสร้าง

# --- [MODIFICATION START] ---
# Get the absolute path of the directory where this file is located
basedir = os.path.abspath(os.path.dirname(__file__))
# Define the absolute path to the assets folder
assets_path = os.path.join(basedir, 'assets')
# --- [MODIFICATION END] ---

# --- App Initialization ---
server = Flask(__name__)
server.config.from_object(Config)

# 2. เชื่อม Flask config เข้ากับ Celery config
#    เพื่อให้ Celery รู้จักค่า CELERY_BROKER_URL จากไฟล์ config.py
#celery.conf.update(server.config)

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # --- [MODIFICATION] Use the absolute path we defined ---
    assets_folder=assets_path
)

# --- Database & Login Manager Setup ---
instance_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)

# --- Database Models ---
# It's better to define models here to avoid circular dependencies
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class UserSelection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol_type = db.Column(db.String(10), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)

class UserAssumptions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    # Forecast Assumptions
    forecast_years = db.Column(db.Integer, default=5)
    eps_growth = db.Column(db.Float, default=10.0)
    terminal_pe = db.Column(db.Float, default=20.0)
    # DCF Assumptions
    dcf_forecast_growth = db.Column(db.Float, default=5.0)
    dcf_perpetual_growth = db.Column(db.Float, default=2.5)
    dcf_wacc = db.Column(db.Float, default=8.0)


@login_manager.user_loader
def load_user(user_id):
    with server.app_context():
        return db.session.get(User, int(user_id))