# app/__init__.py

import os
import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from .config import Config  # <-- [CHANGED] แก้เป็น Relative Import

# --- App Initialization ---
server = Flask(__name__)
server.config.from_object(Config)

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # [NEW] บอก Dash ให้หาโฟลเดอร์ assets ที่นี่
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
)

# --- Database & Login Manager Setup ---
# [CHANGED] แก้ไข Path ให้ชี้ไปที่ instance folder นอกโฟลเดอร์ app
instance_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)
migrate = Migrate(server, db) 

# --- [NEW] Database Models ---
# Import models *หลังจาก* ที่ db ถูกสร้างแล้ว
from . import models 

# --- [NEW] User Loader ---
# ย้าย User Loader มาไว้ที่นี่
@login_manager.user_loader
def load_user(user_id):
    with server.app_context():
        # อ้างอิงผ่าน models.User
        return db.session.get(models.User, int(user_id))