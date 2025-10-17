# app.py (NEW - Core App Initialization)

import os
import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from config import Config

# --- App Initialization ---
server = Flask(__name__)
server.config.from_object(Config)

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder='assets'
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

@login_manager.user_loader
def load_user(user_id):
    with server.app_context():
        return db.session.get(User, int(user_id))