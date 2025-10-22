# config.py

import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
# Load variables from .env file FIRST
load_dotenv(os.path.join(basedir, '.env')) 

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-fallback-secret-key-for-development'
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN') # Assuming you added this to .env

    # --- UPDATED ---
    # Use the WSL distribution name + .local hostname for Redis
    # This automatically resolves to the correct WSL IP address
    REDIS_HOST = 'redis://localhost:6379/0'

    # Read Celery settings from environment variables FIRST (from .env),
    # If not found in .env, THEN use the WSL_REDIS_HOSTNAME as default.
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or REDIS_HOST
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or REDIS_HOST