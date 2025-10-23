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
