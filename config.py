# config.py

import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-fallback-secret-key-for-development'
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

    # ใส่ IP ของ WSL ที่หาเจอ
    WSL_REDIS_IP = 'redis://192.168.241.65:6379/0'

    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or WSL_REDIS_IP
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or WSL_REDIS_IP