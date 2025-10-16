# config.py
# This file loads configuration variables from the environment for the Flask app.

import os
from dotenv import load_dotenv

# Determine the absolute path of the project's root directory.
basedir = os.path.abspath(os.path.dirname(__file__))

# Load environment variables from the .env file located in the root directory.
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    """
    Set Flask configuration variables from the .env file.
    Provides default values as a fallback.
    """
    # Secret key for signing session cookies and other security-related needs.
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-fallback-secret-key-for-development'

    # Database URI for SQLAlchemy.
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')

    # Disable a feature of SQLAlchemy that is not needed and adds overhead.
    SQLALCHEMY_TRACK_MODIFICATIONS = False