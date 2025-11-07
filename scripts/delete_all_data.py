# scripts/delete_all_data.py
import sys
import os
import logging
from sqlalchemy import MetaData
# Setup sys.path to import from app
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# App Imports
try:
    from app import db, server
except ImportError as e:
    logging.error(f"Failed to import app components: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_all_data_and_recreate_schema():
    """
    Deletes all tables defined by SQLAlchemy metadata, and then recreates them.
    This is the safest way to fully reset the database schema and data.
    """
    logging.warning("!!! WARNING: STARTING FULL DATABASE RESET !!!")
    
    with server.app_context():
        try:
            # 1. Drop all tables
            logging.info("1. Dropping all existing tables...")
            db.drop_all()
            logging.info("   All tables dropped successfully.")

            # 2. Recreate tables (This recreates tables based on app/models.py)
            logging.info("2. Recreating the database schema (all tables)...")
            db.create_all()
            logging.info("   Schema recreated successfully.")
            
            logging.info("!!! DATABASE RESET COMPLETE. Data is now empty and schema is fresh. !!!")

        except Exception as e:
            logging.critical(f"FATAL ERROR DURING DATABASE RESET: {e}", exc_info=True)
            db.session.rollback()
            logging.error("Database rollback performed. Reset failed.")
            sys.exit(1)

if __name__ == "__main__":
    delete_all_data_and_recreate_schema()