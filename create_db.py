# create_db.py
from app import server, db

with server.app_context():
    db.create_all()

print("Database tables created successfully.")