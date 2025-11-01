import os
from dotenv import load_dotenv

# --- [FIXED] ---
# 1. หาตำแหน่งของ Root Directory (หนึ่งระดับบนจากไฟล์ config.py นี้)
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
# 2. สร้าง Path ไปยังไฟล์ .env ที่อยู่ใน Root
dotenv_path = os.path.join(basedir, '.env')
# 3. สั่งโหลดไฟล์ .env จาก Path นั้น
load_dotenv(dotenv_path) 
# --- [END FIXED] ---

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-fallback-secret-key-for-development'
    
    # ดึงค่าจาก Environment Variable
    DB_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
    
    # หากไม่พบค่า (เช่น อาจจะมีการตั้งชื่อใน GitHub Secrets เป็น DATABASE_URL) 
    # ก็ให้ลองเช็คชื่อตัวแปรมาตรฐาน
    if not DB_URI:
        DB_URI = os.environ.get('DATABASE_URL')
    
    # **สำคัญ: บังคับให้เกิด Error หากไม่พบ Cloud DB URL**
    if not DB_URI:
        raise ValueError("Database connection string (SQLALCHEMY_DATABASE_URI or DATABASE_URL) is not set. Cannot connect to Cloud DB.")

    # กำหนดให้ Flask-SQLAlchemy ใช้ค่าที่ดึงมา (ซึ่งตอนนี้เป็น Cloud DB URL เสมอ)
    SQLALCHEMY_DATABASE_URI = DB_URI
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN') # Assuming you added this to .env