# celery_worker.py
from celery import Celery
import os
from config import Config  # <--- 1. Import Config ของเราเข้ามา

# 2. ไม่ต้อง Hardcode WSL_REDIS_IP ที่นี่แล้ว
#    เราจะใช้ค่าจาก Config object โดยตรง

celery = Celery(
    'tasks',
    # 3. ดึงค่า Broker และ Backend จาก Config object
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND
)

celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Bangkok',
    enable_utc=True,
)
