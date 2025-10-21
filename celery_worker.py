# celery_worker.py

from celery import Celery
import os

# ใส่ IP ของ WSL ที่หาเจอ
WSL_REDIS_IP = 'redis://localhost:6379/0'

celery = Celery(
    'tasks',  # <--- เปลี่ยนจาก '__main__' เป็น 'tasks'
    broker=os.environ.get('CELERY_BROKER_URL', WSL_REDIS_IP),
    backend=os.environ.get('CELERY_RESULT_BACKEND', WSL_REDIS_IP)
)

celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Bangkok',
    enable_utc=True,
)