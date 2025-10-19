# tasks.py
from celery_worker import celery
from data_handler import get_news_and_sentiment as original_get_news
import logging

@celery.task(bind=True)
def get_news_and_sentiment_task(self, company_name: str):
    """
    นี่คืองาน (Task) ที่ Celery Worker จะทำเบื้องหลัง
    มันจะเรียกใช้ฟังก์ชันดึงข่าวเดิมจาก data_handler.py
    """
    logging.info(f"Starting sentiment task for: {company_name}")
    try:
        # เรียกใช้ฟังก์ชันเดิมที่ทำงานหนักจริงๆ
        result = original_get_news(company_name)
        logging.info(f"Sentiment task for {company_name} completed successfully.")
        return result
    except Exception as e:
        # หากเกิดข้อผิดพลาด Celery จะสามารถบันทึกสถานะ 'FAILURE' ได้
        logging.error(f"Sentiment task for {company_name} failed: {e}", exc_info=True)
        # re-raise the exception so Celery knows it's a failure
        raise e