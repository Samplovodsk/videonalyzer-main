import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class Config:
    """Конфигурация приложения"""
    
    # Основные настройки Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Настройки базы данных
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///video_analysis.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Настройки модели YOLO
    YOLO_MODEL = os.environ.get('YOLO_MODEL') or 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.5'))
    
    # Настройки обработки видео
    FRAME_RATE = int(os.environ.get('FRAME_RATE', '1'))
    MAX_QUEUE_SIZE = int(os.environ.get('MAX_QUEUE_SIZE', '100'))
    
    # Настройки уведомлений
    ENABLE_EMAIL_ALERTS = os.environ.get('ENABLE_EMAIL_ALERTS', 'False').lower() == 'true'
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    EMAIL_USER = os.environ.get('EMAIL_USER', '')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
    
    # Подозрительные классы объектов
    SUSPICIOUS_CLASSES = {
        'person': 0,
        'knife': 43,
        'gun': 44,
        'bottle': 39,
        'cell phone': 67,
        'laptop': 63,
        'backpack': 24,
        'handbag': 26
    }
    
    # Пороги для определения подозрительного поведения
    PROXIMITY_THRESHOLD = 0.7  # Если объект занимает более 70% кадра
    WEAPON_CONFIDENCE_THRESHOLD = 0.6  # Минимальная уверенность для оружия
    SUSPICIOUS_OBJECT_CONFIDENCE = 0.7  # Минимальная уверенность для подозрительных объектов
