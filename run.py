#!/usr/bin/env python3
"""
Скрипт для запуска системы анализа видеопотока
"""

import os
import sys
import logging
from app import app, db

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Проверка зависимостей"""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        from flask import Flask
        print("[OK] Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"[ERROR] Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False

def initialize_database():
    """Инициализация базы данных"""
    try:
        with app.app_context():
            db.create_all()
        print("[OK] База данных инициализирована")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации БД: {e}")
        return False

def main():
    """Основная функция"""
    print("Запуск системы анализа видеопотока...")
    print("=" * 50)
    
    # Настройка логирования
    setup_logging()
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # Инициализация базы данных
    if not initialize_database():
        sys.exit(1)
    
    print("\nИнформация о системе:")
    print(f"   - Модель YOLO: {app.config['YOLO_MODEL']}")
    print(f"   - Порог уверенности: {app.config['CONFIDENCE_THRESHOLD']}")
    print(f"   - База данных: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"   - Email уведомления: {'Включены' if app.config['ENABLE_EMAIL_ALERTS'] else 'Отключены'}")
    
    print("\nВеб-интерфейс доступен по адресу:")
    print("   http://localhost:5000")
    print("\nAPI endpoints:")
    print("   - POST /api/start_analysis - Запуск анализа")
    print("   - POST /api/stop_analysis - Остановка анализа")
    print("   - POST /api/upload_frame - Загрузка кадра")
    print("   - GET  /api/get_results - Получение результатов")
    print("   - GET  /api/events - Список событий")
    print("   - GET  /api/status - Статус системы")
    
    print("\n" + "=" * 50)
    print("Система готова к работе!")
    print("Нажмите Ctrl+C для остановки")
    print("=" * 50)
    
    try:
        # Запуск приложения
        app.run(
            debug=app.config['DEBUG'],
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] Система остановлена пользователем")
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
