from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import threading
import queue
import time
from datetime import datetime
from ultralytics import YOLO
import os
import logging
from config import Config
from video_processor import VideoProcessor
from notification_system import NotificationSystem

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Настройка базы данных
db = SQLAlchemy(app)

# Модель для хранения событий
class SecurityEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    event_type = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)
    frame_data = db.Column(db.Text)  # Base64 encoded frame
    location = db.Column(db.String(200))
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'confidence': self.confidence,
            'description': self.description,
            'location': self.location
        }

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация YOLO модели
model = YOLO(app.config['YOLO_MODEL'])

# Инициализация системы уведомлений
notification_system = NotificationSystem(app.config)

# Инициализация процессора видео
video_processor = VideoProcessor(model, app.config, notification_system)

# Утилита для приведения данных к JSON-совместимому виду
def _to_jsonable(obj):
    try:
        import numpy as np
    except Exception:
        np = None
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    # fallback
    try:
        return str(obj)
    except Exception:
        return None

# Функция для сохранения события в базу данных
def save_security_event(event_data, frame_data=None):
    """Сохраняет событие безопасности в базу данных"""
    try:
        security_event = SecurityEvent(
            event_type=event_data['type'],
            confidence=event_data['confidence'],
            description=event_data['description'],
            frame_data=frame_data,
            location='Camera 1'  # Можно настроить
        )
        
        db.session.add(security_event)
        db.session.commit()
        
        logger.info(f"Событие сохранено в БД: {event_data['type']}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении события: {e}")
        return False

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """Запускает анализ видеопотока"""
    try:
        if video_processor.start_processing():
            return jsonify({'status': 'started', 'message': 'Анализ запущен'})
        else:
            return jsonify({'status': 'already_running', 'message': 'Анализ уже запущен'})
    except Exception as e:
        logger.error(f"Ошибка при запуске анализа: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis():
    """Останавливает анализ видеопотока"""
    try:
        video_processor.stop_processing()
        return jsonify({'status': 'stopped', 'message': 'Анализ остановлен'})
    except Exception as e:
        logger.error(f"Ошибка при остановке анализа: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/upload_frame', methods=['POST'])
def upload_frame():
    """Принимает кадр для анализа"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if frame_data and video_processor.is_processing:
            if video_processor.add_frame(frame_data):
                return jsonify({'status': 'success', 'message': 'Кадр принят'})
            else:
                return jsonify({'status': 'error', 'message': 'Очередь переполнена'})
        else:
            return jsonify({'status': 'error', 'message': 'Анализ не запущен или кадр не получен'})
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке кадра: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get_results')
def get_results():
    """Получает результаты анализа"""
    try:
        result = video_processor.get_result()
        if result:
            # Сохраняем подозрительные события в базу данных
            for event in result.get('suspicious_events', []):
                save_security_event(event, frame_data=event.get('frame_b64'))
            
            safe = _to_jsonable(result)
            return jsonify({'status': 'success', 'data': safe})
        else:
            return jsonify({'status': 'no_data', 'message': 'Нет новых результатов'})
    except Exception as e:
        logger.error(f"Ошибка при получении результатов: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/events')
def get_events():
    """Получает список событий безопасности"""
    try:
        events = SecurityEvent.query.order_by(SecurityEvent.timestamp.desc()).limit(50).all()
        return jsonify({
            'status': 'success',
            'events': [event.to_dict() for event in events]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/clear_events', methods=['POST'])
def clear_events():
    """Очищает все события безопасности"""
    try:
        num_deleted = db.session.query(SecurityEvent).delete()
        db.session.commit()
        logger.info(f"Очищено событий: {num_deleted}")
        return jsonify({'status': 'success', 'deleted': num_deleted})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Ошибка при очистке событий: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/event/<int:event_id>/frame')
def get_event_frame(event_id):
    """Получает кадр события"""
    try:
        event = SecurityEvent.query.get_or_404(event_id)
        if event.frame_data:
            return jsonify({
                'status': 'success',
                'frame_data': event.frame_data
            })
        else:
            return jsonify({'status': 'error', 'message': 'Кадр не найден'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/status')
def get_status():
    """Получает статус системы"""
    try:
        stats = video_processor.get_stats()
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        logger.error(f"Ошибка при получении статуса: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/metrics/charts')
def get_metrics_charts():
    """Получает графики метрик модели"""
    try:
        charts = video_processor.get_metrics_charts()
        return jsonify({
            'status': 'success',
            'charts': charts
        })
    except Exception as e:
        logger.error(f"Ошибка при генерации графиков: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/metrics/report')
def get_metrics_report():
    """Получает отчет по метрикам"""
    try:
        report = video_processor.export_metrics_report()
        return jsonify({
            'status': 'success',
            'report': report
        })
    except Exception as e:
        logger.error(f"Ошибка при генерации отчета: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/metrics/current')
def get_current_metrics():
    """Получает текущие метрики"""
    try:
        metrics = video_processor.metrics_analyzer.get_current_metrics()
        return jsonify({
            'status': 'success',
            'metrics': _to_jsonable(metrics)
        })
    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("Запуск системы анализа видеопотока...")
    print("Откройте http://localhost:5000 в браузере")
    app.run(debug=True, host='0.0.0.0', port=5000)
