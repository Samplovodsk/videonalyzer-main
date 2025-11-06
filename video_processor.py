import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
from collections import deque
import logging
import base64
from metrics_analyzer import ModelMetricsAnalyzer

class VideoProcessor:
    """Класс для обработки видеопотока в реальном времени"""
    
    def __init__(self, model, config, notification_system):
        self.model = model
        self.config = config
        self.notification_system = notification_system
        
        # Очереди
        self.input_queue = queue.Queue(maxsize=config['MAX_QUEUE_SIZE'])
        self.output_queue = queue.Queue()
        
        # Состояние
        self.is_processing = False
        self.frame_count = 0
        self.event_count = 0
        
        # История для анализа движения
        self.frame_history = deque(maxlen=10)  # Храним последние 10 кадров
        self.detection_history = deque(maxlen=30)  # История детекций
        
        # Система метрик
        self.metrics_analyzer = ModelMetricsAnalyzer()
        
        # Статистика
        self.stats = {
            'frames_processed': 0,
            'detections_total': 0,
            'suspicious_events': 0,
            'processing_time': 0,
            'fps': 0
        }
        
        self.logger = logging.getLogger(__name__)
        
    def start_processing(self):
        """Запускает обработку видео"""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Обработка видео запущена")
            return True
        return False
    
    def stop_processing(self):
        """Останавливает обработку видео"""
        self.is_processing = False
        self.logger.info("Обработка видео остановлена")
    
    def add_frame(self, frame_data):
        """Добавляет кадр в очередь обработки"""
        try:
            if not self.input_queue.full():
                self.input_queue.put(frame_data, timeout=1)
                return True
            else:
                self.logger.warning("Очередь обработки переполнена")
                return False
        except queue.Full:
            return False
    
    def get_result(self):
        """Получает результат обработки"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_loop(self):
        """Основной цикл обработки"""
        while self.is_processing:
            try:
                # Получаем кадр из очереди
                frame_data = self.input_queue.get(timeout=1)
                
                # Декодируем кадр
                frame = self._decode_frame(frame_data)
                if frame is None:
                    continue
                
                # Обрабатываем кадр
                start_time = time.time()
                result = self._analyze_frame(frame)
                processing_time = time.time() - start_time
                
                # Обновляем статистику
                self._update_stats(processing_time)
                
                # Вкладываем снимок кадра в каждое событие (для БД/клиента)
                if result['suspicious_events']:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        for ev in result['suspicious_events']:
                            ev['frame_b64'] = frame_b64
                    except Exception as e:
                        self.logger.warning(f"Не удалось прикрепить кадр к событиям: {e}")

                # Сохраняем результат
                self.output_queue.put(result)
                
                # Обрабатываем подозрительные события
                if result['suspicious_events']:
                    self._handle_suspicious_events(result['suspicious_events'], frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Ошибка в цикле обработки: {e}")
                continue
    
    def _decode_frame(self, frame_data):
        """Декодирует кадр из base64"""
        try:
            frame_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            self.logger.error(f"Ошибка декодирования кадра: {e}")
            return None
    
    def _analyze_frame(self, frame):
        """Анализирует кадр на предмет подозрительных действий"""
        # Добавляем кадр в историю
        self.frame_history.append(frame.copy())
        
        # Получаем детекции от YOLO
        results = self.model(frame)
        
        detections = []
        suspicious_events = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем данные детекции
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    detections.append(detection)
                    
                    # Проверяем на подозрительные объекты
                    if (class_name in self.config['SUSPICIOUS_CLASSES'] and 
                        confidence > self.config['CONFIDENCE_THRESHOLD']):
                        
                        # Анализируем поведение
                        behavior_analysis = self._analyze_behavior(frame, detection)
                        if behavior_analysis['is_suspicious']:
                            suspicious_events.append({
                                'type': behavior_analysis['type'],
                                'confidence': confidence,
                                'description': behavior_analysis['description'],
                                'detection': detection,
                                'timestamp': datetime.utcnow().isoformat()
                            })
        
        # Добавляем детекции в историю
        self.detection_history.append(detections)
        
        # Анализируем движение между кадрами
        movement_analysis = self._analyze_movement()
        if movement_analysis['is_suspicious']:
            suspicious_events.append(movement_analysis)
        
        # Добавляем данные в анализатор метрик
        processing_time = time.time() - start_time if 'start_time' in locals() else 0.1
        self.metrics_analyzer.add_prediction(detections, processing_time)
        
        return {
            'detections': detections,
            'suspicious_events': suspicious_events,
            'timestamp': datetime.utcnow().isoformat(),
            'frame_count': self.frame_count,
            'stats': self.stats.copy(),
            'metrics': self.metrics_analyzer.get_current_metrics()
        }
    
    def _analyze_behavior(self, frame, detection):
        """Анализирует поведение обнаруженного объекта"""
        class_name = detection['class']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Анализ размера и позиции
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        frame_height, frame_width = frame.shape[:2]
        
        # Проверка близости к камере
        if class_name == 'person':
            if height > frame_height * self.config['PROXIMITY_THRESHOLD']:
                return {
                    'is_suspicious': True,
                    'type': 'suspicious_proximity',
                    'description': f'Человек слишком близко к камере (размер: {height:.0f}px)',
                    'confidence': confidence
                }
            
            # Проверка на необычное поведение
            if self._check_unusual_behavior(detection):
                return {
                    'is_suspicious': True,
                    'type': 'unusual_behavior',
                    'description': 'Обнаружено необычное поведение человека',
                    'confidence': confidence
                }
        
        # Проверка оружия
        elif class_name in ['knife', 'gun']:
            return {
                'is_suspicious': True,
                'type': 'weapon_detected',
                'description': f'Обнаружено оружие: {class_name}',
                'confidence': confidence
            }
        
        # Проверка подозрительных объектов
        elif (class_name in ['bottle', 'backpack', 'handbag'] and 
              confidence > self.config['SUSPICIOUS_OBJECT_CONFIDENCE']):
            return {
                'is_suspicious': True,
                'type': 'suspicious_object',
                'description': f'Подозрительный объект: {class_name}',
                'confidence': confidence
            }
        
        return {
            'is_suspicious': False,
            'type': 'normal',
            'description': 'Нормальное поведение'
        }
    
    def _check_unusual_behavior(self, detection):
        """Проверяет на необычное поведение на основе истории"""
        if len(self.detection_history) < 5:
            return False
        
        # Простая логика: если человек появляется и исчезает часто
        recent_detections = list(self.detection_history)[-5:]
        person_detections = 0
        
        for detections in recent_detections:
            for det in detections:
                if det['class'] == 'person' and det['confidence'] > 0.5:
                    person_detections += 1
                    break
        
        # Если человек появляется в менее чем 60% кадров - подозрительно
        return person_detections < 3
    
    def _analyze_movement(self):
        """Анализ движения временно отключен (устранение предупреждений OpenCV)"""
        return {'is_suspicious': False}
    
    def _handle_suspicious_events(self, events, frame):
        """Обрабатывает подозрительные события"""
        for event in events:
            self.event_count += 1
            
            # Кодируем кадр для сохранения
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Отправляем уведомление
            self.notification_system.send_alert(event, frame_b64)
            
            self.logger.warning(f"Подозрительное событие: {event['description']}")
    
    def _update_stats(self, processing_time):
        """Обновляет статистику обработки"""
        self.frame_count += 1
        self.stats['frames_processed'] = self.frame_count
        self.stats['processing_time'] = processing_time
        
        # Вычисляем FPS
        if processing_time > 0:
            self.stats['fps'] = 1.0 / processing_time
    
    def get_stats(self):
        """Возвращает статистику обработки"""
        return {
            'is_processing': self.is_processing,
            'queue_size': self.input_queue.qsize(),
            'results_available': not self.output_queue.empty(),
            'frame_count': self.frame_count,
            'event_count': self.event_count,
            'stats': self.stats.copy(),
            'metrics': self.metrics_analyzer.get_current_metrics()
        }
    
    def get_metrics_charts(self):
        """Возвращает графики метрик"""
        return self.metrics_analyzer.generate_all_charts()
    
    def export_metrics_report(self):
        """Экспортирует отчет по метрикам"""
        return self.metrics_analyzer.export_metrics_report()
