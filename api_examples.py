#!/usr/bin/env python3
"""
Примеры использования API системы анализа видеопотока
"""

import requests
import base64
import cv2
import time
import json

class VideoAnalysisAPI:
    """Клиент для работы с API системы анализа видеопотока"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def start_analysis(self):
        """Запускает анализ видеопотока"""
        response = self.session.post(f"{self.base_url}/api/start_analysis")
        return response.json()
    
    def stop_analysis(self):
        """Останавливает анализ видеопотока"""
        response = self.session.post(f"{self.base_url}/api/stop_analysis")
        return response.json()
    
    def upload_frame(self, frame_data):
        """Загружает кадр для анализа"""
        data = {"frame": frame_data}
        response = self.session.post(
            f"{self.base_url}/api/upload_frame",
            json=data
        )
        return response.json()
    
    def get_results(self):
        """Получает результаты анализа"""
        response = self.session.get(f"{self.base_url}/api/get_results")
        return response.json()
    
    def get_events(self):
        """Получает список событий безопасности"""
        response = self.session.get(f"{self.base_url}/api/events")
        return response.json()
    
    def get_status(self):
        """Получает статус системы"""
        response = self.session.get(f"{self.base_url}/api/status")
        return response.json()
    
    def get_event_frame(self, event_id):
        """Получает кадр события"""
        response = self.session.get(f"{self.base_url}/api/event/{event_id}/frame")
        return response.json()

def capture_frame_from_camera():
    """Захватывает кадр с веб-камеры"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Кодируем кадр в base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64
    return None

def example_basic_usage():
    """Пример базового использования API"""
    print("🔍 Пример базового использования API")
    print("=" * 50)
    
    api = VideoAnalysisAPI()
    
    # Проверяем статус системы
    print("1. Проверка статуса системы...")
    status = api.get_status()
    print(f"   Статус: {status}")
    
    # Запускаем анализ
    print("\n2. Запуск анализа...")
    start_result = api.start_analysis()
    print(f"   Результат: {start_result}")
    
    # Захватываем и отправляем несколько кадров
    print("\n3. Отправка кадров...")
    for i in range(5):
        frame_data = capture_frame_from_camera()
        if frame_data:
            result = api.upload_frame(frame_data)
            print(f"   Кадр {i+1}: {result['status']}")
            time.sleep(1)
    
    # Получаем результаты
    print("\n4. Получение результатов...")
    for i in range(3):
        results = api.get_results()
        if results['status'] == 'success':
            data = results['data']
            print(f"   Обнаружено объектов: {len(data['detections'])}")
            print(f"   Подозрительных событий: {len(data['suspicious_events'])}")
        time.sleep(2)
    
    # Получаем события
    print("\n5. Получение событий...")
    events = api.get_events()
    if events['status'] == 'success':
        print(f"   Всего событий: {len(events['events'])}")
        for event in events['events'][:3]:  # Показываем первые 3
            print(f"   - {event['event_type']}: {event['description']}")
    
    # Останавливаем анализ
    print("\n6. Остановка анализа...")
    stop_result = api.stop_analysis()
    print(f"   Результат: {stop_result}")

def example_continuous_monitoring():
    """Пример непрерывного мониторинга"""
    print("\n🔄 Пример непрерывного мониторинга")
    print("=" * 50)
    
    api = VideoAnalysisAPI()
    
    # Запускаем анализ
    api.start_analysis()
    print("Анализ запущен. Нажмите Ctrl+C для остановки...")
    
    try:
        while True:
            # Захватываем кадр
            frame_data = capture_frame_from_camera()
            if frame_data:
                # Отправляем кадр
                api.upload_frame(frame_data)
                
                # Получаем результаты
                results = api.get_results()
                if results['status'] == 'success':
                    data = results['data']
                    detections = data['detections']
                    events = data['suspicious_events']
                    
                    if detections:
                        print(f"📊 Обнаружено: {[d['class'] for d in detections]}")
                    
                    if events:
                        for event in events:
                            print(f"🚨 СОБЫТИЕ: {event['description']}")
            
            time.sleep(1)  # Пауза между кадрами
            
    except KeyboardInterrupt:
        print("\nОстановка мониторинга...")
        api.stop_analysis()

def example_event_analysis():
    """Пример анализа событий"""
    print("\n📋 Пример анализа событий")
    print("=" * 50)
    
    api = VideoAnalysisAPI()
    
    # Получаем все события
    events_response = api.get_events()
    
    if events_response['status'] == 'success':
        events = events_response['events']
        
        # Группируем события по типам
        event_types = {}
        for event in events:
            event_type = event['event_type']
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        print("Статистика событий:")
        for event_type, event_list in event_types.items():
            print(f"  {event_type}: {len(event_list)} событий")
            
            # Показываем последнее событие каждого типа
            if event_list:
                latest = event_list[0]  # События отсортированы по времени
                print(f"    Последнее: {latest['description']}")
                print(f"    Время: {latest['timestamp']}")
                print(f"    Уверенность: {latest['confidence']:.2%}")
    
    else:
        print("Ошибка получения событий:", events_response['message'])

def example_custom_detection():
    """Пример настройки детекции"""
    print("\n⚙️ Пример настройки детекции")
    print("=" * 50)
    
    api = VideoAnalysisAPI()
    
    # Запускаем анализ
    api.start_analysis()
    
    # Отправляем кадр с человеком
    frame_data = capture_frame_from_camera()
    if frame_data:
        result = api.upload_frame(frame_data)
        print(f"Кадр отправлен: {result['status']}")
        
        # Ждем обработки
        time.sleep(2)
        
        # Получаем результаты
        results = api.get_results()
        if results['status'] == 'success':
            data = results['data']
            
            print("\nДетальная информация о детекциях:")
            for detection in data['detections']:
                print(f"  Объект: {detection['class']}")
                print(f"  Уверенность: {detection['confidence']:.2%}")
                print(f"  Координаты: {detection['bbox']}")
                print(f"  Время: {detection['timestamp']}")
                print()
            
            print("Подозрительные события:")
            for event in data['suspicious_events']:
                print(f"  Тип: {event['type']}")
                print(f"  Описание: {event['description']}")
                print(f"  Уверенность: {event['confidence']:.2%}")
                print()
    
    api.stop_analysis()

if __name__ == "__main__":
    print("🚀 Примеры использования API системы анализа видеопотока")
    print("Убедитесь, что сервер запущен на http://localhost:5000")
    print()
    
    try:
        # Запускаем примеры
        example_basic_usage()
        example_event_analysis()
        example_custom_detection()
        
        # Непрерывный мониторинг (раскомментируйте для запуска)
        # example_continuous_monitoring()
        
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения к серверу")
        print("Убедитесь, что сервер запущен: python run.py")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
