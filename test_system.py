#!/usr/bin/env python3
"""
Тесты для системы анализа видеопотока
"""

import unittest
import numpy as np
import cv2
import base64
import tempfile
import os
from unittest.mock import Mock, patch
import sys

# Добавляем путь к модулям
sys.path.append('.')

from config import Config
from video_processor import VideoProcessor
from notification_system import NotificationSystem

class TestConfig(unittest.TestCase):
    """Тесты конфигурации"""
    
    def test_config_loading(self):
        """Тест загрузки конфигурации"""
        config = Config()
        
        # Проверяем основные настройки
        self.assertIsNotNone(config.YOLO_MODEL)
        self.assertIsInstance(config.CONFIDENCE_THRESHOLD, float)
        self.assertIsInstance(config.SUSPICIOUS_CLASSES, dict)
        
        # Проверяем подозрительные классы
        self.assertIn('person', config.SUSPICIOUS_CLASSES)
        self.assertIn('knife', config.SUSPICIOUS_CLASSES)

class TestVideoProcessor(unittest.TestCase):
    """Тесты процессора видео"""
    
    def setUp(self):
        """Настройка тестов"""
        self.config = Config()
        self.mock_model = Mock()
        self.mock_notification = Mock()
        self.processor = VideoProcessor(
            self.mock_model, 
            self.config, 
            self.mock_notification
        )
    
    def test_processor_initialization(self):
        """Тест инициализации процессора"""
        self.assertFalse(self.processor.is_processing)
        self.assertEqual(self.processor.frame_count, 0)
        self.assertEqual(self.processor.event_count, 0)
    
    def test_frame_encoding_decoding(self):
        """Тест кодирования и декодирования кадров"""
        # Создаем тестовый кадр
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Кодируем в base64
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Декодируем обратно
        decoded_frame = self.processor._decode_frame(frame_b64)
        
        # Проверяем, что кадр декодировался
        self.assertIsNotNone(decoded_frame)
        self.assertEqual(decoded_frame.shape[:2], test_frame.shape[:2])
    
    def test_behavior_analysis(self):
        """Тест анализа поведения"""
        # Создаем тестовый кадр
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Тест детекции человека
        person_detection = {
            'class': 'person',
            'confidence': 0.8,
            'bbox': [100, 100, 200, 400]  # Высокий человек
        }
        
        result = self.processor._analyze_behavior(test_frame, person_detection)
        self.assertIsInstance(result, dict)
        self.assertIn('is_suspicious', result)
        self.assertIn('type', result)
        self.assertIn('description', result)
        
        # Тест детекции оружия
        weapon_detection = {
            'class': 'knife',
            'confidence': 0.9,
            'bbox': [100, 100, 150, 200]
        }
        
        result = self.processor._analyze_behavior(test_frame, weapon_detection)
        self.assertTrue(result['is_suspicious'])
        self.assertEqual(result['type'], 'weapon_detected')
    
    def test_stats_update(self):
        """Тест обновления статистики"""
        initial_frames = self.processor.frame_count
        
        # Обновляем статистику
        self.processor._update_stats(0.1)  # 0.1 секунды обработки
        
        self.assertEqual(self.processor.frame_count, initial_frames + 1)
        self.assertEqual(self.processor.stats['frames_processed'], initial_frames + 1)
        self.assertEqual(self.processor.stats['processing_time'], 0.1)
        self.assertEqual(self.processor.stats['fps'], 10.0)  # 1/0.1

class TestNotificationSystem(unittest.TestCase):
    """Тесты системы уведомлений"""
    
    def setUp(self):
        """Настройка тестов"""
        self.config = Config()
        self.config['ENABLE_EMAIL_ALERTS'] = False  # Отключаем email для тестов
        self.notification = NotificationSystem(self.config)
    
    def test_notification_initialization(self):
        """Тест инициализации системы уведомлений"""
        self.assertIsNotNone(self.notification.config)
        self.assertIsNotNone(self.notification.logger)
    
    def test_event_logging(self):
        """Тест логирования событий"""
        event_data = {
            'type': 'test_event',
            'description': 'Тестовое событие',
            'confidence': 0.8
        }
        
        # Логирование не должно вызывать исключений
        self.notification.log_event(event_data)
    
    def test_alert_sending(self):
        """Тест отправки уведомлений"""
        event_data = {
            'type': 'test_event',
            'description': 'Тестовое событие',
            'confidence': 0.8
        }
        
        # Отправка уведомления не должна вызывать исключений
        result = self.notification.send_alert(event_data)
        self.assertTrue(result)

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_full_processing_pipeline(self):
        """Тест полного пайплайна обработки"""
        # Создаем тестовый кадр
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Кодируем кадр
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Создаем мок-модель
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.xyxy = [np.array([100, 100, 200, 200])]
        mock_box.conf = [np.array([0.8])]
        mock_box.cls = [np.array([0])]  # person class
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'person'}
        
        # Создаем процессор
        config = Config()
        notification = NotificationSystem(config)
        processor = VideoProcessor(mock_model, config, notification)
        
        # Тестируем анализ кадра
        result = processor._analyze_frame(test_frame)
        
        self.assertIn('detections', result)
        self.assertIn('suspicious_events', result)
        self.assertIn('timestamp', result)
        self.assertIsInstance(result['detections'], list)
        self.assertIsInstance(result['suspicious_events'], list)

class TestPerformance(unittest.TestCase):
    """Тесты производительности"""
    
    def test_frame_processing_speed(self):
        """Тест скорости обработки кадров"""
        import time
        
        # Создаем тестовый кадр
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Создаем мок-модель
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        # Создаем процессор
        config = Config()
        notification = NotificationSystem(config)
        processor = VideoProcessor(mock_model, config, notification)
        
        # Измеряем время обработки
        start_time = time.time()
        processor._analyze_frame(test_frame)
        processing_time = time.time() - start_time
        
        # Обработка должна быть быстрой (менее 1 секунды для мока)
        self.assertLess(processing_time, 1.0)
    
    def test_memory_usage(self):
        """Тест использования памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Создаем много кадров
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Проверяем, что память не растет критически
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Увеличение памяти должно быть разумным (менее 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)

def run_tests():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов системы анализа видеопотока")
    print("=" * 60)
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestVideoProcessor))
    test_suite.addTest(unittest.makeSuite(TestNotificationSystem))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    test_suite.addTest(unittest.makeSuite(TestPerformance))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print(f"Тестов выполнено: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Неудач: {len(result.failures)}")
    
    if result.failures:
        print("\nНеудачные тесты:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nОшибки:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
