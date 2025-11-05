# 🤖 Ресурсы для улучшения модели машинного обучения

## 📊 Дополнительные модели YOLO

### Более точные модели
```python
# В config.py можно изменить модель на более точную
YOLO_MODEL = 'yolov8s.pt'  # Средняя точность, быстрая
YOLO_MODEL = 'yolov8m.pt'  # Высокая точность, средняя скорость
YOLO_MODEL = 'yolov8l.pt'  # Очень высокая точность, медленная
YOLO_MODEL = 'yolov8x.pt'  # Максимальная точность, очень медленная
```

### Специализированные модели
```python
# Детекция людей
YOLO_MODEL = 'yolov8n-pose.pt'  # Детекция поз человека

# Детекция лиц
YOLO_MODEL = 'yolov8n-face.pt'  # Детекция лиц
```

## 🔧 Установка дополнительных библиотек

### Ускорение обработки
```bash
# CUDA для GPU ускорения
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorRT для NVIDIA GPU
pip install tensorrt

# OpenVINO для Intel CPU/GPU
pip install openvino
```

### Дополнительные модели
```bash
# Detectron2 от Facebook
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# MediaPipe от Google
pip install mediapipe

# YOLOv5 (альтернатива)
pip install ultralytics
```

### Обработка видео
```bash
# FFmpeg для работы с видео
pip install ffmpeg-python

# VideoIO для работы с видеофайлами
pip install imageio[ffmpeg]

# OpenCV с дополнительными модулями
pip install opencv-contrib-python
```

## 📚 Обучение собственной модели

### 1. Подготовка данных

#### Создание датасета
```python
# Пример создания датасета для детекции подозрительных действий
import cv2
import json
import os

def create_dataset():
    # Создаем структуру папок
    os.makedirs('dataset/images', exist_ok=True)
    os.makedirs('dataset/labels', exist_ok=True)
    
    # Аннотации в формате YOLO
    annotations = {
        'person': 0,
        'knife': 1,
        'gun': 2,
        'bottle': 3,
        'suspicious_behavior': 4
    }
    
    return annotations
```

#### Аннотирование данных
```python
# Используйте инструменты для аннотирования:
# - LabelImg: https://github.com/tzutalin/labelImg
# - CVAT: https://github.com/openvinotoolkit/cvat
# - Roboflow: https://roboflow.com/
```

### 2. Обучение модели

#### Обучение YOLO
```python
from ultralytics import YOLO

# Загружаем предобученную модель
model = YOLO('yolov8n.pt')

# Обучаем на своих данных
results = model.train(
    data='dataset.yaml',  # Файл конфигурации датасета
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'  # Используем GPU если доступен
)

# Сохраняем обученную модель
model.save('custom_model.pt')
```

#### Конфигурация датасета (dataset.yaml)
```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 5  # Количество классов
names: ['person', 'knife', 'gun', 'bottle', 'suspicious_behavior']
```

### 3. Файн-тюнинг модели

#### Адаптация под конкретные условия
```python
# Настройка параметров обучения
model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    lr0=0.01,  # Начальная скорость обучения
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=2.0,
    label_smoothing=0.0,
    nbs=64,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    plots=True
)
```

## 🎯 Улучшение точности детекции

### 1. Предобработка данных
```python
# В video_processor.py добавить предобработку
def preprocess_frame(frame):
    # Улучшение контраста
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    
    # Шумоподавление
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # Увеличение резкости
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    return frame
```

### 2. Постобработка результатов
```python
# Фильтрация ложных срабатываний
def filter_detections(detections, confidence_threshold=0.5):
    filtered = []
    for detection in detections:
        if detection['confidence'] > confidence_threshold:
            # Дополнительные проверки
            if detection['class'] == 'person':
                # Проверяем размер человека
                bbox = detection['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if height > 50:  # Минимальный размер человека
                    filtered.append(detection)
            else:
                filtered.append(detection)
    return filtered
```

### 3. Трекинг объектов
```python
# Добавление трекинга для улучшения стабильности
from collections import defaultdict
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.tracks = defaultdict(list)
        self.next_id = 0
    
    def update(self, detections):
        # Простой трекинг по IoU
        for detection in detections:
            # Логика трекинга
            pass
        return tracked_objects
```

## 📊 Дополнительные датасеты

### Публичные датасеты
```python
# COCO Dataset - 80 классов объектов
# Скачать: https://cocodataset.org/

# Open Images Dataset - 600 классов
# Скачать: https://storage.googleapis.com/openimages/web/index.html

# Pascal VOC - 20 классов
# Скачать: http://host.robots.ox.ac.uk/pascal/VOC/

# Custom Security Dataset
# Создать собственный датасет для детекции нарушений
```

### Специализированные датасеты
```python
# Детекция оружия
# - Gun Detection Dataset
# - Weapon Detection Dataset

# Детекция подозрительного поведения
# - Suspicious Activity Detection Dataset
# - Anomaly Detection Dataset
```

## 🚀 Облачные ресурсы

### Google Colab
```python
# Бесплатный GPU для обучения
# https://colab.research.google.com/

# Пример использования в Colab
!pip install ultralytics
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='dataset.yaml', epochs=100)
```

### AWS SageMaker
```python
# Облачное обучение моделей
# https://aws.amazon.com/sagemaker/

# Пример конфигурации
training_config = {
    'instance_type': 'ml.p3.2xlarge',
    'framework': 'pytorch',
    'framework_version': '1.12.1',
    'py_version': 'py38'
}
```

### Microsoft Azure ML
```python
# Облачная платформа для ML
# https://azure.microsoft.com/en-us/products/machine-learning

# Пример использования
from azureml.core import Workspace, Experiment
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='video-analysis')
```

## 🔧 Оптимизация производительности

### 1. Квантизация модели
```python
# Уменьшение размера модели
import torch
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# Квантизация для ускорения
model.quantize()
```

### 2. Оптимизация для мобильных устройств
```python
# Конвертация в ONNX
model.export(format='onnx')

# Конвертация в TensorFlow Lite
model.export(format='tflite')
```

### 3. Параллельная обработка
```python
# Использование нескольких GPU
model.train(data='dataset.yaml', device=[0, 1, 2, 3])

# Многопроцессорная обработка
from multiprocessing import Pool
import cv2

def process_frame(frame):
    # Обработка кадра
    return processed_frame

with Pool(4) as p:
    results = p.map(process_frame, frames)
```

## 📈 Метрики и оценка

### Метрики качества
```python
# mAP (mean Average Precision)
# Precision и Recall
# F1-Score
# IoU (Intersection over Union)

def calculate_metrics(predictions, ground_truth):
    # Расчет метрик
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score
```

### Валидация модели
```python
# Кросс-валидация
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(dataset):
    # Обучение и валидация
    pass
```

## 🎯 Рекомендации по улучшению

### 1. Для повышения точности
- Используйте более точные модели (yolov8m, yolov8l)
- Увеличьте размер датасета
- Добавьте аугментацию данных
- Используйте ансамбли моделей

### 2. Для повышения скорости
- Используйте более быстрые модели (yolov8n)
- Оптимизируйте размер входного изображения
- Используйте GPU ускорение
- Примените квантизацию модели

### 3. Для специфических задач
- Обучите модель на своих данных
- Добавьте специализированные классы
- Используйте transfer learning
- Примените domain adaptation

## 📚 Полезные ресурсы

### Документация
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Туториалы
- [YOLO Tutorial](https://github.com/ultralytics/yolov5)
- [Computer Vision Tutorials](https://opencv-python-tutroals.readthedocs.io/)
- [Deep Learning for Computer Vision](https://cs231n.stanford.edu/)

### Сообщества
- [YOLO Community](https://github.com/ultralytics/yolov5/discussions)
- [OpenCV Community](https://opencv.org/)
- [PyTorch Community](https://discuss.pytorch.org/)
