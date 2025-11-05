# 🔧 Инструкции по установке

## 📋 Быстрая установка

### 1. Установка Python зависимостей
```bash
# Автоматическая установка всех зависимостей
python install_dependencies.py

# Или вручную
pip install -r requirements.txt
```

### 2. Запуск системы
```bash
# Запуск с проверками
python run.py

# Или напрямую
python app.py
```

## 🐍 Установка Python

### Windows
1. Скачайте Python с [python.org](https://www.python.org/downloads/)
2. Убедитесь, что отмечена опция "Add Python to PATH"
3. Установите Python 3.8 или выше

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.8 python3.8-pip python3.8-venv
```

### macOS
```bash
# Установите Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установите Python
brew install python@3.8
```

## 📦 Установка зависимостей

### Основные зависимости
```bash
# Flask для веб-сервера
pip install Flask==2.3.3 Flask-CORS==4.0.0

# OpenCV для обработки видео
pip install opencv-python==4.8.1.78

# YOLO для детекции объектов
pip install ultralytics==8.0.196

# NumPy для работы с массивами
pip install numpy==1.24.3

# Pillow для работы с изображениями
pip install Pillow==10.0.1

# PyTorch для машинного обучения
pip install torch==2.0.1 torchvision==0.15.2

# SQLAlchemy для базы данных
pip install SQLAlchemy==2.0.21 Flask-SQLAlchemy==3.0.5

# Дополнительные утилиты
pip install python-dotenv==1.0.0 requests==2.31.0
```

### GPU ускорение (опционально)
```bash
# CUDA для NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Проверка CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🚀 Первый запуск

### 1. Проверка установки
```bash
# Проверка Python
python --version

# Проверка зависимостей
python -c "import cv2, numpy, flask, ultralytics; print('✅ Все зависимости установлены')"
```

### 2. Запуск системы
```bash
# Запуск с автоматическими проверками
python run.py
```

### 3. Открытие в браузере
Перейдите по адресу: **http://localhost:5000**

## 🔧 Настройка

### Конфигурация системы
Отредактируйте файл `config.py`:
```python
# Основные настройки
YOLO_MODEL = 'yolov8n.pt'  # Модель YOLO
CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности
PROXIMITY_THRESHOLD = 0.7   # Порог близости

# Настройки уведомлений
ENABLE_EMAIL_ALERTS = True
EMAIL_USER = 'your-email@gmail.com'
EMAIL_PASSWORD = 'your-app-password'
```

### Настройка email уведомлений
1. Включите двухфакторную аутентификацию в Gmail
2. Создайте пароль приложения
3. Укажите данные в `config.py`

## 🐳 Docker установка

### Создание Dockerfile
```dockerfile
FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["python", "run.py"]
```

### Запуск с Docker
```bash
# Сборка образа
docker build -t videoanalyzer .

# Запуск контейнера
docker run -p 5000:5000 videoanalyzer
```

## 🧪 Тестирование

### Запуск тестов
```bash
# Все тесты
python test_system.py

# Примеры API
python api_examples.py

# Проверка камеры
python -c "import cv2; print('Камера доступна:', cv2.VideoCapture(0).isOpened())"
```

## 🆘 Устранение неполадок

### Проблемы с камерой
```bash
# Проверка доступных камер
python -c "import cv2; cap = cv2.VideoCapture(0); print('Камера 0:', cap.isOpened())"

# Проверка разрешений браузера
# Убедитесь, что браузер имеет доступ к камере
```

### Проблемы с YOLO
```bash
# Переустановка ultralytics
pip uninstall ultralytics
pip install ultralytics

# Проверка модели
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Модель загружена')"
```

### Проблемы с OpenCV
```bash
# Переустановка OpenCV
pip uninstall opencv-python
pip install opencv-python

# Установка дополнительных компонентов
pip install opencv-contrib-python
```

### Проблемы с PyTorch
```bash
# Переустановка PyTorch
pip uninstall torch torchvision
pip install torch torchvision

# Проверка CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 📊 Системные требования

### Минимальные требования
- **OS**: Windows 10, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8+
- **RAM**: 4GB
- **CPU**: 2 ядра
- **Место**: 2GB

### Рекомендуемые требования
- **RAM**: 8GB+
- **CPU**: 4+ ядра
- **GPU**: NVIDIA GPU с CUDA
- **SSD**: Для быстрой работы

## 🔄 Обновление

### Обновление зависимостей
```bash
# Обновление всех пакетов
pip install --upgrade -r requirements.txt

# Обновление конкретного пакета
pip install --upgrade ultralytics
```

### Обновление модели YOLO
```bash
# Загрузка новой модели
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

## 📞 Поддержка

Если у вас возникли проблемы:

1. Проверьте логи в консоли
2. Запустите тесты: `python test_system.py`
3. Проверьте системные требования
4. Обратитесь к документации в `README.md`
