# 🚀 Руководство по развертыванию

## 📋 Требования к системе

### Минимальные требования
- **ОС**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 или выше
- **RAM**: 4GB (рекомендуется 8GB+)
- **CPU**: 2 ядра (рекомендуется 4+)
- **Место на диске**: 2GB свободного места

### Рекомендуемые требования
- **GPU**: NVIDIA GPU с поддержкой CUDA (для ускорения YOLO)
- **RAM**: 16GB+
- **CPU**: 8+ ядер
- **SSD**: Для быстрой работы базы данных

## 🔧 Установка зависимостей

### 1. Установка Python

#### Windows
```bash
# Скачайте Python с официального сайта
# https://www.python.org/downloads/
# Убедитесь, что отмечена опция "Add Python to PATH"
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.8 python3.8-pip python3.8-venv
```

#### macOS
```bash
# Установите Homebrew, если не установлен
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установите Python
brew install python@3.8
```

### 2. Создание виртуального окружения

```bash
# Создайте виртуальное окружение
python -m venv videoanalyzer_env

# Активируйте окружение
# Windows:
videoanalyzer_env\Scripts\activate
# Linux/macOS:
source videoanalyzer_env/bin/activate
```

### 3. Установка зависимостей

```bash
# Обновите pip
pip install --upgrade pip

# Установите зависимости
pip install -r requirements.txt
```

### 4. Установка CUDA (опционально, для GPU)

#### Windows
```bash
# Скачайте CUDA Toolkit с сайта NVIDIA
# https://developer.nvidia.com/cuda-downloads
# Установите PyTorch с поддержкой CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Ubuntu
```bash
# Установите CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repository-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repository-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Установите PyTorch с поддержкой CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## ⚙️ Конфигурация

### 1. Настройка config.py

```python
# Основные настройки
YOLO_MODEL = 'yolov8n.pt'  # или yolov8s.pt, yolov8m.pt для лучшей точности
CONFIDENCE_THRESHOLD = 0.5
PROXIMITY_THRESHOLD = 0.7

# Настройки производительности
FRAME_RATE = 1  # Кадров в секунду
MAX_QUEUE_SIZE = 100

# Настройки уведомлений
ENABLE_EMAIL_ALERTS = True
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USER = 'your-email@gmail.com'
EMAIL_PASSWORD = 'your-app-password'
```

### 2. Настройка email уведомлений

#### Gmail
1. Включите двухфакторную аутентификацию
2. Создайте пароль приложения:
   - Перейдите в настройки Google аккаунта
   - Безопасность → Пароли приложений
   - Создайте пароль для "Почта"
3. Используйте этот пароль в конфигурации

#### Другие провайдеры
```python
# Outlook/Hotmail
SMTP_SERVER = 'smtp-mail.outlook.com'
SMTP_PORT = 587

# Yahoo
SMTP_SERVER = 'smtp.mail.yahoo.com'
SMTP_PORT = 587
```

## 🚀 Запуск системы

### 1. Первый запуск

```bash
# Запустите систему
python run.py
```

### 2. Проверка работоспособности

```bash
# Запустите тесты
python test_system.py

# Проверьте API
python api_examples.py
```

### 3. Веб-интерфейс

Откройте браузер и перейдите по адресу: **http://localhost:5000**

## 🐳 Развертывание с Docker

### 1. Создание Dockerfile

```dockerfile
FROM python:3.8-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . /app
WORKDIR /app

# Открытие порта
EXPOSE 5000

# Запуск приложения
CMD ["python", "run.py"]
```

### 2. Создание docker-compose.yml

```yaml
version: '3.8'

services:
  videoanalyzer:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - ENABLE_EMAIL_ALERTS=true
      - EMAIL_USER=your-email@gmail.com
      - EMAIL_PASSWORD=your-password
    restart: unless-stopped
```

### 3. Запуск с Docker

```bash
# Сборка и запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## ☁️ Развертывание в облаке

### AWS EC2

1. **Создание инстанса**:
   - Выберите Ubuntu 20.04 LTS
   - Минимум t3.medium (2 vCPU, 4GB RAM)
   - Для GPU: p3.2xlarge или выше

2. **Настройка безопасности**:
   ```bash
   # Откройте порт 5000
   aws ec2 authorize-security-group-ingress \
     --group-id sg-xxxxxxxxx \
     --protocol tcp \
     --port 5000 \
     --cidr 0.0.0.0/0
   ```

3. **Установка на сервере**:
   ```bash
   # Подключитесь к серверу
   ssh -i your-key.pem ubuntu@your-server-ip
   
   # Клонируйте репозиторий
   git clone <your-repo-url>
   cd videoanalyzer
   
   # Установите зависимости
   sudo apt update
   sudo apt install python3.8 python3.8-pip
   pip3 install -r requirements.txt
   
   # Запустите приложение
   python3 run.py
   ```

### Google Cloud Platform

1. **Создание VM**:
   ```bash
   gcloud compute instances create videoanalyzer \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --machine-type=e2-medium \
     --zone=us-central1-a
   ```

2. **Настройка файрвола**:
   ```bash
   gcloud compute firewall-rules create allow-videoanalyzer \
     --allow tcp:5000 \
     --source-ranges 0.0.0.0/0
   ```

### Azure

1. **Создание VM**:
   ```bash
   az vm create \
     --resource-group myResourceGroup \
     --name videoanalyzer \
     --image UbuntuLTS \
     --size Standard_B2s \
     --admin-username azureuser
   ```

2. **Открытие порта**:
   ```bash
   az vm open-port \
     --resource-group myResourceGroup \
     --name videoanalyzer \
     --port 5000
   ```

## 🔒 Безопасность

### 1. Настройка HTTPS

```python
# В app.py добавьте SSL
if __name__ == '__main__':
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        ssl_context=('cert.pem', 'key.pem')
    )
```

### 2. Аутентификация

```python
# Добавьте базовую аутентификацию
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == 'admin' and password == 'secret'

@app.route('/api/start_analysis', methods=['POST'])
@auth.login_required
def start_analysis():
    # ... существующий код
```

### 3. Ограничение доступа

```python
# Ограничьте доступ по IP
from flask import request

@app.before_request
def limit_remote_addr():
    allowed_ips = ['127.0.0.1', '192.168.1.0/24']
    if request.remote_addr not in allowed_ips:
        abort(403)
```

## 📊 Мониторинг

### 1. Логирование

```python
# Настройте детальное логирование
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_analysis.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Метрики

```python
# Добавьте сбор метрик
from prometheus_client import Counter, Histogram, generate_latest

frames_processed = Counter('frames_processed_total', 'Total processed frames')
processing_time = Histogram('frame_processing_seconds', 'Frame processing time')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### 3. Health Check

```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })
```

## 🔧 Обслуживание

### 1. Резервное копирование

```bash
# Создайте скрипт резервного копирования
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp video_analysis.db "backup_${DATE}.db"
```

### 2. Очистка логов

```bash
# Очистите старые логи
find /app/logs -name "*.log" -mtime +30 -delete
```

### 3. Обновление модели

```bash
# Обновите YOLO модель
pip install --upgrade ultralytics
```

## 🆘 Устранение неполадок

### Частые проблемы

1. **Ошибка "No module named 'cv2'"**:
   ```bash
   pip install opencv-python
   ```

2. **Ошибка "CUDA out of memory"**:
   - Уменьшите размер кадра
   - Используйте CPU вместо GPU
   - Увеличьте интервал между кадрами

3. **Медленная работа**:
   - Используйте GPU
   - Уменьшите разрешение камеры
   - Оптимизируйте настройки YOLO

4. **Проблемы с камерой**:
   ```bash
   # Проверьте доступные камеры
   python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

### Логи и диагностика

```bash
# Просмотр логов
tail -f video_analysis.log

# Проверка статуса системы
curl http://localhost:5000/api/status

# Тестирование API
python api_examples.py
```
