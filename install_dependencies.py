#!/usr/bin/env python3
"""
Скрипт для установки зависимостей системы анализа видеопотока
"""

import subprocess
import sys
import os

def install_package(package):
    """Устанавливает пакет через pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Проверяет, установлен ли пакет"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Основная функция установки"""
    print("🔧 Установка зависимостей для системы анализа видеопотока")
    print("=" * 60)
    
    # Список основных зависимостей
    packages = [
        ("flask", "Flask==2.3.3"),
        ("flask_cors", "Flask-CORS==4.0.0"),
        ("cv2", "opencv-python==4.8.1.78"),
        ("ultralytics", "ultralytics==8.0.196"),
        ("numpy", "numpy==1.24.3"),
        ("PIL", "Pillow==10.0.1"),
        ("torch", "torch==2.0.1"),
        ("torchvision", "torchvision==0.15.2"),
        ("sqlalchemy", "SQLAlchemy==2.0.21"),
        ("flask_sqlalchemy", "Flask-SQLAlchemy==3.0.5"),
        ("dotenv", "python-dotenv==1.0.0"),
        ("requests", "requests==2.31.0")
    ]
    
    print("Проверка и установка пакетов...")
    print()
    
    failed_packages = []
    
    for import_name, package_name in packages:
        print(f"📦 {package_name}...", end=" ")
        
        if check_package(import_name):
            print("✅ Уже установлен")
        else:
            print("⬇️ Устанавливается...", end=" ")
            if install_package(package_name):
                print("✅ Установлен")
            else:
                print("❌ Ошибка установки")
                failed_packages.append(package_name)
    
    print("\n" + "=" * 60)
    
    if failed_packages:
        print("❌ Не удалось установить следующие пакеты:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\nПопробуйте установить их вручную:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False
    else:
        print("✅ Все зависимости успешно установлены!")
        print("\nТеперь вы можете запустить систему:")
        print("   python run.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
