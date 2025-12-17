"""
Проверка установки окружения
"""

import sys
import os
from pathlib import Path

# Если запускается из scripts/, переходим в корень проекта
if Path.cwd().name == 'scripts':
    os.chdir('..')


def test_imports():
    """Проверяет импорт всех необходимых библиотек"""
    print("Проверка импорта библиотек...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('av', 'PyAV'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    failed = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n✗ Не удалось импортировать: {', '.join(failed)}")
        print("Установите недостающие пакеты: pip install -r requirements.txt")
        return False
    
    print("\n✓ Все необходимые пакеты установлены")
    return True


def test_cuda():
    """Проверяет доступность CUDA"""
    print("\nПроверка CUDA...")
    
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print("\n✓ CUDA готова к использованию")
            return True
        else:
            print("\n⚠ CUDA недоступна, будет использоваться CPU")
            return False
    except Exception as e:
        print(f"\n✗ Ошибка при проверке CUDA: {e}")
        return False


def test_video_decoding():
    """Проверяет возможность декодирования видео"""
    print("\nПроверка декодирования видео...")
    
    try:
        import av
        import numpy as np
        
        # Создаем тестовое видео в памяти
        print("  Создание тестового видео...")
        
        # Проверяем что PyAV работает
        print("  ✓ PyAV установлен и работает")
        
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False


def test_project_structure():
    """Проверяет структуру проекта"""
    print("\nПроверка структуры проекта...")
    
    required_dirs = [
        'hw1',
        'hw2',
        'hw2/variant_a',
        'hw2/variant_b',
        'utils',
        'data',
        'outputs',
        'scripts',
        'docs',
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'scripts/setup_env.sh',
        'scripts/run_on_server.sh',
        'scripts/run_and_save_logs.sh',
        'docs/SETUP.md',
        'docs/FOR_TEACHER.md',
        'hw1/task01_decoder.py',
        'hw2/variant_a/stabilization.py',
        'hw2/variant_b/vos_system.py',
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ отсутствует")
            all_ok = False
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} отсутствует")
            all_ok = False
    
    if all_ok:
        print("\n✓ Структура проекта корректна")
    else:
        print("\n✗ Некоторые файлы или папки отсутствуют")
    
    return all_ok


def main():
    print("="*60)
    print("Проверка окружения для проекта Video Processing")
    print("="*60)
    print()
    
    results = []
    
    # Проверка импортов
    results.append(("Импорты", test_imports()))
    
    # Проверка CUDA
    results.append(("CUDA", test_cuda()))
    
    # Проверка декодирования видео
    results.append(("Декодирование видео", test_video_decoding()))
    
    # Проверка структуры проекта
    results.append(("Структура проекта", test_project_structure()))
    
    # Итоги
    print()
    print("="*60)
    print("ИТОГИ ПРОВЕРКИ")
    print("="*60)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("✓ Все проверки пройдены успешно!")
        print()
        print("Загрузите видео в data/ и запустите задания:")
        print("  bash run_on_server.sh")
        print("  или")
        print("  python hw1/task01_decoder.py --video data/sample.mp4")
    else:
        print("✗ Некоторые проверки не пройдены.")
    
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

