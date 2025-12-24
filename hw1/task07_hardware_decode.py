"""
Задача 7: Аппаратное декодирование

Сравнение CPU (PyAV) и GPU (decord) декодирования
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from utils import read_clip


def decode_with_pyav(video_file, num_frames=16, stride=2, num_runs=10):
    """
    Декодирование с помощью PyAV (CPU)
    """
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        frames, _, _ = read_clip(video_file, num_frames=num_frames, stride=stride)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return np.mean(times) * 1000  # в мс


def decode_with_decord(video_file, num_frames=16, stride=2, num_runs=10, use_gpu=True):
    """
    Декодирование с помощью decord (GPU/CPU)
    """
    try:
        import decord
        from decord import VideoReader, cpu, gpu
    except ImportError:
        print("✗ decord не установлен. Установите: pip install decord")
        return None
    
    times = []
    ctx = gpu(0) if use_gpu and torch.cuda.is_available() else cpu(0)
    
    try:
        for _ in range(num_runs):
            start = time.time()
            
            vr = VideoReader(video_file, ctx=ctx)
            total_frames = len(vr)
            
            # Выбираем индексы кадров
            indices = list(range(0, min(num_frames * stride, total_frames), stride))[:num_frames]
            
            frames = vr.get_batch(indices).asnumpy()
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        return np.mean(times) * 1000  # в мс
    
    except Exception as e:
        print(f"✗ Ошибка при декодировании с decord: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Задача 7: Аппаратное декодирование')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--num_runs', type=int, default=20, 
                        help='Число прогонов для усреднения')
    
    args = parser.parse_args()
    
    print(f"=== Задача 7: Аппаратное декодирование ===")
    print(f"Папка с видео: {args.video_dir}")
    print()
    
    # Проверяем CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступна: {cuda_available}")
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Поддержка NVDEC: проверяется...")
    print()
    
    # Собираем список видеофайлов
    video_dir = Path(args.video_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f"*{ext}")))
    
    if len(video_files) == 0:
        print(f"✗ Не найдено видеофайлов в {args.video_dir}")
        return 1
    
    # Берем первый видеофайл для тестирования
    test_video = str(video_files[0])
    print(f"Тестовое видео: {Path(test_video).name}")
    print(f"Декодируем {args.clip_len} кадров, stride=2")
    print(f"Усредняем по {args.num_runs} прогонам")
    print()
    
    # Результаты
    results = []
    
    # PyAV (CPU)
    print("1. PyAV (CPU)...", end=' ')
    pyav_time = decode_with_pyav(test_video, num_frames=args.clip_len, 
                                  stride=2, num_runs=args.num_runs)
    print(f"{pyav_time:.2f} мс")
    results.append({'method': 'PyAV (CPU)', 'time': pyav_time})
    
    # decord (CPU)
    print("2. decord (CPU)...", end=' ')
    decord_cpu_time = decode_with_decord(test_video, num_frames=args.clip_len,
                                         stride=2, num_runs=args.num_runs, use_gpu=False)
    if decord_cpu_time:
        print(f"{decord_cpu_time:.2f} мс")
        results.append({'method': 'decord (CPU)', 'time': decord_cpu_time})
    else:
        print("недоступно")
    
    # decord (GPU)
    if cuda_available:
        print("3. decord (GPU/NVDEC)...", end=' ')
        decord_gpu_time = decode_with_decord(test_video, num_frames=args.clip_len,
                                             stride=2, num_runs=args.num_runs, use_gpu=True)
        if decord_gpu_time:
            print(f"{decord_gpu_time:.2f} мс")
            results.append({'method': 'decord (GPU)', 'time': decord_gpu_time})
        else:
            print("недоступно")
    
    print()
    
    # Таблица результатов
    print("✓ Таблица результатов:")
    print()
    print("| Метод               | Среднее время (мс) | FPS   | Ускорение |")
    print("|---------------------|-------------------|-------|-----------|")
    
    baseline_time = results[0]['time']
    
    for result in results:
        fps = 1000.0 / result['time'] * args.clip_len if result['time'] > 0 else 0
        speedup = baseline_time / result['time'] if result['time'] > 0 else 0
        
        print(f"| {result['method']:<19} | {result['time']:>17.2f} | "
              f"{fps:>5.1f} | {speedup:>9.2f}x |")
    
    print()
    
    # Выводы
    print("Выводы:")
    print("  - PyAV использует CPU для декодирования (libav/ffmpeg)")
    print("  - decord может использовать GPU через NVDEC (NVIDIA)")
    print("  - Аппаратное декодирование (GPU) обычно быстрее для больших видео")
    print("  - Для небольших клипов overhead может перевесить выигрыш")
    print()
    
    if cuda_available and len(results) > 2:
        gpu_speedup = baseline_time / results[-1]['time']
        if gpu_speedup > 1.2:
            print(f"  ✓ GPU декодирование дает ускорение {gpu_speedup:.1f}x!")
        else:
            print(f"  ⚠ GPU декодирование дает лишь {gpu_speedup:.1f}x (малый клип?)")
    
    print()
    print("=== Задача 7 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())


