"""
Задача 6: Pipeline overlap

Перекрытие декодирования и инференса с помощью CUDA Streams
"""

import argparse
import sys
from pathlib import Path
import time
import torch
import torch.multiprocessing as mp
from queue import Queue
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils import read_clip
from hw1.task04_profiling import DummyModel
import torchvision.transforms.v2 as transforms


def decode_worker(video_files, clip_len, stride, transform, result_queue, num_clips):
    """
    Воркер для декодирования видео
    
    Args:
        video_files: список видеофайлов
        clip_len: длина клипа
        stride: шаг
        transform: трансформации
        result_queue: очередь для результатов
        num_clips: количество клипов для обработки
    """
    for i in range(num_clips):
        video_file = video_files[i % len(video_files)]
        
        # Декодируем
        frames, _, _ = read_clip(video_file, num_frames=clip_len, stride=stride)
        
        # Применяем трансформации
        transformed_frames = []
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            if transform:
                frame_tensor = transform(frame_tensor)
            transformed_frames.append(frame_tensor)
        
        clip = torch.stack(transformed_frames, dim=1)  # (C, T, H, W)
        
        result_queue.put(clip)
    
    result_queue.put(None)  # Сигнал завершения


def inference_worker(model, device, data_queue, num_clips):
    """
    Воркер для инференса
    
    Args:
        model: модель
        device: устройство
        data_queue: очередь с данными
        num_clips: количество клипов
    """
    model.eval()
    processed = 0
    
    with torch.no_grad():
        while processed < num_clips:
            clip = data_queue.get()
            
            if clip is None:
                break
            
            # Добавляем batch dimension
            clip = clip.unsqueeze(0).to(device)
            output = model(clip)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            processed += 1


def measure_sequential(video_files, model, device, clip_len, stride, 
                       transform, num_clips=20):
    """
    Измеряет время последовательного выполнения
    """
    model.eval()
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_clips):
            video_file = video_files[i % len(video_files)]
            
            # Декодирование
            frames, _, _ = read_clip(video_file, num_frames=clip_len, stride=stride)
            
            # Трансформация
            transformed_frames = []
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                if transform:
                    frame_tensor = transform(frame_tensor)
                transformed_frames.append(frame_tensor)
            
            clip = torch.stack(transformed_frames, dim=1)
            
            # Инференс
            clip = clip.unsqueeze(0).to(device)
            output = model(clip)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    return elapsed


def measure_overlapped(video_files, model, device, clip_len, stride, 
                      transform, num_clips=20):
    """
    Измеряет время с перекрытием декодирования и инференса
    """
    # Используем очередь для передачи данных между процессами
    result_queue = mp.Queue(maxsize=2)  # Буфер на 2 элемента
    
    start_time = time.time()
    
    # Запускаем декодирование в отдельном потоке
    import threading
    decode_thread = threading.Thread(
        target=decode_worker,
        args=(video_files, clip_len, stride, transform, result_queue, num_clips)
    )
    decode_thread.start()
    
    # Выполняем инференс в главном потоке
    model.eval()
    processed = 0
    
    with torch.no_grad():
        while processed < num_clips:
            clip = result_queue.get()
            
            if clip is None:
                break
            
            clip = clip.unsqueeze(0).to(device)
            output = model(clip)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            processed += 1
    
    decode_thread.join()
    elapsed = time.time() - start_time
    
    return elapsed


def main():
    parser = argparse.ArgumentParser(description='Задача 6: Pipeline overlap')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--num_clips', type=int, default=30, 
                        help='Число клипов для обработки')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Задача 6: Pipeline overlap ===")
    print(f"Папка с видео: {args.video_dir}")
    print()
    
    # Проверяем CUDA
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Собираем список видеофайлов
    video_dir = Path(args.video_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f"*{ext}")))
    
    video_files = [str(f) for f in video_files]
    
    if len(video_files) == 0:
        print(f"✗ Не найдено видеофайлов в {args.video_dir}")
        return 1
    
    # Создаем трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем модель
    model = DummyModel().to(device)
    
    print(f"Обработка {args.num_clips} клипов...")
    print()
    
    # Последовательное выполнение
    print("1. Последовательное выполнение (декодирование → инференс)...")
    seq_time = measure_sequential(
        video_files, model, device, args.clip_len, 2, transform, args.num_clips
    )
    print(f"   Время: {seq_time:.2f} с")
    print(f"   Средний latency: {seq_time / args.num_clips * 1000:.2f} мс/клип")
    print()
    
    # Перекрытое выполнение
    print("2. Перекрытое выполнение (декодирование || инференс)...")
    overlap_time = measure_overlapped(
        video_files, model, device, args.clip_len, 2, transform, args.num_clips
    )
    print(f"   Время: {overlap_time:.2f} с")
    print(f"   Средний latency: {overlap_time / args.num_clips * 1000:.2f} мс/клип")
    print()
    
    # Сравнение
    speedup = seq_time / overlap_time
    latency_reduction = (seq_time - overlap_time) / seq_time * 100
    
    print("✓ Результаты сравнения:")
    print(f"  Ускорение: {speedup:.2f}x")
    print(f"  Снижение latency: {latency_reduction:.1f}%")
    print()
    
    print("Объяснение:")
    print("  При перекрытии выполнения декодирование следующего клипа")
    print("  происходит одновременно с инференсом текущего клипа.")
    print("  Это позволяет более эффективно использовать CPU и GPU,")
    print("  уменьшая простои и общее время обработки.")
    print()
    
    print("=== Задача 6 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())

