"""
Задача 3: Параллельная загрузка данных

Исследование влияния num_workers на производительность DataLoader
"""

import argparse
import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from hw1.task02_dataset import VideoDataset
import torchvision.transforms.v2 as transforms


def measure_throughput(dataloader, num_batches=20):
    """
    Измеряет throughput (кадров/с) для DataLoader
    
    Args:
        dataloader: DataLoader для тестирования
        num_batches: количество батчей для измерения
        
    Returns:
        throughput: кадров в секунду
    """
    start_time = time.time()
    total_frames = 0
    
    for i, (clips, _) in enumerate(dataloader):
        # clips имеет форму (B, C, T, H, W)
        batch_size = clips.shape[0]
        clip_len = clips.shape[2]
        total_frames += batch_size * clip_len
        
        if i >= num_batches - 1:
            break
    
    elapsed_time = time.time() - start_time
    throughput = total_frames / elapsed_time
    
    return throughput


def main():
    parser = argparse.ArgumentParser(description='Задача 3: Параллельная загрузка данных')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--stride', type=int, default=2, help='Шаг между кадрами')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--num_batches', type=int, default=20, 
                        help='Число батчей для измерения')
    parser.add_argument('--output', type=str, default='outputs/hw1/task03_throughput.png',
                        help='Путь для сохранения графика')
    
    args = parser.parse_args()
    
    print(f"=== Задача 3: Параллельная загрузка данных ===")
    print(f"Папка с видео: {args.video_dir}")
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
    
    print(f"Найдено {len(video_files)} видеофайлов")
    
    # Дублируем файлы для достаточного количества данных
    while len(video_files) < args.batch_size * args.num_batches:
        video_files.extend(video_files[:])
    
    # Создаем трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем dataset
    dataset = VideoDataset(
        video_files,
        clip_len=args.clip_len,
        stride=args.stride,
        transform=transform
    )
    
    # Тестируем разное количество воркеров
    num_workers_list = [0, 1, 2, 4, 8]
    throughputs = []
    
    print(f"Измерение throughput для разных num_workers...")
    print(f"(по {args.num_batches} батчей, batch_size={args.batch_size})")
    print()
    
    for num_workers in num_workers_list:
        print(f"Тестирование num_workers={num_workers}...", end=' ')
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
        
        try:
            throughput = measure_throughput(dataloader, num_batches=args.num_batches)
            throughputs.append(throughput)
            print(f"{throughput:.2f} кадров/с")
        except Exception as e:
            print(f"Ошибка: {e}")
            throughputs.append(0)
    
    print()
    
    # Находим оптимальное значение
    best_idx = np.argmax(throughputs)
    best_workers = num_workers_list[best_idx]
    best_throughput = throughputs[best_idx]
    
    print(f"✓ Результаты:")
    print(f"  Лучший num_workers: {best_workers}")
    print(f"  Максимальный throughput: {best_throughput:.2f} кадров/с")
    
    # Проверяем насыщение
    saturation_threshold = 0.95
    for i in range(best_idx + 1, len(throughputs)):
        if throughputs[i] < throughputs[best_idx] * saturation_threshold:
            print(f"  Насыщение достигается при num_workers={num_workers_list[i-1]}")
            break
    
    print()
    
    # Строим график
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_workers_list, throughputs, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=best_throughput * saturation_threshold, color='r', 
                linestyle='--', alpha=0.5, label='95% от максимума')
    plt.xlabel('Число воркеров (num_workers)', fontsize=12)
    plt.ylabel('Throughput (кадров/с)', fontsize=12)
    plt.title('Зависимость throughput от числа воркеров', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ График сохранен в {output_path}")
    print()
    print("=== Задача 3 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())


