"""
Задача 5: Prefetch и pinned memory

Сравнение производительности с и без prefetch_factor и pin_memory
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
from hw1.task04_profiling import DummyModel
import torchvision.transforms.v2 as transforms


def measure_fps_and_jitter(dataloader, model, device, num_batches=50):
    """
    Измеряет средний FPS и jitter (вариацию времени между батчами)
    
    Args:
        dataloader: DataLoader
        model: модель для инференса
        device: устройство
        num_batches: количество батчей
        
    Returns:
        avg_fps: средний FPS
        jitter: стандартное отклонение времени между батчами
    """
    model.eval()
    batch_times = []
    
    with torch.no_grad():
        prev_time = time.time()
        
        for i, (clips, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Перенос и инференс
            clips = clips.to(device)
            output = model(clips)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Время обработки батча
            current_time = time.time()
            batch_time = current_time - prev_time
            batch_times.append(batch_time)
            prev_time = current_time
    
    # Вычисляем метрики
    batch_times = np.array(batch_times[1:])  # Пропускаем первый (прогрев)
    
    # FPS = кадров в батче / время обработки батча
    # Предполагаем batch_size и clip_len из первого батча
    avg_batch_time = np.mean(batch_times)
    avg_fps = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
    
    # Jitter = стандартное отклонение времени между батчами
    jitter = np.std(batch_times)
    
    return avg_fps, jitter


def main():
    parser = argparse.ArgumentParser(description='Задача 5: Prefetch и pinned memory')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--num_batches', type=int, default=50, 
                        help='Число батчей для измерения')
    parser.add_argument('--output', type=str, default='outputs/hw1/task05_prefetch.png',
                        help='Путь для сохранения графика')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Задача 5: Prefetch и pinned memory ===")
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
    
    # Дублируем файлы
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
        stride=2,
        transform=transform
    )
    
    # Создаем модель
    model = DummyModel().to(device)
    
    # Тестируем разные конфигурации
    configs = [
        {'name': 'Базовый', 'prefetch_factor': None, 'pin_memory': False},
        {'name': 'Pin memory', 'prefetch_factor': None, 'pin_memory': True},
        {'name': 'Prefetch=2', 'prefetch_factor': 2, 'pin_memory': False},
        {'name': 'Оба', 'prefetch_factor': 2, 'pin_memory': True},
    ]
    
    results = []
    
    print("Тестирование конфигураций:")
    print()
    
    for config in configs:
        print(f"Конфигурация: {config['name']}")
        print(f"  prefetch_factor={config['prefetch_factor']}, "
              f"pin_memory={config['pin_memory']}")
        
        # Создаем DataLoader с заданными параметрами
        dataloader_kwargs = {
            'dataset': dataset,
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': config['pin_memory'] and device.type == 'cuda',
        }
        
        if config['prefetch_factor'] is not None:
            dataloader_kwargs['prefetch_factor'] = config['prefetch_factor']
        
        dataloader = DataLoader(**dataloader_kwargs)
        
        # Измеряем производительность
        fps, jitter = measure_fps_and_jitter(
            dataloader, model, device, num_batches=args.num_batches
        )
        
        results.append({
            'name': config['name'],
            'fps': fps,
            'jitter': jitter
        })
        
        print(f"  FPS: {fps:.2f}")
        print(f"  Jitter: {jitter:.4f} с")
        print()
    
    # Анализ результатов
    print("✓ Сравнение результатов:")
    baseline_fps = results[0]['fps']
    baseline_jitter = results[0]['jitter']
    
    for result in results:
        fps_improvement = (result['fps'] - baseline_fps) / baseline_fps * 100
        jitter_improvement = (baseline_jitter - result['jitter']) / baseline_jitter * 100
        
        print(f"{result['name']}:")
        print(f"  FPS: {result['fps']:.2f} ({fps_improvement:+.1f}%)")
        print(f"  Jitter: {result['jitter']:.4f} с ({jitter_improvement:+.1f}%)")
    
    print()
    
    # Объяснение pin_memory
    print("Объяснение pin_memory:")
    print("  Pinned memory (page-locked memory) позволяет ускорить")
    print("  передачу данных между CPU и GPU, так как:")
    print("  - Данные не могут быть перемещены ОС (не swapped)")
    print("  - GPU может напрямую обращаться к памяти через DMA")
    print("  - Избегается дополнительное копирование в pageable память")
    print()
    
    # Визуализация
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    names = [r['name'] for r in results]
    fps_values = [r['fps'] for r in results]
    jitter_values = [r['jitter'] * 1000 for r in results]  # в мс
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График FPS
    bars1 = ax1.bar(names, fps_values, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('FPS (батчи/с)', fontsize=12)
    ax1.set_title('Сравнение FPS', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars1, fps_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # График Jitter
    bars2 = ax2.bar(names, jitter_values, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Jitter (мс)', fontsize=12)
    ax2.set_title('Сравнение Jitter (меньше = лучше)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars2, jitter_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ График сохранен в {output_path}")
    print()
    print("=== Задача 5 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())


