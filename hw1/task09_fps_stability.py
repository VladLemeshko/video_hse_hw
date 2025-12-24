"""
Задача 9: Измерение стабильности FPS

Измерение коэффициента вариации FPS при разных конфигурациях
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


def measure_fps_stability(dataloader, model, device, num_iterations=100):
    """
    Измеряет стабильность FPS
    
    Args:
        dataloader: DataLoader
        model: модель
        device: устройство
        num_iterations: количество итераций
        
    Returns:
        fps_values: список мгновенных FPS
        avg_fps: средний FPS
        cv: коэффициент вариации
    """
    model.eval()
    fps_values = []
    
    with torch.no_grad():
        prev_time = time.time()
        
        for i, (clips, _) in enumerate(dataloader):
            if i >= num_iterations:
                break
            
            # Перенос и инференс
            clips = clips.to(device, non_blocking=True)
            output = model(clips)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Вычисляем мгновенный FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps_values.append(instant_fps)
            
            prev_time = current_time
    
    fps_values = np.array(fps_values)
    
    # Метрики
    avg_fps = np.mean(fps_values)
    std_fps = np.std(fps_values)
    cv = std_fps / avg_fps if avg_fps > 0 else 0
    
    return fps_values, avg_fps, cv


def main():
    parser = argparse.ArgumentParser(description='Задача 9: Стабильность FPS')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--num_iterations', type=int, default=100, 
                        help='Число итераций для измерения')
    parser.add_argument('--output', type=str, default='outputs/hw1/task09_fps_stability.png',
                        help='Путь для сохранения графика')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Задача 9: Измерение стабильности FPS ===")
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
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем модель
    model = DummyModel().to(device)
    
    # Тестируем разные конфигурации
    configs = [
        {'batch_size': 2, 'num_workers': 2, 'prefetch': 2},
        {'batch_size': 4, 'num_workers': 4, 'prefetch': 2},
        {'batch_size': 8, 'num_workers': 4, 'prefetch': 4},
    ]
    
    results = []
    
    print(f"Измерение стабильности FPS ({args.num_iterations} итераций)...")
    print()
    
    for i, config in enumerate(configs):
        print(f"Конфигурация {i+1}:")
        print(f"  batch_size={config['batch_size']}, "
              f"num_workers={config['num_workers']}, "
              f"prefetch_factor={config['prefetch']}")
        
        # Дублируем файлы для достаточного количества данных
        video_files_extended = video_files[:]
        needed = config['batch_size'] * args.num_iterations
        while len(video_files_extended) < needed:
            video_files_extended.extend(video_files)
        
        # Создаем dataset и dataloader
        dataset = VideoDataset(
            video_files_extended[:needed],
            clip_len=args.clip_len,
            stride=2,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            prefetch_factor=config['prefetch'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Измеряем
        fps_values, avg_fps, cv = measure_fps_stability(
            dataloader, model, device, num_iterations=args.num_iterations
        )
        
        results.append({
            'config': config,
            'fps_values': fps_values,
            'avg_fps': avg_fps,
            'cv': cv
        })
        
        print(f"  Средний FPS: {avg_fps:.2f}")
        print(f"  Std FPS: {np.std(fps_values):.2f}")
        print(f"  CV: {cv:.4f} {'✓' if cv < 0.05 else ''}")
        print()
    
    # Анализ
    print("✓ Результаты анализа:")
    print()
    print("| Config                          | Avg FPS | CV     | Стабильный? |")
    print("|--------------------------------|---------|--------|-------------|")
    
    for i, result in enumerate(results):
        cfg = result['config']
        config_str = f"B={cfg['batch_size']}, W={cfg['num_workers']}, P={cfg['prefetch']}"
        stable = "✓" if result['cv'] < 0.05 else "✗"
        
        print(f"| {config_str:<30} | {result['avg_fps']:>7.2f} | "
              f"{result['cv']:>6.4f} | {stable:>11} |")
    
    print()
    print("Примечание: Стабильным считается CV < 0.05")
    print()
    
    # Визуализация
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_configs = len(results)
    fig, axes = plt.subplots(num_configs, 2, figsize=(14, 5 * num_configs))
    
    if num_configs == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        cfg = result['config']
        config_name = f"B={cfg['batch_size']}, W={cfg['num_workers']}, P={cfg['prefetch']}"
        
        # График FPS во времени
        ax1 = axes[i, 0]
        ax1.plot(result['fps_values'], linewidth=1, alpha=0.7)
        ax1.axhline(y=result['avg_fps'], color='r', linestyle='--', 
                   label=f'Среднее: {result["avg_fps"]:.2f}')
        ax1.axhline(y=result['avg_fps'] * 0.95, color='orange', linestyle=':', 
                   alpha=0.5, label='±5%')
        ax1.axhline(y=result['avg_fps'] * 1.05, color='orange', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Итерация')
        ax1.set_ylabel('FPS (батчи/с)')
        ax1.set_title(f'{config_name}\nFPS во времени')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Гистограмма FPS
        ax2 = axes[i, 1]
        ax2.hist(result['fps_values'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=result['avg_fps'], color='r', linestyle='--', 
                   label=f'Среднее: {result["avg_fps"]:.2f}')
        ax2.set_xlabel('FPS (батчи/с)')
        ax2.set_ylabel('Частота')
        ax2.set_title(f'{config_name}\nРаспределение FPS (CV={result["cv"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ График сохранен в {output_path}")
    print()
    
    # Рекомендации
    best_idx = min(range(len(results)), key=lambda i: results[i]['cv'])
    best_config = results[best_idx]['config']
    
    print("Рекомендации:")
    print(f"  Наиболее стабильная конфигурация:")
    print(f"    batch_size={best_config['batch_size']}")
    print(f"    num_workers={best_config['num_workers']}")
    print(f"    prefetch_factor={best_config['prefetch']}")
    print(f"    CV={results[best_idx]['cv']:.4f}")
    print()
    
    print("=== Задача 9 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())


