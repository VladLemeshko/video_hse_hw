"""
Задача 4: Профилирование этапов пайплайна

Измерение времени выполнения стадий: декодирование, препроцессинг, инференс
"""

import argparse
import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
import torch.profiler
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from hw1.task02_dataset import VideoDataset
import torchvision.transforms.v2 as transforms


class DummyModel(torch.nn.Module):
    """Заглушка для инференса"""
    
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x.squeeze()


def profile_pipeline(dataloader, model, device, num_batches=10):
    """
    Профилирует пайплайн обработки видео
    
    Args:
        dataloader: DataLoader для данных
        model: модель для инференса
        device: устройство (cuda/cpu)
        num_batches: количество батчей для профилирования
        
    Returns:
        dict с временами для каждого этапа
    """
    model.eval()
    
    times = {
        'data_loading': [],
        'transfer': [],
        'inference': []
    }
    
    with torch.no_grad():
        for i, (clips, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Время загрузки данных уже прошло (встроено в DataLoader)
            data_time = time.time()
            
            # Перенос на GPU
            transfer_start = time.time()
            clips = clips.to(device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            transfer_time = time.time() - transfer_start
            
            # Инференс
            infer_start = time.time()
            output = model(clips)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            infer_time = time.time() - infer_start
            
            times['transfer'].append(transfer_time)
            times['inference'].append(infer_time)
    
    return times


def main():
    parser = argparse.ArgumentParser(description='Задача 4: Профилирование пайплайна')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--num_batches', type=int, default=20, 
                        help='Число батчей для профилирования')
    parser.add_argument('--output', type=str, default='outputs/hw1/task04_profiling.png',
                        help='Путь для сохранения графика')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Задача 4: Профилирование пайплайна ===")
    print(f"Папка с видео: {args.video_dir}")
    print()
    
    # Проверяем CUDA
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
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
    
    # Создаем dataset и dataloader
    dataset = VideoDataset(
        video_files,
        clip_len=args.clip_len,
        stride=2,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Создаем модель
    model = DummyModel().to(device)
    
    print("Профилирование пайплайна...")
    
    # Измеряем время декодирования отдельно
    decode_times = []
    for i, video_file in enumerate(video_files[:args.num_batches]):
        start = time.time()
        from utils import read_clip
        frames, _, _ = read_clip(video_file, num_frames=args.clip_len, stride=2)
        decode_times.append(time.time() - start)
    
    avg_decode_time = np.mean(decode_times) * 1000  # в мс
    
    # Профилируем препроцессинг и инференс
    times = profile_pipeline(dataloader, model, device, num_batches=args.num_batches)
    
    avg_transfer_time = np.mean(times['transfer']) * 1000  # в мс
    avg_infer_time = np.mean(times['inference']) * 1000  # в мс
    
    print()
    print("✓ Результаты профилирования (мс):")
    print(f"  Декодирование (L_dec):  {avg_decode_time:.2f} мс")
    print(f"  Препроцессинг (L_prep): {avg_transfer_time:.2f} мс")
    print(f"  Инференс (L_inf):       {avg_infer_time:.2f} мс")
    print()
    
    # Соотношение времен
    total_time = avg_decode_time + avg_transfer_time + avg_infer_time
    print(f"  Соотношение L_dec:L_prep:L_inf = "
          f"{avg_decode_time/total_time:.2f}:"
          f"{avg_transfer_time/total_time:.2f}:"
          f"{avg_infer_time/total_time:.2f}")
    print()
    
    # Визуализация
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stages = ['Декодирование', 'Препроцессинг', 'Инференс']
    times_ms = [avg_decode_time, avg_transfer_time, avg_infer_time]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Столбчатая диаграмма
    bars = ax1.bar(stages, times_ms, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Время (мс)', fontsize=12)
    ax1.set_title('Время выполнения этапов пайплайна', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, time_val in zip(bars, times_ms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f} мс',
                ha='center', va='bottom', fontsize=10)
    
    # Круговая диаграмма
    ax2.pie(times_ms, labels=stages, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Распределение времени по этапам', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ График сохранен в {output_path}")
    print()
    print("=== Задача 4 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())

