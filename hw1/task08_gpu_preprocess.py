"""
Задача 8: Оптимизация препроцессинга

Сравнение препроцессинга на CPU и GPU
"""

import argparse
import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from hw1.task02_dataset import VideoDataset
from hw1.task04_profiling import DummyModel


class VideoDatasetGPU(VideoDataset):
    """
    Dataset с препроцессингом на GPU
    """
    
    def __init__(self, video_files, clip_len=16, stride=2, transform=None, device='cuda'):
        super().__init__(video_files, clip_len, stride, transform=None)
        self.gpu_transform = transform
        self.device = device
    
    def __getitem__(self, idx):
        # Получаем сырые кадры без трансформаций
        video_file = self.video_files[idx]
        
        from utils import read_clip
        frames, _, _ = read_clip(
            video_file,
            start=0,
            num_frames=self.clip_len,
            stride=self.stride
        )
        
        # Конвертируем в tensor (CPU)
        # (T, H, W, C) -> (C, T, H, W)
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        return clip, Path(video_file).name


def measure_pipeline_latency(dataloader, model, device, num_batches=20, 
                            gpu_preprocess=False, gpu_transform=None):
    """
    Измеряет латентность этапов пайплайна
    
    Args:
        dataloader: DataLoader
        model: модель
        device: устройство
        num_batches: количество батчей
        gpu_preprocess: выполнять ли препроцессинг на GPU
        gpu_transform: трансформации для GPU
        
    Returns:
        dict с временами этапов
    """
    model.eval()
    
    times = {
        'decode': [],
        'transfer': [],
        'preprocess': [],
        'inference': []
    }
    
    with torch.no_grad():
        for i, (clips, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Декодирование уже произошло в DataLoader
            
            # Перенос на GPU
            transfer_start = time.time()
            clips = clips.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            transfer_time = time.time() - transfer_start
            
            # Препроцессинг
            preprocess_start = time.time()
            
            if gpu_preprocess and gpu_transform:
                # Применяем трансформации на GPU
                B, C, T, H, W = clips.shape
                # Reshape для применения 2D трансформаций
                clips_reshaped = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                clips_transformed = gpu_transform(clips_reshaped)
                clips = clips_transformed.reshape(B, T, C, 
                                                 clips_transformed.shape[-2], 
                                                 clips_transformed.shape[-1])
                clips = clips.permute(0, 2, 1, 3, 4)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            preprocess_time = time.time() - preprocess_start
            
            # Инференс
            infer_start = time.time()
            output = model(clips)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            infer_time = time.time() - infer_start
            
            times['transfer'].append(transfer_time)
            times['preprocess'].append(preprocess_time)
            times['inference'].append(infer_time)
    
    return times


def main():
    parser = argparse.ArgumentParser(description='Задача 8: GPU препроцессинг')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--num_batches', type=int, default=20, 
                        help='Число батчей для измерения')
    parser.add_argument('--output', type=str, default='outputs/hw1/task08_gpu_preprocess.png',
                        help='Путь для сохранения графика')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Задача 8: GPU препроцессинг ===")
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
    
    # Создаем модель
    model = DummyModel().to(device)
    
    # Вариант 1: Препроцессинг на CPU
    print("1. Препроцессинг на CPU...")
    
    cpu_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_cpu = VideoDataset(
        video_files,
        clip_len=args.clip_len,
        stride=2,
        transform=cpu_transform
    )
    
    dataloader_cpu = DataLoader(
        dataset_cpu,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    times_cpu = measure_pipeline_latency(
        dataloader_cpu, model, device, num_batches=args.num_batches,
        gpu_preprocess=False
    )
    
    avg_times_cpu = {k: np.mean(v) * 1000 for k, v in times_cpu.items()}
    total_cpu = sum(avg_times_cpu.values())
    
    print(f"   Transfer: {avg_times_cpu['transfer']:.2f} мс")
    print(f"   Preprocess: {avg_times_cpu['preprocess']:.2f} мс")
    print(f"   Inference: {avg_times_cpu['inference']:.2f} мс")
    print(f"   Total: {total_cpu:.2f} мс")
    print()
    
    # Вариант 2: Препроцессинг на GPU
    if device.type == 'cuda':
        print("2. Препроцессинг на GPU...")
        
        gpu_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset_gpu = VideoDatasetGPU(
            video_files,
            clip_len=args.clip_len,
            stride=2,
            transform=gpu_transform,
            device=device
        )
        
        dataloader_gpu = DataLoader(
            dataset_gpu,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        times_gpu = measure_pipeline_latency(
            dataloader_gpu, model, device, num_batches=args.num_batches,
            gpu_preprocess=True, gpu_transform=gpu_transform
        )
        
        avg_times_gpu = {k: np.mean(v) * 1000 for k, v in times_gpu.items()}
        total_gpu = sum(avg_times_gpu.values())
        
        print(f"   Transfer: {avg_times_gpu['transfer']:.2f} мс")
        print(f"   Preprocess: {avg_times_gpu['preprocess']:.2f} мс")
        print(f"   Inference: {avg_times_gpu['inference']:.2f} мс")
        print(f"   Total: {total_gpu:.2f} мс")
        print()
        
        # Сравнение
        speedup = total_cpu / total_gpu
        print(f"✓ Ускорение с GPU препроцессингом: {speedup:.2f}x")
        print()
    else:
        avg_times_gpu = avg_times_cpu
        total_gpu = total_cpu
    
    # Визуализация
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stages = ['Transfer', 'Preprocess', 'Inference']
    
    cpu_values = [avg_times_cpu['transfer'], 
                  avg_times_cpu['preprocess'], 
                  avg_times_cpu['inference']]
    
    gpu_values = [avg_times_gpu['transfer'], 
                  avg_times_gpu['preprocess'], 
                  avg_times_gpu['inference']]
    
    x = np.arange(len(stages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, cpu_values, width, label='CPU препроцессинг',
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, gpu_values, width, label='GPU препроцессинг',
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Время (мс)', fontsize=12)
    ax.set_title('Сравнение препроцессинга на CPU и GPU', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ График сохранен в {output_path}")
    print()
    print("=== Задача 8 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())

