"""
Задача 2: Реализация Dataset для видеоклипов

Создание класса VideoDataset для работы с PyTorch DataLoader
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils import read_clip


class VideoDataset(Dataset):
    """
    Dataset для загрузки видеоклипов
    """
    
    def __init__(self, video_files, clip_len=16, stride=2, transform=None):
        """
        Args:
            video_files: список путей к видеофайлам
            clip_len: длина клипа в кадрах
            stride: шаг между кадрами
            transform: трансформации для препроцессинга
        """
        self.video_files = video_files
        self.clip_len = clip_len
        self.stride = stride
        self.transform = transform
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        """
        Возвращает один клип
        
        Returns:
            clip: тензор формы (C, T, H, W) после трансформаций
            filename: имя файла
        """
        video_file = self.video_files[idx]
        
        # Читаем клип
        frames, _, _ = read_clip(
            video_file,
            start=0,
            num_frames=self.clip_len,
            stride=self.stride
        )
        
        # Применяем трансформации
        if self.transform:
            # Трансформируем каждый кадр
            transformed_frames = []
            for frame in frames:
                # Конвертируем в tensor
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                # Конвертируем в float если еще uint8
                if frame_tensor.dtype == torch.uint8:
                    frame_tensor = frame_tensor.float() / 255.0
                frame_tensor = self.transform(frame_tensor)
                transformed_frames.append(frame_tensor)
            
            # Собираем в клип (C, T, H, W)
            clip = torch.stack(transformed_frames, dim=1)
        else:
            # Без трансформаций: (T, H, W, C) -> (C, T, H, W)
            clip = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        return clip, Path(video_file).name


def main():
    parser = argparse.ArgumentParser(description='Задача 2: Dataset для видеоклипов')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--stride', type=int, default=2, help='Шаг между кадрами')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--num_workers', type=int, default=2, help='Число воркеров')
    
    args = parser.parse_args()
    
    print(f"=== Задача 2: Dataset для видеоклипов ===")
    print(f"Папка с видео: {args.video_dir}")
    print(f"Параметры: clip_len={args.clip_len}, stride={args.stride}")
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
    
    print(f"✓ Dataset создан, размер: {len(dataset)}")
    
    # Создаем DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✓ DataLoader создан:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print()
    
    # Тестируем загрузку
    print("Тестирование загрузки данных...")
    try:
        for i, (clips, filenames) in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Форма клипов: {clips.shape}")  # Должно быть (B, C, T, H, W)
            print(f"  Dtype: {clips.dtype}")
            print(f"  Min/Max: {clips.min():.3f}/{clips.max():.3f}")
            print(f"  Файлы: {filenames[:2]}...")
            print()
            
            if i >= 2:  # Показываем только первые 3 батча
                break
        
        print("=== Задача 2 выполнена успешно! ===")
        
    except Exception as e:
        print(f"✗ Ошибка при загрузке: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

