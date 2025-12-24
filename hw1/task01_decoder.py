"""
Задача 1: Базовый декодер видеокадров

Реализация функции чтения кадров из видео с помощью PyAV
"""

import argparse
import sys
from pathlib import Path

# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent.parent))

from utils import read_clip, visualize_frames


def main():
    parser = argparse.ArgumentParser(description='Задача 1: Базовый декодер видеокадров')
    parser.add_argument('--video', type=str, required=True, help='Путь к видеофайлу')
    parser.add_argument('--start', type=int, default=0, help='Начальный кадр')
    parser.add_argument('--num_frames', type=int, default=16, help='Количество кадров')
    parser.add_argument('--stride', type=int, default=2, help='Шаг между кадрами')
    parser.add_argument('--output', type=str, default='outputs/hw1/task01_frames.png', 
                        help='Путь для сохранения визуализации')
    
    args = parser.parse_args()
    
    print(f"=== Задача 1: Базовый декодер видеокадров ===")
    print(f"Видеофайл: {args.video}")
    print(f"Параметры: start={args.start}, num_frames={args.num_frames}, stride={args.stride}")
    print()
    
    # Читаем клип
    try:
        frames, frame_indices, fps = read_clip(
            args.video, 
            start=args.start, 
            num_frames=args.num_frames, 
            stride=args.stride
        )
        
        print(f"✓ Успешно прочитано {len(frames)} кадров")
        print(f"  Форма: {frames.shape}")
        print(f"  FPS видео: {fps:.2f}")
        print(f"  Индексы кадров: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
        print()
        
        # Визуализируем кадры
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualize_frames(
            frames,
            title=f"Декодированные кадры (stride={args.stride}, n={len(frames)})",
            save_path=str(output_path)
        )
        
        print(f"✓ Визуализация сохранена в {output_path}")
        print()
        
        # Дополнительная информация
        print("Первый кадр:")
        print(f"  Индекс: {frame_indices[0]}")
        print(f"  Размер: {frames[0].shape}")
        print(f"  Min/Max значения: {frames[0].min()}/{frames[0].max()}")
        print()
        
        print("Последний кадр:")
        print(f"  Индекс: {frame_indices[-1]}")
        print(f"  Размер: {frames[-1].shape}")
        print()
        
        print("=== Задача 1 выполнена успешно! ===")
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


