"""
Вариант A: Система временной стабилизации масок

Реализация anti-flicker системы для кадр-по-кадр сегментации
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import read_clip, get_video_info, visualize_masks_comparison, plot_metrics
from hw2.variant_a.metrics import (
    compute_mask_stability, 
    compute_boundary_jitter,
    visualize_stability_map,
    compute_temporal_iou
)


def load_segmentation_model(model_name='deeplabv3'):
    """
    Загружает предобученную модель сегментации
    
    Args:
        model_name: название модели (deeplabv3, fcn, etc.)
        
    Returns:
        модель и трансформации
    """
    import torchvision.models.segmentation as models
    import torchvision.transforms as transforms
    
    if model_name == 'deeplabv3':
        model = models.deeplabv3_resnet101(pretrained=True)
    elif model_name == 'fcn':
        model = models.fcn_resnet101(pretrained=True)
    else:
        raise ValueError(f"Неподдерживаемая модель: {model_name}")
    
    model.eval()
    
    # Трансформации для входа
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform


def segment_frames(frames, model, transform, device, target_class=15):
    """
    Сегментирует кадры с помощью модели
    
    Args:
        frames: массив кадров (T, H, W, 3)
        model: модель сегментации
        transform: трансформации
        device: устройство
        target_class: целевой класс для сегментации (15 = person в COCO)
        
    Returns:
        список масок
    """
    masks = []
    
    for frame in tqdm(frames, desc="Сегментация кадров"):
        # Препроцессинг
        input_tensor = transform(frame).unsqueeze(0).to(device)
        
        # Инференс
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        # Получаем маску для целевого класса
        output_predictions = output.argmax(0).cpu().numpy()
        mask = (output_predictions == target_class).astype(np.float32)
        
        # Resize до оригинального размера если нужно
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        masks.append(mask)
    
    return masks


def smooth_masks_temporal_window(masks, window_size=5):
    """
    Сглаживание масок усреднением по временному окну
    
    Args:
        masks: список масок
        window_size: размер окна
        
    Returns:
        сглаженные маски
    """
    smoothed_masks = []
    half_window = window_size // 2
    
    for i in range(len(masks)):
        # Определяем границы окна
        start = max(0, i - half_window)
        end = min(len(masks), i + half_window + 1)
        
        # Усредняем маски в окне
        window_masks = masks[start:end]
        smoothed_mask = np.mean(window_masks, axis=0)
        
        smoothed_masks.append(smoothed_mask)
    
    return smoothed_masks


def smooth_masks_median(masks, window_size=5):
    """
    Медианное сглаживание масок
    
    Args:
        masks: список масок
        window_size: размер окна
        
    Returns:
        сглаженные маски
    """
    smoothed_masks = []
    half_window = window_size // 2
    
    for i in range(len(masks)):
        start = max(0, i - half_window)
        end = min(len(masks), i + half_window + 1)
        
        window_masks = np.array(masks[start:end])
        smoothed_mask = np.median(window_masks, axis=0)
        
        smoothed_masks.append(smoothed_mask)
    
    return smoothed_masks


def smooth_masks_motion_aware(masks, frames, window_size=5):
    """
    Сглаживание с учетом движения
    
    Веса окна зависят от величины локального движения
    
    Args:
        masks: список масок
        frames: список кадров
        window_size: размер окна
        
    Returns:
        сглаженные маски
    """
    smoothed_masks = []
    half_window = window_size // 2
    
    # Вычисляем оптический поток между соседними кадрами
    flows = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Магнитуда потока
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flows.append(flow_mag)
    
    for i in range(len(masks)):
        start = max(0, i - half_window)
        end = min(len(masks), i + half_window + 1)
        
        # Вычисляем веса на основе движения
        weights = []
        for j in range(start, end):
            if j == i:
                weight = 1.0
            else:
                # Чем больше движение, тем меньше вес
                flow_idx = min(j, len(flows) - 1)
                motion_magnitude = np.mean(flows[flow_idx])
                weight = np.exp(-motion_magnitude / 10.0)  # Экспоненциальное затухание
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Взвешенное усреднение
        weighted_mask = np.zeros_like(masks[0])
        for j, w in zip(range(start, end), weights):
            weighted_mask += w * masks[j]
        
        smoothed_masks.append(weighted_mask)
    
    return smoothed_masks


def main():
    parser = argparse.ArgumentParser(
        description='Вариант A: Система временной стабилизации масок'
    )
    parser.add_argument('--video', type=str, required=True, 
                        help='Путь к видеофайлу')
    parser.add_argument('--output', type=str, default='outputs/hw2/variant_a',
                        help='Папка для выходных файлов')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Количество кадров для обработки')
    parser.add_argument('--model', type=str, default='deeplabv3',
                        choices=['deeplabv3', 'fcn'],
                        help='Модель сегментации')
    parser.add_argument('--smoothing', type=str, default='window',
                        choices=['window', 'median', 'motion'],
                        help='Метод сглаживания')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Размер окна сглаживания')
    parser.add_argument('--target_class', type=int, default=15,
                        help='Целевой класс для сегментации (15=person)')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Вариант A: Система временной стабилизации масок ===")
    print(f"Видеофайл: {args.video}")
    print()
    
    # Создаем выходную папку
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Устройство
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    print()
    
    # Шаг 1: Читаем видео
    print("1. Чтение видео...")
    frames, frame_indices, fps = read_clip(
        args.video,
        num_frames=args.num_frames,
        stride=1
    )
    print(f"   Прочитано {len(frames)} кадров")
    print()
    
    # Шаг 2: Загружаем модель сегментации
    print(f"2. Загрузка модели сегментации ({args.model})...")
    model, transform = load_segmentation_model(args.model)
    model = model.to(device)
    print("   Модель загружена")
    print()
    
    # Шаг 3: Сегментация кадров
    print("3. Сегментация кадров...")
    original_masks = segment_frames(
        frames, model, transform, device, target_class=args.target_class
    )
    print(f"   Получено {len(original_masks)} масок")
    print()
    
    # Шаг 4: Оценка нестабильности оригинальных масок
    print("4. Оценка нестабильности оригинальных масок...")
    original_metrics = compute_mask_stability(original_masks, frames)
    original_boundary_jitter = compute_boundary_jitter(original_masks)
    original_iou = compute_temporal_iou(original_masks)
    
    print("   Метрики оригинальных масок:")
    print(f"     Средний IoU: {original_metrics['mean_iou']:.4f}")
    print(f"     Min IoU: {original_metrics['min_iou']:.4f}")
    print(f"     Std IoU: {original_metrics['iou_std']:.4f}")
    print(f"     Среднее изменение: {original_metrics['mean_change']:.4f}")
    print(f"     Граничное дрожание: {original_boundary_jitter:.2f}")
    print(f"     Мерцающие пиксели на кадр: {original_metrics['avg_flicker_per_frame']:.2f}")
    print()
    
    # Шаг 5: Применяем сглаживание
    print(f"5. Применение сглаживания (метод: {args.smoothing}, window={args.window_size})...")
    
    if args.smoothing == 'window':
        smoothed_masks = smooth_masks_temporal_window(original_masks, args.window_size)
    elif args.smoothing == 'median':
        smoothed_masks = smooth_masks_median(original_masks, args.window_size)
    elif args.smoothing == 'motion':
        smoothed_masks = smooth_masks_motion_aware(original_masks, frames, args.window_size)
    
    print("   Сглаживание завершено")
    print()
    
    # Шаг 6: Оценка после сглаживания
    print("6. Оценка после сглаживания...")
    smoothed_metrics = compute_mask_stability(smoothed_masks, frames)
    smoothed_boundary_jitter = compute_boundary_jitter(smoothed_masks)
    smoothed_iou = compute_temporal_iou(smoothed_masks)
    
    print("   Метрики сглаженных масок:")
    print(f"     Средний IoU: {smoothed_metrics['mean_iou']:.4f}")
    print(f"     Min IoU: {smoothed_metrics['min_iou']:.4f}")
    print(f"     Std IoU: {smoothed_metrics['iou_std']:.4f}")
    print(f"     Среднее изменение: {smoothed_metrics['mean_change']:.4f}")
    print(f"     Граничное дрожание: {smoothed_boundary_jitter:.2f}")
    print(f"     Мерцающие пиксели на кадр: {smoothed_metrics['avg_flicker_per_frame']:.2f}")
    print()
    
    # Шаг 7: Сравнение
    print("7. Сравнение До/После:")
    print()
    print(f"   IoU improvement: {(smoothed_metrics['mean_iou'] - original_metrics['mean_iou']):.4f}")
    print(f"   Стабильность (↓ std): {original_metrics['iou_std']:.4f} → {smoothed_metrics['iou_std']:.4f}")
    print(f"   Изменения (↓): {original_metrics['mean_change']:.4f} → {smoothed_metrics['mean_change']:.4f}")
    print(f"   Дрожание (↓): {original_boundary_jitter:.2f} → {smoothed_boundary_jitter:.2f}")
    print(f"   Мерцание (↓): {original_metrics['avg_flicker_per_frame']:.2f} → {smoothed_metrics['avg_flicker_per_frame']:.2f}")
    print()
    
    # Визуализации
    print("8. Создание визуализаций...")
    
    # График IoU
    plot_metrics(
        {'Original IoU': original_iou, 'Smoothed IoU': smoothed_iou},
        title='Temporal IoU между соседними кадрами',
        save_path=str(output_dir / 'iou_comparison.png')
    )
    
    # Сравнение масок
    visualize_masks_comparison(
        frames[::10],  # Каждый 10-й кадр
        [original_masks[i] for i in range(0, len(original_masks), 10)],
        [smoothed_masks[i] for i in range(0, len(smoothed_masks), 10)],
        save_path=str(output_dir / 'masks_comparison.png')
    )
    
    # Карта нестабильности
    stability_map_orig = visualize_stability_map(original_masks)
    stability_map_smooth = visualize_stability_map(smoothed_masks)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(stability_map_orig, cmap='hot')
    ax1.set_title('Карта нестабильности (оригинал)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(stability_map_smooth, cmap='hot')
    ax2.set_title('Карта нестабильности (сглаженная)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Визуализации сохранены в {output_dir}/")
    print()
    
    # Технический вывод
    print("="*60)
    print("ТЕХНИЧЕСКИЙ ВЫВОД:")
    print("="*60)
    print()
    print(f"Метод сглаживания: {args.smoothing}")
    print()
    print("Когда сглаживание помогает:")
    print("  - Уменьшает мерцание масок между кадрами")
    print("  - Повышает временную согласованность (↑ IoU)")
    print("  - Снижает дрожание границ объекта")
    print()
    print("Когда сглаживание ухудшает:")
    print("  - При резких движениях объекта (размывает границы)")
    print("  - При изменении масштаба или появлении/исчезновении объекта")
    print("  - Добавляет задержку (latency) в обработку")
    print()
    print("Параметры:")
    print(f"  - Размер окна {args.window_size} кадров")
    if args.smoothing == 'motion':
        print("  - Motion-aware сглаживание учитывает оптический поток")
        print("  - Меньше размывает быстро движущиеся объекты")
    print()
    
    print("=== Вариант A выполнен успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())

