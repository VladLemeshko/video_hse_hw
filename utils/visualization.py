"""
Утилиты для визуализации
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Optional


def visualize_frames(frames: np.ndarray, title: str = "Frames", save_path: Optional[str] = None):
    """
    Визуализирует последовательность кадров
    
    Args:
        frames: массив кадров (T, H, W, 3)
        title: заголовок
        save_path: путь для сохранения (если None, показывает)
    """
    num_frames = min(len(frames), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_frames):
        axes[i].imshow(frames[i])
        axes[i].set_title(f'Frame {i}')
        axes[i].axis('off')
    
    # Скрываем неиспользуемые subplot'ы
    for i in range(num_frames, 8):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_mask_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Накладывает маску на кадр
    
    Args:
        frame: кадр RGB (H, W, 3)
        mask: маска (H, W), значения 0-1
        alpha: прозрачность маски
        
    Returns:
        кадр с наложенной маской
    """
    # Нормализуем маску
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    # Создаем цветную маску (красная)
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 0] = mask_norm * 255  # Красный канал
    
    # Смешиваем
    overlay = cv2.addWeighted(frame, 1 - alpha, mask_color.astype(np.uint8), alpha, 0)
    
    return overlay


def visualize_masks_comparison(frames: np.ndarray, 
                               masks_original: List[np.ndarray],
                               masks_smoothed: List[np.ndarray],
                               save_path: Optional[str] = None):
    """
    Сравнивает оригинальные и сглаженные маски
    
    Args:
        frames: кадры (T, H, W, 3)
        masks_original: оригинальные маски
        masks_smoothed: сглаженные маски
        save_path: путь для сохранения
    """
    num_samples = min(4, len(frames))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        idx = i * (len(frames) // num_samples)
        
        # Оригинальный кадр
        axes[i, 0].imshow(frames[idx])
        axes[i, 0].set_title(f'Frame {idx}')
        axes[i, 0].axis('off')
        
        # Оригинальная маска
        overlay_orig = visualize_mask_overlay(frames[idx], masks_original[idx])
        axes[i, 1].imshow(overlay_orig)
        axes[i, 1].set_title('Original Mask')
        axes[i, 1].axis('off')
        
        # Сглаженная маска
        overlay_smooth = visualize_mask_overlay(frames[idx], masks_smoothed[idx])
        axes[i, 2].imshow(overlay_smooth)
        axes[i, 2].set_title('Smoothed Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(metrics: dict, title: str = "Metrics", save_path: Optional[str] = None):
    """
    Строит графики метрик
    
    Args:
        metrics: словарь с метриками (имя -> список значений)
        title: заголовок
        save_path: путь для сохранения
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Frame')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_optical_flow(flow: np.ndarray, save_path: Optional[str] = None):
    """
    Визуализирует оптический поток
    
    Args:
        flow: оптический поток (H, W, 2)
        save_path: путь для сохранения
    """
    h, w = flow.shape[:2]
    
    # Создаем HSV представление
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Вычисляем магнитуду и угол
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Угол -> Hue
    hsv[..., 0] = ang * 180 / np.pi / 2
    # Полная насыщенность
    hsv[..., 1] = 255
    # Магнитуда -> Value
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Конвертируем в RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title('Optical Flow Visualization')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

