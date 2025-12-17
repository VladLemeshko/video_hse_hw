"""
Метрики для оценки временной нестабильности масок
"""

import numpy as np
from typing import List


def compute_temporal_iou(masks: List[np.ndarray]) -> List[float]:
    """
    Вычисляет IoU между соседними масками
    
    Args:
        masks: список масок (каждая - np.ndarray формы (H, W))
        
    Returns:
        список IoU значений между соседними масками
    """
    iou_values = []
    
    for i in range(len(masks) - 1):
        mask1 = masks[i] > 0.5
        mask2 = masks[i + 1] > 0.5
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        iou_values.append(iou)
    
    return iou_values


def compute_mask_stability(masks: List[np.ndarray], 
                          frames: List[np.ndarray]) -> dict:
    """
    Оценивает стабильность масок во времени
    
    Args:
        masks: список масок
        frames: список кадров
        
    Returns:
        dict с метриками стабильности
    """
    # IoU между соседними кадрами
    iou_values = compute_temporal_iou(masks)
    
    # Количество изменений маски (где маска меняется)
    mask_changes = []
    for i in range(len(masks) - 1):
        diff = np.abs(masks[i] - masks[i + 1])
        change_ratio = diff.mean()
        mask_changes.append(change_ratio)
    
    # Количество мерцающих пикселей (меняются туда-сюда)
    flicker_count = 0
    if len(masks) >= 3:
        for i in range(len(masks) - 2):
            # Пиксель мерцает если: был 1 -> стал 0 -> стал 1 (или наоборот)
            m0 = masks[i] > 0.5
            m1 = masks[i + 1] > 0.5
            m2 = masks[i + 2] > 0.5
            
            # Мерцание типа 1-0-1
            flicker1 = np.logical_and(m0, np.logical_and(~m1, m2))
            # Мерцание типа 0-1-0
            flicker2 = np.logical_and(~m0, np.logical_and(m1, ~m2))
            
            flicker = np.logical_or(flicker1, flicker2).sum()
            flicker_count += flicker
    
    return {
        'mean_iou': np.mean(iou_values) if iou_values else 0.0,
        'min_iou': np.min(iou_values) if iou_values else 0.0,
        'iou_std': np.std(iou_values) if iou_values else 0.0,
        'mean_change': np.mean(mask_changes) if mask_changes else 0.0,
        'total_flicker_pixels': flicker_count,
        'avg_flicker_per_frame': flicker_count / max(1, len(masks) - 2)
    }


def compute_boundary_jitter(masks: List[np.ndarray]) -> float:
    """
    Вычисляет дрожание границ маски между кадрами
    
    Args:
        masks: список масок
        
    Returns:
        среднее отклонение границ
    """
    import cv2
    
    boundary_diffs = []
    
    for i in range(len(masks) - 1):
        # Находим границы
        mask1_binary = (masks[i] > 0.5).astype(np.uint8)
        mask2_binary = (masks[i + 1] > 0.5).astype(np.uint8)
        
        # Вычисляем границы через эрозию
        kernel = np.ones((3, 3), np.uint8)
        boundary1 = mask1_binary - cv2.erode(mask1_binary, kernel, iterations=1)
        boundary2 = mask2_binary - cv2.erode(mask2_binary, kernel, iterations=1)
        
        # Разница между границами
        diff = np.abs(boundary1.astype(float) - boundary2.astype(float))
        boundary_diffs.append(diff.sum())
    
    return np.mean(boundary_diffs) if boundary_diffs else 0.0


def visualize_stability_map(masks: List[np.ndarray]) -> np.ndarray:
    """
    Создает карту нестабильности (где маски часто меняются)
    
    Args:
        masks: список масок
        
    Returns:
        карта нестабильности (H, W), значения 0-1
    """
    if len(masks) < 2:
        return np.zeros_like(masks[0])
    
    # Суммируем изменения по всем кадрам
    stability_map = np.zeros_like(masks[0], dtype=float)
    
    for i in range(len(masks) - 1):
        diff = np.abs(masks[i] - masks[i + 1])
        stability_map += diff
    
    # Нормализуем
    stability_map = stability_map / (len(masks) - 1)
    
    return stability_map

