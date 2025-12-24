"""
Утилиты для работы с оптическим потоком
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def compute_optical_flow_farneback(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Вычисляет оптический поток методом Farnebäck
    
    Args:
        frame1: первый кадр (RGB)
        frame2: второй кадр (RGB)
        
    Returns:
        поток формы (H, W, 2)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    return flow


def compute_optical_flow_lucas_kanade(frame1: np.ndarray, 
                                      frame2: np.ndarray,
                                      points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет оптический поток методом Lucas-Kanade (sparse)
    
    Args:
        frame1: первый кадр (RGB)
        frame2: второй кадр (RGB)
        points: точки для отслеживания, если None - используются углы
        
    Returns:
        (новые точки, статус)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    if points is None:
        # Находим углы
        points = cv2.goodFeaturesToTrack(
            gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
    
    if points is None or len(points) == 0:
        return None, None
    
    # Вычисляем поток
    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, points, None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    return new_points, status


def warp_mask_forward(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Переносит маску вперед с помощью оптического потока
    
    Args:
        mask: маска на кадре t (H, W)
        flow: оптический поток от t к t+1 (H, W, 2)
        
    Returns:
        перенесенная маска на кадре t+1
    """
    h, w = flow.shape[:2]
    
    # Создаем сетку координат
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w)
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
    
    # Добавляем поток
    flow_map += flow
    
    # Переносим маску
    warped_mask = cv2.remap(
        mask.astype(np.float32),
        flow_map[:, :, 0],
        flow_map[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped_mask


def warp_mask_backward(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Переносит маску назад с помощью обратного оптического потока
    
    Args:
        mask: маска на кадре t+1 (H, W)
        flow: оптический поток от t+1 к t (H, W, 2)
        
    Returns:
        перенесенная маска на кадре t
    """
    return warp_mask_forward(mask, flow)


def fill_holes_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Заполняет дырки в маске с помощью морфологических операций
    
    Args:
        mask: бинарная маска
        kernel_size: размер ядра
        
    Returns:
        обработанная маска
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Закрытие (closing) - заполняет дырки
    mask_binary = (mask > 0.5).astype(np.uint8)
    filled = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    
    return filled.astype(np.float32)


def smooth_mask_boundaries(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Сглаживает границы маски
    
    Args:
        mask: маска
        kernel_size: размер ядра для размытия
        
    Returns:
        сглаженная маска
    """
    # Гауссово размытие
    smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    return smoothed


def compute_flow_confidence(flow: np.ndarray) -> np.ndarray:
    """
    Вычисляет меру уверенности в оптическом потоке
    
    Args:
        flow: оптический поток (H, W, 2)
        
    Returns:
        карта уверенности (H, W), значения 0-1
    """
    # Магнитуда потока
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Нормализуем
    max_magnitude = np.percentile(magnitude, 95)  # 95-й перцентиль
    
    if max_magnitude > 0:
        confidence = 1.0 - np.clip(magnitude / max_magnitude, 0, 1)
    else:
        confidence = np.ones_like(magnitude)
    
    return confidence


def detect_occlusions(flow_forward: np.ndarray, flow_backward: np.ndarray,
                     threshold: float = 1.0) -> np.ndarray:
    """
    Определяет окклюзии путем проверки согласованности forward-backward потоков
    
    Args:
        flow_forward: прямой поток (H, W, 2)
        flow_backward: обратный поток (H, W, 2)
        threshold: порог для определения окклюзии
        
    Returns:
        бинарная маска окклюзий (H, W)
    """
    h, w = flow_forward.shape[:2]
    
    # Создаем сетку
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Переносим координаты вперед
    x_forward = x + flow_forward[..., 0]
    y_forward = y + flow_forward[..., 1]
    
    # Интерполируем обратный поток в новых позициях
    flow_back_warped_x = cv2.remap(
        flow_backward[..., 0], x_forward, y_forward,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    flow_back_warped_y = cv2.remap(
        flow_backward[..., 1], x_forward, y_forward,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    
    # Вычисляем разницу
    diff_x = flow_forward[..., 0] + flow_back_warped_x
    diff_y = flow_forward[..., 1] + flow_back_warped_y
    diff_magnitude = np.sqrt(diff_x**2 + diff_y**2)
    
    # Окклюзии там, где разница большая
    occlusions = (diff_magnitude > threshold).astype(np.uint8)
    
    return occlusions


