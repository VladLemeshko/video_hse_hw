"""
Утилиты для работы с видео
"""

import av
import numpy as np
import cv2
from typing import Tuple, Optional


def read_clip(filename: str, start: int = 0, num_frames: int = 16, stride: int = 2) -> np.ndarray:
    """
    Читает клип из видеофайла с помощью PyAV
    
    Args:
        filename: путь к видеофайлу
        start: начальный кадр
        num_frames: количество кадров для чтения
        stride: шаг между кадрами
        
    Returns:
        numpy.ndarray формы (T, H, W, 3)
    """
    container = av.open(filename)
    video_stream = container.streams.video[0]
    
    frames = []
    frame_indices = []
    
    for i, frame in enumerate(container.decode(video=0)):
        if i < start:
            continue
            
        if (i - start) % stride == 0:
            # Конвертируем в numpy array
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
            frame_indices.append(i)
            
        if len(frames) >= num_frames:
            break
    
    container.close()
    
    if len(frames) == 0:
        raise ValueError(f"Не удалось прочитать кадры из {filename}")
    
    return np.stack(frames, axis=0), frame_indices, video_stream.average_rate


def get_video_info(filename: str) -> dict:
    """
    Получает информацию о видеофайле
    
    Args:
        filename: путь к видеофайлу
        
    Returns:
        dict с информацией о видео
    """
    container = av.open(filename)
    video_stream = container.streams.video[0]
    
    info = {
        'duration': float(container.duration / av.time_base),
        'num_frames': video_stream.frames,
        'fps': float(video_stream.average_rate),
        'width': video_stream.width,
        'height': video_stream.height,
        'codec': video_stream.codec_context.name,
    }
    
    container.close()
    return info


def optical_flow_farneback(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """
    Вычисляет оптический поток методом Farnebäck
    
    Args:
        prev_frame: предыдущий кадр (RGB)
        next_frame: следующий кадр (RGB)
        
    Returns:
        поток формы (H, W, 2) с компонентами (dx, dy)
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    return flow


def warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Переносит маску с помощью оптического потока
    
    Args:
        mask: бинарная маска (H, W)
        flow: оптический поток (H, W, 2)
        
    Returns:
        перенесенная маска (H, W)
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
        cv2.INTER_LINEAR
    )
    
    return warped_mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Вычисляет IoU между двумя масками
    
    Args:
        mask1: первая маска
        mask2: вторая маска
        
    Returns:
        IoU значение
    """
    intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
    union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


