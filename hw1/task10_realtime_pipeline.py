"""
Задача 10: Финальное задание - мини-RT пайплайн

Near-real-time пайплайн с двухуровневой очередью и CUDA Streams
"""

import argparse
import sys
from pathlib import Path
import time
import threading
from queue import Queue
from collections import deque
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils import read_clip, get_video_info
from hw1.task04_profiling import DummyModel
import torchvision.transforms.v2 as transforms
import cv2


class RealtimeVideoPipeline:
    """
    Near-real-time видео пайплайн с оптимизациями
    """
    
    def __init__(self, video_source, model, device, clip_len=16, stride=2,
                 frame_queue_size=32, clip_queue_size=4):
        """
        Args:
            video_source: путь к видеофайлу или RTSP URL
            model: модель для инференса
            device: устройство (cuda/cpu)
            clip_len: длина клипа
            stride: шаг между кадрами
            frame_queue_size: размер очереди кадров
            clip_queue_size: размер очереди клипов
        """
        self.video_source = video_source
        self.model = model.to(device)
        self.device = device
        self.clip_len = clip_len
        self.stride = stride
        
        # Двухуровневая очередь
        self.frame_queue = Queue(maxsize=frame_queue_size)
        self.clip_queue = Queue(maxsize=clip_queue_size)
        
        # Флаг остановки
        self.stop_flag = threading.Event()
        
        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # CUDA Stream для асинхронных операций
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
        # Метрики
        self.latencies = deque(maxlen=100)
        self.fps_values = deque(maxlen=100)
        
    def decode_thread(self):
        """Поток для декодирования кадров"""
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"✗ Не удалось открыть {self.video_source}")
            return
        
        frame_count = 0
        
        while not self.stop_flag.is_set():
            ret, frame = cap.read()
            
            if not ret:
                # Перезапускаем видео для непрерывного воспроизведения
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Конвертируем BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                self.frame_queue.put((frame_count, frame_rgb), timeout=0.1)
                frame_count += 1
            except:
                pass  # Очередь заполнена, пропускаем кадр
        
        cap.release()
    
    def clip_assembly_thread(self):
        """Поток для сборки клипов из кадров"""
        frame_buffer = []
        
        while not self.stop_flag.is_set():
            try:
                frame_idx, frame = self.frame_queue.get(timeout=0.1)
                frame_buffer.append((frame_idx, frame))
                
                # Собираем клип когда накопилось достаточно кадров
                if len(frame_buffer) >= self.clip_len * self.stride:
                    # Выбираем кадры с нужным stride
                    clip_frames = [frame_buffer[i][1] 
                                  for i in range(0, len(frame_buffer), self.stride)]
                    clip_frames = clip_frames[:self.clip_len]
                    
                    if len(clip_frames) == self.clip_len:
                        # Собираем в тензор
                        clip_array = np.stack(clip_frames, axis=0)
                        
                        # Конвертируем в torch tensor
                        clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2).float() / 255.0
                        
                        # Применяем трансформации
                        C, T, H, W = clip_tensor.shape
                        transformed = []
                        for t in range(T):
                            frame_t = self.transform(clip_tensor[:, t, :, :])
                            transformed.append(frame_t)
                        
                        clip_transformed = torch.stack(transformed, dim=1)  # (C, T, H, W)
                        
                        # Pinned memory для быстрого переноса на GPU
                        if self.device.type == 'cuda':
                            clip_transformed = clip_transformed.pin_memory()
                        
                        timestamp = time.time()
                        self.clip_queue.put((timestamp, clip_transformed))
                    
                    # Очищаем буфер, оставляя перекрытие
                    frame_buffer = frame_buffer[self.stride * self.clip_len // 2:]
                    
            except:
                pass
    
    def inference_thread(self):
        """Поток для инференса"""
        self.model.eval()
        
        with torch.no_grad():
            while not self.stop_flag.is_set():
                try:
                    timestamp_start, clip = self.clip_queue.get(timeout=0.1)
                    
                    # Перенос на GPU (асинхронный)
                    if self.stream:
                        with torch.cuda.stream(self.stream):
                            clip = clip.unsqueeze(0).to(self.device, non_blocking=True)
                            output = self.model(clip)
                        torch.cuda.synchronize()
                    else:
                        clip = clip.unsqueeze(0).to(self.device)
                        output = self.model(clip)
                    
                    # Измеряем latency
                    latency = (time.time() - timestamp_start) * 1000  # мс
                    self.latencies.append(latency)
                    
                except:
                    pass
    
    def run(self, duration=10):
        """
        Запускает пайплайн
        
        Args:
            duration: длительность работы в секундах
        """
        print("Запуск пайплайна...")
        
        # Запускаем потоки
        threads = [
            threading.Thread(target=self.decode_thread, name="Decode"),
            threading.Thread(target=self.clip_assembly_thread, name="ClipAssembly"),
            threading.Thread(target=self.inference_thread, name="Inference")
        ]
        
        for t in threads:
            t.start()
        
        # Работаем заданное время
        time.sleep(duration)
        
        # Останавливаем
        print("\nОстановка пайплайна...")
        self.stop_flag.set()
        
        for t in threads:
            t.join(timeout=2)
        
        print("Пайплайн остановлен.")
    
    def get_metrics(self):
        """Возвращает метрики производительности"""
        if len(self.latencies) == 0:
            return None
        
        latencies = np.array(self.latencies)
        
        return {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'jitter': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
        }


def main():
    parser = argparse.ArgumentParser(description='Задача 10: Real-time пайплайн')
    parser.add_argument('--video', type=str, required=True, 
                        help='Путь к видеофайлу или RTSP URL')
    parser.add_argument('--clip_len', type=int, default=16, help='Длина клипа')
    parser.add_argument('--duration', type=int, default=20, 
                        help='Длительность работы (сек)')
    parser.add_argument('--frame_queue_size', type=int, default=32,
                        help='Размер очереди кадров')
    parser.add_argument('--clip_queue_size', type=int, default=4,
                        help='Размер очереди клипов')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    
    args = parser.parse_args()
    
    print(f"=== Задача 10: Near-real-time пайплайн ===")
    print(f"Видео источник: {args.video}")
    print()
    
    # Проверяем CUDA
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Получаем информацию о видео
    try:
        if not args.video.startswith('rtsp://'):
            info = get_video_info(args.video)
            print("Информация о видео:")
            print(f"  Разрешение: {info['width']}x{info['height']}")
            print(f"  FPS: {info['fps']:.2f}")
            print(f"  Кадров: {info['num_frames']}")
            print(f"  Кодек: {info['codec']}")
            print()
    except Exception as e:
        print(f"⚠ Не удалось получить информацию о видео: {e}")
        print()
    
    # Создаем модель
    model = DummyModel().to(device)
    
    # Создаем пайплайн
    pipeline = RealtimeVideoPipeline(
        video_source=args.video,
        model=model,
        device=device,
        clip_len=args.clip_len,
        stride=2,
        frame_queue_size=args.frame_queue_size,
        clip_queue_size=args.clip_queue_size
    )
    
    print(f"Конфигурация пайплайна:")
    print(f"  Clip length: {args.clip_len}")
    print(f"  Frame queue size: {args.frame_queue_size}")
    print(f"  Clip queue size: {args.clip_queue_size}")
    print(f"  CUDA Streams: {'Да' if device.type == 'cuda' else 'Нет'}")
    print(f"  Pinned memory: {'Да' if device.type == 'cuda' else 'Нет'}")
    print()
    
    # Запускаем пайплайн
    print(f"Работаем {args.duration} секунд...")
    print()
    
    start_time = time.time()
    pipeline.run(duration=args.duration)
    total_time = time.time() - start_time
    
    # Получаем метрики
    metrics = pipeline.get_metrics()
    
    if metrics:
        print()
        print("✓ Итоговые метрики производительности:")
        print()
        print(f"  Средний latency:  {metrics['avg_latency']:.2f} мс")
        print(f"  P95 latency:      {metrics['p95_latency']:.2f} мс")
        print(f"  P99 latency:      {metrics['p99_latency']:.2f} мс")
        print(f"  Jitter (std):     {metrics['jitter']:.2f} мс")
        print(f"  Min latency:      {metrics['min_latency']:.2f} мс")
        print(f"  Max latency:      {metrics['max_latency']:.2f} мс")
        print()
        
        # Оценка throughput
        num_clips = len(pipeline.latencies)
        throughput = num_clips / total_time
        
        print(f"  Обработано клипов: {num_clips}")
        print(f"  Throughput:       {throughput:.2f} клипов/с")
        print()
        
        # Оценка для real-time
        target_fps = 30  # Целевой FPS
        target_latency = 1000 / target_fps  # мс
        
        print("Оценка для real-time:")
        if metrics['p95_latency'] < target_latency:
            print(f"  ✓ P95 latency ({metrics['p95_latency']:.2f} мс) < "
                  f"целевой ({target_latency:.2f} мс для {target_fps} FPS)")
            print(f"  Система может работать в real-time!")
        else:
            print(f"  ✗ P95 latency ({metrics['p95_latency']:.2f} мс) > "
                  f"целевой ({target_latency:.2f} мс для {target_fps} FPS)")
            print(f"  Требуется дополнительная оптимизация")
        
        print()
        
        # Сохраняем отчет
        output_dir = Path('outputs/hw1')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'task10_report.txt'
        with open(report_path, 'w') as f:
            f.write("=== Real-time Video Pipeline Report ===\n\n")
            f.write(f"Video source: {args.video}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Duration: {total_time:.2f} s\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Clip length: {args.clip_len}\n")
            f.write(f"  Frame queue size: {args.frame_queue_size}\n")
            f.write(f"  Clip queue size: {args.clip_queue_size}\n\n")
            
            f.write("Performance Metrics:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.2f} ms\n")
            f.write(f"\n  Throughput: {throughput:.2f} clips/s\n")
            f.write(f"  Total clips: {num_clips}\n")
        
        print(f"✓ Отчет сохранен в {report_path}")
    else:
        print("✗ Не удалось собрать метрики")
    
    print()
    print("=== Задача 10 выполнена успешно! ===")
    
    return 0


if __name__ == "__main__":
    exit(main())

