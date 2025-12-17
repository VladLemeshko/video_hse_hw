# Video Processing - Домашние Задания

Реализация домашних заданий по курсу обработки видео.

## Содержание

- **HW1**: Оптимизация видео-пайплайна (10 задач) - см. [HW1.md](HW1.md)
- **HW2**: Video Object Segmentation и стабилизация масок - см. [HW2.md](HW2.md)

## Структура проекта

```
video/
├── hw1/                    # Задания по видео-пайплайну
│   ├── task01_decoder.py   # Базовый декодер
│   ├── task02_dataset.py   # VideoDataset
│   ├── task03_parallel.py  # Параллельная загрузка
│   ├── task04_profiling.py # Профилирование
│   ├── task05_prefetch.py  # Prefetch и pinned memory
│   ├── task06_pipeline_overlap.py  # Pipeline overlap
│   ├── task07_hardware_decode.py   # Аппаратное декодирование
│   ├── task08_gpu_preprocess.py    # GPU препроцессинг
│   ├── task09_fps_stability.py     # Стабильность FPS
│   ├── task10_realtime_pipeline.py # Real-time пайплайн
│   └── run_all.py          # Запуск всех задач
│
├── hw2/                    # Video Object Segmentation
│   ├── variant_a/          # Стабилизация масок (anti-flicker)
│   │   ├── stabilization.py
│   │   └── metrics.py
│   └── variant_b/          # VOS с переносом маски
│       ├── vos_system.py
│       └── optical_flow.py
│
├── utils/                  # Утилиты
│   ├── video_utils.py      # Работа с видео
│   └── visualization.py    # Визуализация
│
├── data/                   # Папка для видео (не в Git)
├── outputs/                # Результаты (не в Git)
└── requirements.txt        # Зависимости
```

## Установка

### Зависимости

```bash
pip install -r requirements.txt
```

Основные библиотеки:
- PyTorch (с CUDA support)
- PyAV (декодирование видео)
- OpenCV (оптический поток)
- TorchVision (модели сегментации)

### Проверка установки

```bash
python test_setup.py
```

## Запуск

### HW1: Видео-пайплайн

```bash
# Все задачи
python hw1/run_all.py --video_dir data --use_cuda

# Отдельная задача
python hw1/task01_decoder.py --video data/sample.mp4
python hw1/task03_parallel.py --video_dir data --batch_size 4
python hw1/task04_profiling.py --video_dir data --use_cuda
```

### HW2 Вариант A: Стабилизация масок

```bash
python hw2/variant_a/stabilization.py \
    --video data/sample.mp4 \
    --num_frames 100 \
    --smoothing motion \
    --window_size 5 \
    --use_cuda
```

Доступные методы сглаживания:
- `window` - усреднение по временному окну
- `median` - медианный фильтр
- `motion` - сглаживание с учетом оптического потока

### HW2 Вариант B: VOS с переносом маски

```bash
python hw2/variant_b/vos_system.py \
    --video data/sample.mp4 \
    --num_frames 50 \
    --initial_mask grabcut
```

## Результаты

Все результаты сохраняются в папке `outputs/`:

```
outputs/
├── hw1/
│   ├── task03_throughput.png       # График throughput
│   ├── task04_profiling.png        # Профилирование этапов
│   ├── task05_prefetch.png         # Сравнение prefetch
│   ├── task08_gpu_preprocess.png   # CPU vs GPU
│   ├── task09_fps_stability.png    # Стабильность FPS
│   └── task10_report.txt           # Метрики RT пайплайна
│
└── hw2/
    ├── variant_a/
    │   ├── iou_comparison.png      # Temporal IoU
    │   ├── masks_comparison.png    # Сравнение масок
    │   └── stability_maps.png      # Карты нестабильности
    │
    └── variant_b/
        ├── initial_mask.png        # Начальная маска
        ├── comparison_all_methods.png  # Сравнение методов
        └── error_analysis_*.png    # Анализ ошибок
```

## Реализованные оптимизации (HW1)

- ✓ Параллельная загрузка данных (multiprocessing)
- ✓ Prefetch и pinned memory
- ✓ Pipeline overlap (декодирование || инференс)
- ✓ Аппаратное декодирование (NVDEC)
- ✓ GPU препроцессинг
- ✓ CUDA Streams
- ✓ Двухуровневая очередь (frames + clips)

## Реализованные методы (HW2)

### Вариант A:
- Кадр-по-кадр сегментация (DeepLabv3/FCN)
- 3 метода сглаживания масок
- Метрики нестабильности (IoU, мерцание, дрожание)

### Вариант B:
- Начальная маска (GrabCut)
- Оптический поток (Farnebäck)
- Перенос маски (forward/backward/bidirectional)
- Постобработка и анализ ошибков

## Требования

- Python 3.8+
- CUDA 11.0+ (для GPU)
- 16GB RAM
- GPU с 8GB+ VRAM (рекомендуется)

## Примечания

- Папки `data/` и `outputs/` не добавляются в Git (см. `.gitignore`)
- Видео данные нужно загружать отдельно
- Для полного функционала необходима GPU с CUDA
