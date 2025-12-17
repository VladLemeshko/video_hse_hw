# Результаты выполнения заданий

**Дата:** Wed Dec 17 12:49:27 UTC 2025  
**Лог выполнения:** logs/run_20251217_123202.log

## Структура результатов

```
outputs/
├── hw1/
│   ├── outputs/hw1/task01_frames.png
│   ├── outputs/hw1/task05_prefetch.png
│   ├── outputs/hw1/task10_report.txt
│   ├── outputs/hw1/task03_throughput.png
│   ├── outputs/hw1/task09_fps_stability.png
│   ├── outputs/hw1/task04_profiling.png
│
└── hw2/
    ├── variant_a/
    │   ├── outputs/hw2/variant_a/motion/stability_maps.png
    │   ├── outputs/hw2/variant_a/motion/masks_comparison.png
    │   ├── outputs/hw2/variant_a/motion/iou_comparison.png
    │   ├── outputs/hw2/variant_a/window/stability_maps.png
    │   ├── outputs/hw2/variant_a/window/masks_comparison.png
    │   ├── outputs/hw2/variant_a/window/iou_comparison.png
    └── variant_b/
        ├── outputs/hw2/variant_b/error_analysis_Forward.png
        ├── outputs/hw2/variant_b/error_analysis_Bidirectional.png
        ├── outputs/hw2/variant_b/comparison_all_methods.png
        ├── outputs/hw2/variant_b/error_analysis_Full_Pipeline.png
        ├── outputs/hw2/variant_b/initial_mask.png
```

## Описание результатов

### HW1: Видео-пайплайн

- **task01_frames.png** - Визуализация декодированных кадров
- **task03_throughput.png** - График зависимости throughput от num_workers
- **task04_profiling.png** - Время выполнения этапов пайплайна
- **task05_prefetch.png** - Сравнение FPS и jitter с/без prefetch
- **task08_gpu_preprocess.png** - Сравнение CPU vs GPU препроцессинга
- **task09_fps_stability.png** - Графики стабильности FPS
- **task10_report.txt** - Метрики real-time пайплайна

### HW2 Вариант A: Стабилизация масок

- **iou_comparison.png** - Temporal IoU (до/после сглаживания)
- **masks_comparison.png** - Визуальное сравнение масок
- **stability_maps.png** - Карты нестабильности

### HW2 Вариант B: VOS

- **initial_mask.png** - Начальная маска на кадре 0
- **comparison_all_methods.png** - Сравнение 4 методов переноса
- **error_analysis_*.png** - Анализ ошибок для каждого метода

## Команды для воспроизведения

```bash
# Полный запуск с логами
bash run_and_save_logs.sh

# Или отдельные задания
python hw1/task01_decoder.py --video data/test.mp4
python hw2/variant_a/stabilization.py --video data/test.mp4 --num_frames 30 --use_cuda
```

## Примечания

- Все графики в формате PNG, 150 DPI
- Метрики в текстовых файлах (.txt)
- Полный вывод консоли в logs/run_20251217_123202.log
