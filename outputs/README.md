# Результаты выполнения заданий

Эта папка содержит все выходные файлы (графики, визуализации, метрики) после выполнения заданий.

## Структура

```
outputs/
├── hw1/                    # Результаты HW1
│   ├── task01_frames.png
│   ├── task03_throughput.png
│   ├── task04_profiling.png
│   ├── task05_prefetch.png
│   ├── task08_gpu_preprocess.png
│   ├── task09_fps_stability.png
│   └── task10_report.txt
│
└── hw2/                    # Результаты HW2
    ├── variant_a/          # Стабилизация масок
    │   ├── window/         # Метод temporal window
    │   └── motion/         # Метод motion-aware
    └── variant_b/          # VOS
        ├── initial_mask.png
        ├── comparison_all_methods.png
        └── error_analysis_*.png
```

## Воспроизведение

Для генерации этих результатов запустите:

```bash
bash run_and_save_logs.sh
```

Или отдельные задания согласно документации в README.md.

## Для преподавателя

Все результаты доступны в этом репозитории для просмотра и проверки.  
Логи выполнения находятся в папке `logs/`.

