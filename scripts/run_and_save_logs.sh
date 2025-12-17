#!/bin/bash

# Скрипт для запуска всех заданий с сохранением логов

echo "========================================="
echo "Запуск заданий с сохранением логов"
echo "========================================="
echo ""

# Создаем папки
mkdir -p logs outputs/hw1 outputs/hw2/variant_a outputs/hw2/variant_b

# Проверяем окружение
if [ ! -d "venv_video" ]; then
    echo "Виртуальное окружение не найдено!"
    echo "Запустите сначала: bash setup_env.sh"
    exit 1
fi

# Активируем окружение
source venv_video/bin/activate

# Проверяем наличие данных
if [ ! -d "data" ] || [ -z "$(ls -A data/*.mp4 2>/dev/null)" ]; then
    echo "Папка data пуста!"
    echo "Загрузите видео: bash download_test_video.sh"
    exit 1
fi

# Определяем путь к видео
VIDEO_DIR="data"
TEST_VIDEO=$(ls data/*.mp4 | head -n 1)

echo "Используется видео: $TEST_VIDEO"
echo ""

# Дата запуска
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/run_${TIMESTAMP}.log"

echo "Лог сохраняется в: $LOG_FILE"
echo ""

# Функция для запуска с логированием
run_with_log() {
    local title="$1"
    local cmd="$2"
    
    echo "" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "$title" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $title - выполнено" | tee -a "$LOG_FILE"
    else
        echo "✗ $title - ошибка" | tee -a "$LOG_FILE"
    fi
}

# Начало логирования
echo "=========================================" > "$LOG_FILE"
echo "Лог запуска заданий" >> "$LOG_FILE"
echo "Дата: $(date)" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# HW1
read -p "Запустить HW1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_with_log "HW1 - Задача 1: Декодер" \
        "python hw1/task01_decoder.py --video $TEST_VIDEO"
    
    run_with_log "HW1 - Задача 2: Dataset" \
        "python hw1/task02_dataset.py --video_dir $VIDEO_DIR --clip_len 16 --batch_size 4"
    
    run_with_log "HW1 - Задача 3: Параллельная загрузка" \
        "python hw1/task03_parallel.py --video_dir $VIDEO_DIR --num_batches 10"
    
    run_with_log "HW1 - Задача 4: Профилирование" \
        "python hw1/task04_profiling.py --video_dir $VIDEO_DIR --num_batches 10 --use_cuda"
    
    run_with_log "HW1 - Задача 5: Prefetch" \
        "python hw1/task05_prefetch.py --video_dir $VIDEO_DIR --num_batches 30 --use_cuda"
    
    run_with_log "HW1 - Задача 8: GPU препроцессинг" \
        "python hw1/task08_gpu_preprocess.py --video_dir $VIDEO_DIR --num_batches 10 --use_cuda"
    
    run_with_log "HW1 - Задача 9: Стабильность FPS" \
        "python hw1/task09_fps_stability.py --video_dir $VIDEO_DIR --num_iterations 50 --use_cuda"
    
    run_with_log "HW1 - Задача 10: Real-time пайплайн" \
        "python hw1/task10_realtime_pipeline.py --video $TEST_VIDEO --duration 10 --use_cuda"
fi

# HW2 Вариант A
read -p "Запустить HW2 Вариант A? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_with_log "HW2 Вариант A - Сглаживание (window)" \
        "python hw2/variant_a/stabilization.py --video $TEST_VIDEO --num_frames 30 --smoothing window --output outputs/hw2/variant_a/window --use_cuda"
    
    run_with_log "HW2 Вариант A - Сглаживание (motion)" \
        "python hw2/variant_a/stabilization.py --video $TEST_VIDEO --num_frames 30 --smoothing motion --output outputs/hw2/variant_a/motion --use_cuda"
fi

# HW2 Вариант B
read -p "Запустить HW2 Вариант B? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_with_log "HW2 Вариант B - VOS" \
        "python hw2/variant_b/vos_system.py --video $TEST_VIDEO --num_frames 30 --output outputs/hw2/variant_b"
fi

# Завершение
echo "" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "Все задания завершены!" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Результаты:" | tee -a "$LOG_FILE"
echo "  - Выходные файлы: outputs/" | tee -a "$LOG_FILE"
echo "  - Лог выполнения: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Создаем итоговый отчет
REPORT_FILE="logs/RESULTS_SUMMARY.md"
cat > "$REPORT_FILE" << EOF
# Результаты выполнения заданий

**Дата:** $(date)  
**Лог выполнения:** $LOG_FILE

## Структура результатов

\`\`\`
outputs/
├── hw1/
$(find outputs/hw1 -type f 2>/dev/null | sed 's/^/│   ├── /')
│
└── hw2/
    ├── variant_a/
$(find outputs/hw2/variant_a -type f 2>/dev/null | sed 's/^/    │   ├── /')
    └── variant_b/
$(find outputs/hw2/variant_b -type f 2>/dev/null | sed 's/^/        ├── /')
\`\`\`

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

\`\`\`bash
# Полный запуск с логами
bash run_and_save_logs.sh

# Или отдельные задания
python hw1/task01_decoder.py --video $TEST_VIDEO
python hw2/variant_a/stabilization.py --video $TEST_VIDEO --num_frames 30 --use_cuda
\`\`\`

## Примечания

- Все графики в формате PNG, 150 DPI
- Метрики в текстовых файлах (.txt)
- Полный вывод консоли в $LOG_FILE
EOF

echo "✓ Создан отчет: $REPORT_FILE"

echo ""
echo "Готово! Проверьте результаты:"
echo "  - outputs/     - все графики и визуализации"
echo "  - logs/        - логи выполнения и отчет"

