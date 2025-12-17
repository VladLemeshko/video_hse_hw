#!/bin/bash

echo "========================================="
echo "Запуск заданий"
echo "========================================="
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Проверяем окружение
if [ ! -d "venv_video" ]; then
    echo "Виртуальное окружение не найдено!"
    echo "Запустите сначала: bash setup_env.sh"
    exit 1
fi

# Активируем окружение
echo -e "${GREEN}Активация виртуального окружения...${NC}"
source venv_video/bin/activate

# Проверяем наличие данных
if [ ! -d "data" ] || [ -z "$(ls -A data)" ]; then
    echo "Папка data пуста!"
    echo "Загрузите тестовые видео в папку data/"
    exit 1
fi

# Определяем путь к тестовому видео
VIDEO_DIR="data"
TEST_VIDEO=$(ls data/*.mp4 | head -n 1)

if [ -z "$TEST_VIDEO" ]; then
    echo "Не найдено видеофайлов в папке data/"
    exit 1
fi

echo -e "${BLUE}Используется тестовое видео: $TEST_VIDEO${NC}"
echo ""

# ==========================================
# Домашнее задание 1
# ==========================================

echo ""
echo "========================================="
echo "ДОМАШНЕЕ ЗАДАНИЕ 1: Видео-пайплайн"
echo "========================================="
echo ""

read -p "Запустить HW1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python hw1/run_all.py --video_dir $VIDEO_DIR --use_cuda
    
    echo ""
    echo "✓ HW1 завершено. Результаты в outputs/hw1/"
fi

# ==========================================
# Домашнее задание 2 - Вариант A
# ==========================================

echo ""
echo "========================================="
echo "ДОМАШНЕЕ ЗАДАНИЕ 2 - Вариант A"
echo "Система временной стабилизации масок"
echo "========================================="
echo ""

read -p "Запустить HW2 Вариант A? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Тестируем разные методы сглаживания
    
    echo ""
    echo "Метод 1: Временное окно (усреднение)"
    python hw2/variant_a/stabilization.py \
        --video $TEST_VIDEO \
        --num_frames 50 \
        --smoothing window \
        --window_size 5 \
        --output outputs/hw2/variant_a/window \
        --use_cuda
    
    echo ""
    echo "Метод 2: Медианное сглаживание"
    python hw2/variant_a/stabilization.py \
        --video $TEST_VIDEO \
        --num_frames 50 \
        --smoothing median \
        --window_size 5 \
        --output outputs/hw2/variant_a/median \
        --use_cuda
    
    echo ""
    echo "Метод 3: Motion-aware сглаживание"
    python hw2/variant_a/stabilization.py \
        --video $TEST_VIDEO \
        --num_frames 50 \
        --smoothing motion \
        --window_size 5 \
        --output outputs/hw2/variant_a/motion \
        --use_cuda
    
    echo ""
    echo "✓ HW2 Вариант A завершено. Результаты в outputs/hw2/variant_a/"
fi

# ==========================================
# Домашнее задание 2 - Вариант B
# ==========================================

echo ""
echo "========================================="
echo "ДОМАШНЕЕ ЗАДАНИЕ 2 - Вариант B"
echo "VOS с переносом маски"
echo "========================================="
echo ""

read -p "Запустить HW2 Вариант B? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python hw2/variant_b/vos_system.py \
        --video $TEST_VIDEO \
        --num_frames 50 \
        --initial_mask grabcut \
        --output outputs/hw2/variant_b
    
    echo ""
    echo "✓ HW2 Вариант B завершено. Результаты в outputs/hw2/variant_b/"
fi

# ==========================================
# Завершение
# ==========================================

echo ""
echo "========================================="
echo "ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!"
echo "========================================="
echo ""
echo "Результаты сохранены в outputs/"
echo ""

