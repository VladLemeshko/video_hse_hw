#!/bin/bash

# Скрипт для обрезки видео до 30 секунд

if [ $# -eq 0 ]; then
    echo "Использование: bash trim_video.sh input.mp4 [output.mp4] [длительность_сек]"
    echo ""
    echo "Примеры:"
    echo "  bash trim_video.sh data/long.mp4"
    echo "  bash trim_video.sh data/long.mp4 data/short.mp4"
    echo "  bash trim_video.sh data/long.mp4 data/short.mp4 15"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}_trimmed.mp4}"
DURATION="${3:-30}"

if [ ! -f "$INPUT" ]; then
    echo "✗ Файл не найден: $INPUT"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "✗ ffmpeg не установлен"
    echo "Установите: apt-get install ffmpeg  или  brew install ffmpeg"
    exit 1
fi

echo "Обрезка видео..."
echo "  Вход: $INPUT"
echo "  Выход: $OUTPUT"
echo "  Длительность: $DURATION секунд"
echo ""

# Пробуем быстрое копирование без перекодирования
if ffmpeg -i "$INPUT" -t "$DURATION" -c copy "$OUTPUT" -y 2>/dev/null; then
    echo "✓ Готово (быстрое копирование)"
else
    # Если не получилось, перекодируем
    echo "Перекодирование..."
    ffmpeg -i "$INPUT" -t "$DURATION" -c:v libx264 -preset fast -crf 23 -c:a aac "$OUTPUT" -y
    echo "✓ Готово (с перекодированием)"
fi

# Показываем информацию
echo ""
echo "Информация о результате:"
ffprobe -v quiet -show_format -show_streams "$OUTPUT" 2>/dev/null | grep -E "duration|width|height|codec_name" | head -5

