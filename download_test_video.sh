#!/bin/bash

# Скрипт для автоматической загрузки короткого тестового видео

echo "Загрузка короткого тестового видео (30 сек)..."

cd data 2>/dev/null || mkdir -p data && cd data

# Короткие тестовые видео (до 30 сек)
SOURCES=(
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
)

for url in "${SOURCES[@]}"; do
    echo "Попытка загрузки: $url"
    
    if command -v wget &> /dev/null; then
        if wget -q --spider "$url" 2>/dev/null; then
            wget -O test.mp4 "$url" && echo "✓ Загружено!" && exit 0
        fi
    elif command -v curl &> /dev/null; then
        if curl -s -f -I "$url" &> /dev/null; then
            curl -L -o test.mp4 "$url" && echo "✓ Загружено!" && exit 0
        fi
    fi
    
    echo "  Не удалось, пробуем следующий..."
done

# Если ничего не сработало, пробуем через Python
echo "Попытка загрузки через Python..."
python3 << 'EOF'
import urllib.request
import sys

# Короткие видео Google
urls = [
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4',
]

for url in urls:
    try:
        print(f"Попытка: {url}")
        urllib.request.urlretrieve(url, 'test.mp4')
        print("✓ Загружено!")
        sys.exit(0)
    except Exception as e:
        print(f"  Ошибка: {e}")
        continue

print("✗ Не удалось загрузить видео")
print("Загрузите вручную через Upload в Jupyter")
sys.exit(1)
EOF

# Проверяем длительность
if [ -f "test.mp4" ]; then
    echo ""
    echo "Проверка длительности..."
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -i test.mp4 2>&1 | grep Duration || echo "Видео загружено"
    fi
fi

