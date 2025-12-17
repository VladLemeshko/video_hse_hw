# Инструкция по установке и запуску

> 📖 Основная документация: [README.md](../README.md)

## Подключение к серверу

**URL:** http://176.109.92.50:8753  
**Пароль:** `'[ ghjrfxe`  
**GPU:** NVIDIA A100

## Установка

### 1. Клонирование проекта

```bash
# Откройте Terminal в Jupyter
cd ~
git clone <your-repo-url> video
cd video
```

### 2. Создание окружения

```bash
# Автоматическая установка
bash scripts/setup_env.sh

# Или вручную
python3 -m venv venv_video
source venv_video/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Проверка

```bash
source venv_video/bin/activate
python scripts/test_setup.py
```

Должно показать:
- ✓ Все импорты работают
- ✓ CUDA доступна
- ✓ GPU: NVIDIA A100

## Загрузка данных

### Автоматическая загрузка (рекомендуется)

```bash
# Скрипт автоматически загрузит короткое видео (~15 сек)
bash download_test_video.sh
```

### Ручная загрузка коротких видео

```bash
cd data

# Короткие видео от Google (~15 сек, 720p, ~2-3 MB)
wget http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4 -O test.mp4
# или
wget http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4 -O test.mp4

# Через curl
curl -L "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" -o test.mp4

# Через Python
python3 -c "import urllib.request; urllib.request.urlretrieve('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4', 'test.mp4')"
```

### Обрезка длинного видео до 30 секунд

```bash
# Автоматический скрипт
bash scripts/trim_video.sh data/long_video.mp4 data/test.mp4 30

# Вручную через ffmpeg
ffmpeg -i data/long_video.mp4 -t 30 -c copy data/test.mp4

# Обрезать текущее test.mp4 до 30 сек
bash scripts/trim_video.sh data/test.mp4 data/test_short.mp4 30
mv data/test_short.mp4 data/test.mp4
```

## Запуск заданий

### Быстрый запуск с сохранением логов (рекомендуется)

```bash
source venv_video/bin/activate
bash scripts/run_and_save_logs.sh
```

Этот скрипт:
- Запускает все задания
- Сохраняет полный вывод консоли в `logs/`
- Создает итоговый отчет `logs/RESULTS_SUMMARY.md`
- Все результаты будут в `outputs/` и `logs/`

### Обычный запуск (без логов)

```bash
bash scripts/run_on_server.sh
```

Скрипт интерактивно предложит выбрать задания.

### HW1

```bash
# Все задачи
python hw1/run_all.py --video_dir data --use_cuda

# Отдельные задачи
python hw1/task01_decoder.py --video data/test.mp4
python hw1/task03_parallel.py --video_dir data
python hw1/task04_profiling.py --video_dir data --use_cuda
```

### HW2 Вариант A

```bash
# Разные методы сглаживания
python hw2/variant_a/stabilization.py \
    --video data/test.mp4 \
    --num_frames 50 \
    --smoothing window \
    --use_cuda

python hw2/variant_a/stabilization.py \
    --video data/test.mp4 \
    --num_frames 50 \
    --smoothing motion \
    --use_cuda
```

### HW2 Вариант B

```bash
python hw2/variant_b/vos_system.py \
    --video data/test.mp4 \
    --num_frames 50
```

## Просмотр результатов

Результаты сохраняются в `outputs/`:
- `outputs/hw1/` - графики и метрики HW1
- `outputs/hw2/variant_a/` - визуализации вариант A
- `outputs/hw2/variant_b/` - визуализации вариант B

Открывайте `.png` файлы прямо в Jupyter File Browser.

## Мониторинг GPU

```bash
# В отдельном терминале
watch -n 1 nvidia-smi
```

## Troubleshooting

### CUDA out of memory

```bash
# Уменьшите batch_size или num_frames
python hw1/task03_parallel.py --video_dir data --batch_size 2
python hw2/variant_a/stabilization.py --video data/test.mp4 --num_frames 30
```

### Модуль не найден

```bash
source venv_video/bin/activate
pip install -r requirements.txt
```

### Видео не декодируется

```bash
# Конвертируйте в поддерживаемый формат
ffmpeg -i data/input.mp4 -c:v libx264 -preset fast data/output.mp4
```

### Не удается скачать видео

```bash
# Попробуйте другие источники или используйте curl вместо wget
curl -L "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" -o data/test.mp4

# Или используйте Python
python3 -c "
import urllib.request
urllib.request.urlretrieve(
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'data/test.mp4'
)
print('Downloaded!')
"
```

## Git workflow

```bash
# Инициализация
git init
git add .
git commit -m "Initial commit"

# Создайте репозиторий на GitHub и свяжите
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main

# Обновления
git add .
git commit -m "Update"
git push
```

## Полезные команды

```bash
# Активация окружения
source venv_video/bin/activate

# Проверка CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Размер outputs
du -sh outputs/

# Архивирование результатов
tar -czf results.tar.gz outputs/
```

