# Скрипты установки и запуска

Эта папка содержит все вспомогательные скрипты проекта.

## Скрипты установки

### `setup_env.sh`
Автоматическая установка виртуального окружения и всех зависимостей.

```bash
bash scripts/setup_env.sh
```

### `test_setup.py`
Проверка корректности установки окружения.

```bash
python scripts/test_setup.py
```

## Скрипты для работы с видео

### `download_test_video.sh`
Автоматическая загрузка тестового видео из нескольких источников.

```bash
bash scripts/download_test_video.sh
```

### `trim_video.sh`
Обрезка видео до нужной длины.

```bash
bash scripts/trim_video.sh input.mp4 output.mp4 30
```

## Скрипты запуска заданий

### `run_and_save_logs.sh` ⭐ (рекомендуется)
Запуск всех заданий с автоматическим сохранением логов.

```bash
bash scripts/run_and_save_logs.sh
```

Создает:
- `logs/run_YYYYMMDD_HHMMSS.log` - полный вывод консоли
- `logs/RESULTS_SUMMARY.md` - итоговый отчет

### `run_on_server.sh`
Интерактивный запуск заданий без логирования.

```bash
bash scripts/run_on_server.sh
```

## Использование

Все скрипты запускаются из **корня проекта**:

```bash
# Правильно ✓
cd video
bash scripts/setup_env.sh

# Неправильно ✗
cd scripts
bash setup_env.sh
```

## Порядок выполнения

1. **Установка:** `bash scripts/setup_env.sh`
2. **Проверка:** `python scripts/test_setup.py`
3. **Данные:** `bash scripts/download_test_video.sh`
4. **Запуск:** `bash scripts/run_and_save_logs.sh`

