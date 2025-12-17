# Папка для видео данных

Эта папка предназначена для хранения видео файлов.

## Загрузка тестовых данных

```bash
# Автоматическая загрузка короткого видео
bash download_test_video.sh

# Или вручную
cd data
wget http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4 -O test.mp4
```

## Примечание

Видео файлы не добавляются в Git из-за большого размера.  
Загрузите их локально перед запуском заданий.

