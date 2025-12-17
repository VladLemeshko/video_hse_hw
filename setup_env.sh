#!/bin/bash

echo "=== Настройка окружения ==="

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверка Python версии
echo "Проверка версии Python..."
python3 --version

# Создание виртуального окружения
echo -e "${GREEN}Создание виртуального окружения...${NC}"
python3 -m venv venv_video

# Активация окружения
echo -e "${GREEN}Активация окружения...${NC}"
source venv_video/bin/activate

# Обновление pip
echo -e "${GREEN}Обновление pip...${NC}"
pip install --upgrade pip setuptools wheel

# Установка PyTorch с CUDA поддержкой
echo -e "${GREEN}Установка PyTorch с CUDA 11.8...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка остальных зависимостей
echo -e "${GREEN}Установка остальных зависимостей...${NC}"
pip install -r requirements.txt

# Проверка CUDA
echo -e "${GREEN}Проверка доступности CUDA...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Создание необходимых папок
echo -e "${GREEN}Создание структуры папок...${NC}"
mkdir -p data outputs/hw1 outputs/hw2/variant_a outputs/hw2/variant_b
mkdir -p logs/tensorboard

echo -e "${GREEN}=== Установка завершена! ===${NC}"
echo ""
echo "Активация: source venv_video/bin/activate"
echo "Проверка: python test_setup.py"

