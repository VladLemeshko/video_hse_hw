"""
Скрипт для последовательного запуска всех заданий HW1
"""

import argparse
import sys
from pathlib import Path
import subprocess


def run_task(task_name, args_dict):
    """
    Запускает задание с заданными аргументами
    
    Args:
        task_name: имя файла задания (без .py)
        args_dict: словарь с аргументами
    """
    print(f"\n{'='*60}")
    print(f"Запуск {task_name}")
    print(f"{'='*60}\n")
    
    task_file = Path(__file__).parent / f"{task_name}.py"
    
    cmd = [sys.executable, str(task_file)]
    
    for key, value in args_dict.items():
        cmd.append(f"--{key}")
        if value is not True:  # Для флагов без значений
            cmd.append(str(value))
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n⚠ Задание {task_name} завершилось с ошибкой")
        return result.returncode
    except Exception as e:
        print(f"\n✗ Ошибка при запуске {task_name}: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Запуск всех заданий HW1')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Папка с видеофайлами')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Использовать CUDA если доступна')
    parser.add_argument('--skip_tasks', type=str, default='',
                        help='Пропустить задания (через запятую, например: 6,7)')
    
    args = parser.parse_args()
    
    # Получаем первый видеофайл для заданий, требующих один файл
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob('*.mp4'))
    
    if len(video_files) == 0:
        print("✗ Не найдено видеофайлов в указанной папке")
        return 1
    
    first_video = str(video_files[0])
    
    # Список пропускаемых заданий
    skip_set = set()
    if args.skip_tasks:
        skip_set = set(int(x.strip()) for x in args.skip_tasks.split(','))
    
    print("=== Запуск всех заданий HW1 ===")
    print(f"Папка с видео: {args.video_dir}")
    print(f"CUDA: {'Да' if args.use_cuda else 'Нет'}")
    if skip_set:
        print(f"Пропускаем задания: {sorted(skip_set)}")
    print()
    
    # Задание 1: Базовый декодер
    if 1 not in skip_set:
        run_task('task01_decoder', {
            'video': first_video,
            'num_frames': 16,
            'stride': 2
        })
    
    # Задание 2: Dataset
    if 2 not in skip_set:
        run_task('task02_dataset', {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'batch_size': 4,
            'num_workers': 2
        })
    
    # Задание 3: Параллельная загрузка
    if 3 not in skip_set:
        run_task('task03_parallel', {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'batch_size': 4,
            'num_batches': 20
        })
    
    # Задание 4: Профилирование
    if 4 not in skip_set:
        task_args = {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'batch_size': 4,
            'num_batches': 20
        }
        if args.use_cuda:
            task_args['use_cuda'] = True
        run_task('task04_profiling', task_args)
    
    # Задание 5: Prefetch
    if 5 not in skip_set:
        task_args = {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'batch_size': 4,
            'num_batches': 50
        }
        if args.use_cuda:
            task_args['use_cuda'] = True
        run_task('task05_prefetch', task_args)
    
    # Задание 6: Pipeline overlap
    if 6 not in skip_set:
        task_args = {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'num_clips': 30
        }
        if args.use_cuda:
            task_args['use_cuda'] = True
        run_task('task06_pipeline_overlap', task_args)
    
    # Задание 7: Аппаратное декодирование
    if 7 not in skip_set:
        run_task('task07_hardware_decode', {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'num_runs': 20
        })
    
    # Задание 8: GPU препроцессинг
    if 8 not in skip_set:
        task_args = {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'batch_size': 4,
            'num_batches': 20
        }
        if args.use_cuda:
            task_args['use_cuda'] = True
        run_task('task08_gpu_preprocess', task_args)
    
    # Задание 9: Стабильность FPS
    if 9 not in skip_set:
        task_args = {
            'video_dir': args.video_dir,
            'clip_len': 16,
            'num_iterations': 100
        }
        if args.use_cuda:
            task_args['use_cuda'] = True
        run_task('task09_fps_stability', task_args)
    
    # Задание 10: Real-time пайплайн
    if 10 not in skip_set:
        task_args = {
            'video': first_video,
            'clip_len': 16,
            'duration': 10
        }
        if args.use_cuda:
            task_args['use_cuda'] = True
        run_task('task10_realtime_pipeline', task_args)
    
    print("\n" + "="*60)
    print("Все задания завершены!")
    print("="*60)
    print("\nРезультаты сохранены в папке outputs/hw1/")
    
    return 0


if __name__ == "__main__":
    exit(main())


