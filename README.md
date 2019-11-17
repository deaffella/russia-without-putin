# Россия без Путина

Это лишь proof-of-concept

## Установка

Для обработки видео сначала необходимо установить фреймворк `ffmpeg`. По ссылке можно найти подходящую для своей операционной системы версию: https://www.ffmpeg.org/download.html

Следующий шаг, установка [OpenCV](https://opencv.org/), [TensorFlow](https://www.tensorflow.org/) и [cvlib](https://github.com/arunponnusamy/cvlib):

```sh
pip install opencv-python tensorflow cvlib
```

## Запуск

Чтобы запустить обрабоку видео необходимо лишь указать путь до него:

```sh
python main.py --input path/to/video.mp4
```

## Тренировка новой модели

# Создание примеров для тренировки

Чтобы ускорить процесс создания позитивных или негативных примеров можно сделать их из кадров какого-нибудь видео.

Для этого в папке с проектом есть отдельный скрипт - `generate_dataset.py`. В него нужно лишь передать путь до папки в которую необходимо сохранять кадры и путь до самого видео, и всё:

```sh
python generate_dataset.py --output path/to/dataset/folder --input path/to/video.mp4
```

# Создание новой модели

Для тренировки новой модели вам необходимо передать в скрипт `train_model.py` путь до папки с позитивными и негативными примерами.

```sh
python train_model.py --dataset path/to/dataset
```

Вот пример структуры такой папки:

```
dataset/
  positive/
  negative/
```

## В планах

- распознавание текста
- распознавание звука