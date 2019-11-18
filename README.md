# Россия без Путина

Это лишь proof-of-concept.

## Установка

Следующий шаг, установка [OpenCV](https://opencv.org/), [TensorFlow](https://www.tensorflow.org/) и [cvlib](https://github.com/arunponnusamy/cvlib):

```sh
pip install -r requirements.txt
```

## Запуск

Чтобы запустить обрабоку видео необходимо указать путь до него и до места куда следует сохранить результат обработки:

```sh
python main.py --input path/to/video.mp4 --output path/to/output.mp4
```

## Тренировка новой модели

Для создания модели вам необходимо передать в скрипт `train_model.py` путь до папки с позитивными и негативными примерами.

```sh
python train_model.py --dataset path/to/dataset
```

Пример структуры такой папки:

```
dataset/
  putin/
  non-putin/
```

## В планах

- распознавание текста
- распознавание звука