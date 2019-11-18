# Россия без Путина

<!-- illustration.png -->

Главная цель проекта создать алгоритм позволяющий с высокой точностью распознавать лицо конкретного человека на видео в реальном времени и при необхоимости скрывать его.

## Установка

Для начала скачиваем репозиторий к себе на компьютер, любым из удобных способов. Например, это можно сделать с помощью команды `git clone`:

```sh
git clone https://github.com/freearhey/russia-without-putin.git
```

Далее заходим в папку с проектом и запускаем установку всех зависимостей:

```sh
# смена директории
cd russia-without-putin

# установка зависимостей
pip install -r requirements.txt
```

Вот и всё, установка закончена!

## Запуск

Чтобы запустить обработку видео трансляции необходимо в скрипт `main.py` через атрибут `--input`  передать лишь ссылку на неё:

```sh
python main.py --input http://uiptv.do.am/1ufc/000000006/playlist.m3u8
```

После чего трансляция автоматически откроется в новом окне:

<!-- screenshot1.png -->

Чтобы сохранить результат обработки у себя на компьютере к вызову скрипта необходимо добавить атрибут `--output` с путем до конечного файла:

```sh
python main.py --input http://uiptv.do.am/1ufc/000000006/playlist.m3u8 --output path/to/output.mp4
```

Вместо прямой трансляции в скрипт можно так же передать и обычное видео:

```sh
python main.py --input path/to/some_video.mp4 --output path/to/output.mp4
```

Прервать запись, как и обработку можно нажатием клавиши 'q'.

## Принцип работы алгоритма

<!-- TODO -->

## Тренировка новой модели

При желании вы можете создать свою собственную модель которая будет настроена для распознавания любых других персон. Для этого в скрипт `train_model.py` необходимо передать путь до папки с позитивными, а так же негативными примерами:

```sh
python train_model.py --dataset path/to/dataset
```

В случае с данным проектом структура такой папки следующая:

```
dataset/
  putin/
  non-putin/
```

Папка `non-putin` в данном случае полностью состоит из фотографий [freearhey/face-dataset](https://github.com/freearhey/face-dataset).

## В планах

- добавить распознавание текста
- добавить распознавание звука
- уменьшить количество "false positive" результатов
- запустить ретрансляцию обработанного видео

## Как помочь?

Если вы нашли какую-то ошибку или у вас есть идея как можно улучшить данный алгоритм, можете написать об этом [сюда](https://github.com/freearhey/russia-without-putin/issues).

## Лицензия

[MIT](LICENSE)