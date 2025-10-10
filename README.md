# Text Autocomplete

Проект для обучения LSTM модели автодополнения текста на основе датасета sentiment140 и сравнения с baseline моделью 
DistilGPT2.

## Структура проекта

```
configs/
└── config.yaml             # Конфигурационный файл

data/
├── tweets.txt              # Исходные данные твитов
├── processed_tweets.json   # Обработанные данные
├── train_texts.json        # Обучающие данные
├── val_texts.json          # Валидационные данные
└── test_texts.json         # Тестовые данные

models/
├── best_model.pt           # Лучшая модель по ROUGE-L метрике
└── model_epoch_N.pt        # Модели после каждой эпохи

src/
├── __init__.py
├── configs.py                    # Конфигурация проекта
├── data_utils.py                 # Утилиты обработки данных
├── next_token_dataset.py         # Dataset класс для обучения
├── lstm_model.py                 # LSTM модель
├── lstm_train.py                 # Функции обучения
├── eval_lstm.py                  # Оценка LSTM модели
├── eval_transformer_pipeline.py  # Оценка трансформера
├── baseline_comparison.py        # Сравнение с baseline моделью (DistilGPT2)
└── model_evaluation.py           # Оценка и сравнение моделей

.gitignore
solution.ipynb              # Основной ноутбук с полным pipeline: обучение, оценка, сравнение
requirements.txt            # Зависимости
```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Скачайте данные и создайте необходимые папки:
```bash
# Скачайте файл с твитами
wget -P data/ https://code.s3.yandex.net/deep-learning/tweets.txt
```

3. Обработанные данные будут созданы автоматически при первом запуске обучения через solution.ipynb

## Использование

Запустите все ячейки в ноутбуке `solution.ipynb`. Ноутбук содержит полный pipeline:
- Загрузка и предобработка данных
- Обучение LSTM модели
- Сравнение с baseline моделью (DistilGPT2)
- Генерация примеров и создание отчетов

## Конфигурация

Все параметры настраиваются в файле `configs/config.yaml`:
Пример:
```yaml
model:
  embed_dim: 256          # Размерность эмбеддингов
  hidden_dim: 256         # Размерность скрытого состояния LSTM
  num_layers: 3           # Количество слоев LSTM

training:
  epochs: 10             # Количество эпох
  lr: 0.0001             # Скорость обучения
  weight_decay: 0.00001  # L2 регуляризация
  batch_size: 32         # Размер батча

data:
  test_size: 0.2         # Доля тестовой выборки
  val_size: 0.5          # Доля валидационной выборки
  random_state: 42       # Seed для воспроизводимости

generation:
  max_new_tokens: 20     # Максимальное количество токенов
  top_k: 20              # Параметр top-k сэмплирования
  temperature: 1.0       # Температура генерации
```

## Результаты

### Модели
Модели сохраняются в папке `models/`:
- `best_model.pt` - лучшая модель по ROUGE-L метрике
- `model_epoch_N.pt` - модели после каждой эпохи

### Визуализация
Графики обучения отображаются автоматически и показывают:
- Потери на обучении и валидации
- ROUGE-L метрику на валидации
- Сравнение производительности моделей
