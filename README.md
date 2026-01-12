# Телеграм-бот для прогнозирования акций на основе временных рядов

Телеграм-бот для анализа и прогнозирования цен акций с использованием ML

## Структура проекта:
```
bot_project/
├── main.py                      # Точка входа
├── config.py                    # Конфигурация приложения
├── requirements.txt             # Зависимости проекта
├── README.md                    # Документация
├── .env                         # Переменные окружения (токен бота)
├── logs.txt                     # Логи пользовательских запросов
├── bot.log                      # Системные логи
├── temp_plots/                  # Временные файлы графиков
│
├── models/
│   ├── base_model.py            # Абстрактный базовый класс для моделей
│   ├── random_forest.py         # Random Forest модель с лаговыми признаками
│   ├── arima_model.py           # ARIMA статистическая модель
│   └── lstm_model.py            # LSTM нейросетевая модель
│
├── services/
│   ├── data_loader.py           # Загрузка данных с Yahoo Finance
│   ├── model_selector.py        # Обучение и выбор лучшей модели
│   ├── trading_analyzer.py      # Анализ торговых стратегий
│   └── visualizer.py            # Визуализация прогнозов
│
├── utils/
│   ├── helpers.py               # Вспомогательные функции
│   └── logger.py                # Настройка логирования
│
└── bot/
    └── handlers.py              # Обработчики команд Telegram
```

## Конфигурирование

Основные настройки задаются в файле config.py, но токен бота должен храниться в .env файле:
```python
TELEGRAM_TOKEN=ваш_токен_бота_здесь
```

Пример параметров в config.py:
```python
# Параметры данных
HISTORY_DAYS = 730  # 2 года исторических данных
FORECAST_DAYS = 30  # Прогноз на 30 дней

# Параметры моделей
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
RF_N_ESTIMATORS = 100
ARIMA_ORDER = (5, 1, 2)
```

## Запуск бота

1. Клонирование репозитория
```bash
git clone https://github.com/ваш_логин/stock_predictor_bot.git
cd stock_predictor_bot
```
2. Создание виртуального окружения (рекомендуется)
```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.\.venv\Scripts\activate
```

3. Установка зависимостей
```bash
pip install -r requirements.txt
```

4. Настройка токена бота
```python
TELEGRAM_TOKEN=YOUR_BOT_TOKEN
```
Получите токен у @BotFather в Telegram и замените YOUR_BOT_TOKEN

5. Запуск бота
```bash
python main.py
```

## Использование бота

1. Отправьте команду /start для начала анализа
2. Введите тикер компании (например, AAPL, MSFT, TSLA)
3. Введите сумму для условной инвестиции в долларах
4. Дождитесь генерации прогноза (1-2 минуты)
5. Получите:
    - График с прогнозом на 30 дней
    - Анализ изменения цены
    - Торговые рекомендации
    - Расчёт потенциальной прибыли

## Результаты прогнозирования носят исключительно учебный характер и не являются финансовой рекомендацией
