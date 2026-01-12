import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

class Config:
    # Токен бота (берётся из .env)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    
    # Параметры данных
    HISTORY_DAYS = 730  # 2 года данных
    FORECAST_DAYS = 30  # Прогноз на 30 дней
    
    # Параметры разбиения данных
    TRAIN_SIZE = 0.8
    
    # Параметры моделей
    # LSTM
    LSTM_EPOCHS = 30
    LSTM_BATCH_SIZE = 32
    LSTM_LOOK_BACK = 60  # Количество лагов
    LSTM_HIDDEN_SIZE = 32
    LSTM_NUM_LAYERS = 2
    
    # Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_N_LAGS = 30  # Количество лагов
    
    # ARIMA
    ARIMA_ORDER = (1, 1, 1)  # (p, d, q)
    
    # Параметры поиска экстремумов
    EXTREMA_ORDER = 3  # Порядок для определения локальных экстремумов

    # Логирование
    LOG_FILE = "user_requests.log"
    BOT_LOG_FILE = "bot.log"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Визуализация
    PLOT_FIGSIZE = (14, 7)  # Размер графика в дюймах
    PLOT_DPI = 100 # Качество изображения
    
    # Ограничения
    MAX_INVESTMENT = 1_000_000_000.0  # 1 миллиард долларов
    MIN_INVESTMENT = 1.0  # Минимальная сумма
    MIN_PROFIT_THRESHOLD = 1.0  # Минимальный размер прибыли для формирования стратегии

# Экземпляр конфигурации
config = Config()