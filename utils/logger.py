"""
Система логирования для бота
"""

import logging
from logging.handlers import RotatingFileHandler
from config import config
import os
import datetime

def setup_logging():
    """Настройка логирования уровня приложения"""
    # Создание директории для логов, если её нет
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Основной файловый обработчик с ротацией
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, config.BOT_LOG_FILE),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    
    # Настройка корневого логгера
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    # Подавление избыточных логов от библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

def log_user_request(
    user_id: int,
    ticker: str,
    amount: float,
    best_model: str,
    metric: float,
    profit: float
):
    """
    Логирование пользовательского запроса в отдельный файл
    
    Формат записи:
    timestamp|user_id|ticker|amount|best_model|metric|profit
    """
    # Создание директории для логов, если её нет
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    user_log_path = os.path.join(log_dir, "user_requests.log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"{timestamp}|{user_id}|{ticker}|{amount:.2f}|"
        f"{best_model}|{metric:.4f}|{profit:.2f}\n"
    )
    
    try:
        with open(user_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        logging.debug(f"Записан запрос пользователя в {user_log_path}")
    except Exception as e:
        logging.error(f"Ошибка записи в лог-файл пользовательских запросов: {e}")