"""
Сервис для загрузки и обработки финансовых данных
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from config import config

logger = logging.getLogger(__name__)

def validate_ticker(ticker: str) -> bool:
    """
    Валидация тикера компании
    
    Проверяет:
    - Наличие символов
    - Длина не более 10 символов
    - Только буквы и цифры
    
    Args:
        ticker (str): Тикер для проверки
        
    Returns:
        bool: True если тикер валидный, иначе False
    """
    if not ticker or len(ticker) > 10:
        return False
    return ticker.isalnum()

def load_stock_data(ticker: str, days: int = 730) -> pd.DataFrame:
    """
    Загрузка исторических данных акций с Yahoo Finance
    
    Args:
        ticker (str): Тикер компании (например, AAPL)
        days (int): Количество дней для загрузки данных
        
    Returns:
        pd.DataFrame: DataFrame с колонкой 'price' или None при ошибке
        
    Raises:
        ValueError: Если тикер некорректный или данные не загружены
    """
    # Валидация тикера перед загрузкой
    if not validate_ticker(ticker):
        logger.error(f"Попытка загрузки с некорректным тикером: {ticker}")
        raise ValueError(f"Некорректный тикер: {ticker}. Используйте формат вроде AAPL, MSFT")
    
    try:
        # Расчёт временного диапазона
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Загрузка данных для {ticker} с {start_date.date()} по {end_date.date()}")
        
        # Загрузка данных с Yahoo Finance
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            progress=False,
            timeout=60
        )
        
        # Проверка успешности загрузки
        if data.empty:
            logger.error(f"Пустой результат для тикера {ticker}")
            raise ValueError(f"Данные для тикера {ticker} не найдены. Проверьте правильность написания.")
        
        # Обработка данных
        df = data[['Close']].copy()
        df.columns = ['price']
        df.index.name = 'date'
        
        logger.info(f"Успешно загружено {len(df)} записей для {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке данных для {ticker}: {str(e)}", exc_info=True)
        raise ValueError(
            f"Не удалось загрузить данные для {ticker}. "
            "Возможные причины:\n"
            "• Неверный тикер\n"
            "• Проблемы с подключением к Yahoo Finance\n"
            "• Тикер не торгуется на бирже"
        ) from e