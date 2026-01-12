"""
Модель ARIMA для прогнозирования временных рядов
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math
import logging
from models.base_model import BaseModel
from config import config

logger = logging.getLogger(__name__)

class ARIMAModel(BaseModel):
    """ARIMA модель для прогнозирования временных рядов"""
    
    def __init__(self):
        """Инициализация ARIMA модели"""
        super().__init__("ARIMA")
        self.order = config.ARIMA_ORDER
        self.model_fit = None
    
    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """
        Обучение ARIMA модели
        
        Args:
            train_data (pd.DataFrame): Данные для обучения
            test_data (pd.DataFrame): Данные для тестирования
            
        Returns:
            float: RMSE на тестовых данных
        """
        try:
            # Разделение данных
            split_idx = int(len(data) * train_size)
            train = data.iloc[:split_idx]['price']
            test = data.iloc[split_idx:]['price']
            
            logger.info(f"Обучение ARIMA с параметрами {self.order}...")
            
            model = ARIMA(train, order=self.order)
            self.model_fit = model.fit()
            
            # Прогноз на тестовые данные
            predictions = self.model_fit.forecast(steps=len(test))
            
            # Расчёт RMSE
            rmse = math.sqrt(mean_squared_error(test, predictions))
            logger.info(f"ARIMA обучена. RMSE: {rmse:.4f}")
            
            self.trained = True
            return rmse
            
        except Exception as e:
            logger.error(f"Ошибка обучения ARIMA: {str(e)}", exc_info=True)
            return float('inf')
    
    def predict(self, steps: int) -> np.ndarray:
        """Прогнозирование с сглаживанием первой точки"""
        if not self.trained or self.model_fit is None:
            raise ValueError("Модель не обучена")
        
        try:
            logger.info(f"Генерация ARIMA прогноза на {steps} дней...")
            predictions = self.model_fit.forecast(steps=steps).values
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования ARIMA: {str(e)}", exc_info=True)
            raise