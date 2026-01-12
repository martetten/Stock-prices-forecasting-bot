"""
Сервис для выбора лучшей модели прогнозирования
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import logging
from models.random_forest import RandomForestModel
from models.arima_model import ARIMAModel
from models.lstm_model import LSTMModel
from models.base_model import BaseModel
from config import config

logger = logging.getLogger(__name__)

class ModelSelector:
    """Класс для обучения и выбора лучшей модели прогнозирования"""
    
    def __init__(self):
        """Инициализация трёх моделей разных типов"""
        self.models = {
            'Random Forest': RandomForestModel(),
            'ARIMA': ARIMAModel(),
            'LSTM': LSTMModel()
        }
        self.best_model_name = None
        self.best_rmse = float('inf')
        self.all_results = {}
    
    def train_and_evaluate(self, data: pd.DataFrame, train_size: float = 0.8) -> dict:
        """
        Обучение всех моделей и оценка их качества
        
        Args:
            data (pd.DataFrame): Исторические данные с колонкой 'price'
            train_size (float): Доля обучающих данных (0-1)
            
        Returns:
            dict: Словарь с результатами:
                - best_model: название лучшей модели
                - best_rmse: RMSE лучшей модели
                - all_results: словарь {модель: RMSE}
        """
        logger.info("Начало обучения моделей...")
        
        # Разделение данных
        split_idx = int(len(data) * train_size)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        test_prices = test_data['price'].values
        
        # Обучение каждой модели
        for name, model in self.models.items():
            logger.info(f"Обучение модели '{name}'...")
            try:
                # Обучение и предсказание на тестовых данных
                rmse = model.train(data, train_size=config.TRAIN_SIZE)
                
                # Сохранение результатов
                self.all_results[name] = rmse
                logger.info(f"Модель '{name}' обучена. RMSE: {rmse:.4f}")
                
            except Exception as e:
                logger.error(f"Ошибка обучения модели '{name}': {str(e)}", exc_info=True)
                self.all_results[name] = float('inf')
        
        # Выбор лучшей модели
        valid_results = {k: v for k, v in self.all_results.items() if v != float('inf')}
        
        if not valid_results:
            raise ValueError("Ни одна модель не обучена успешно. Проверьте данные и параметры моделей.")
        
        self.best_model_name = min(valid_results, key=valid_results.get)
        self.best_rmse = valid_results[self.best_model_name]
        
        logger.info(f"Лучшая модель: {self.best_model_name} с RMSE={self.best_rmse:.4f}")
        
        return {
            'best_model': self.best_model_name,
            'best_rmse': self.best_rmse,
            'all_results': self.all_results
        }
    
    def predict_best_model(self, steps: int) -> np.ndarray:
        """Прогнозирование с использованием лучшей модели"""
        if self.best_model_name is None:
            raise ValueError("Лучшая модель не определена")
        
        best_model = self.models[self.best_model_name]
        
        try:
            return best_model.predict(steps)
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {str(e)}", exc_info=True)
            
            # Попытка использовать любую обученную модель
            for name, model in self.models.items():
                if model.is_trained() and name != self.best_model_name:
                    try:
                        logger.warning(f"Используется запасная модель {name}")
                        return model.predict(steps)
                    except:
                        continue
            
            raise ValueError(f"Не удалось сгенерировать прогноз: {str(e)}")