"""
Базовый класс для всех моделей прогнозирования
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Базовый класс с единым интерфейсом для всех моделей"""
    
    def __init__(self, name: str):
        self.name = name
        self.trained = False
    
    @abstractmethod
    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """
        Обучение модели на данных
        
        Args:
            data: Полный DataFrame с историческими данными
            train_size: Доля данных для обучения (0-1)
            
        Returns:
            RMSE на тестовых данных
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """
        Прогнозирование на будущее
        
        Args:
            steps: Количество дней для прогноза
            
        Returns:
            Массив прогнозируемых цен
        """
        pass
    
    def get_name(self) -> str:
        """
        Получение названия модели
        
        Returns:
            str: Название модели
        """
        return self.name
    
    def is_trained(self) -> bool:
        """
        Проверка, обучена ли модель
        
        Returns:
            bool: True если модель обучена, иначе False
        """
        return self.trained