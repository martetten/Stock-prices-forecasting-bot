"""
Random Forest модель для прогнозирования временных рядов
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
import logging
from models.base_model import BaseModel
from config import config

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest модель с лаговыми признаками"""
    
    def __init__(self):
        super().__init__("Random Forest")
        self.n_lags = config.RF_N_LAGS
        self.scaler = StandardScaler()
        self.feature_names = None
        self.last_data = None 
    
    def _create_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание лаговых признаков"""
        df = data.copy()
        for lag in range(1, self.n_lags + 1):
            df[f'lag_{lag}'] = df['price'].shift(lag)
        return df.dropna()
    
    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """Обучение"""
        try:
            self.last_data = data.copy()

            # Разделение данных
            split_idx = int(len(data) * train_size)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            # Создание лаговых признаков
            train_features = self._create_lagged_features(train_data)
            test_features = self._create_lagged_features(test_data)
            
            if len(train_features) < 10:
                return float('inf')
            
            # Разделение на признаки и целевую переменную
            X_train = train_features[[f'lag_{i}' for i in range(1, self.n_lags + 1)]]
            y_train = train_features['price']
            
            X_test = test_features[[f'lag_{i}' for i in range(1, self.n_lags + 1)]]
            y_test = test_features['price']
            
            # Сохраняем последнюю цену для сглаживания
            self.last_price = train_data['price'].iloc[-1]
            
            # Масштабирование
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучение
            self.model = RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Оценка
            y_pred = self.model.predict(X_test_scaled)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            self.trained = True
            return rmse
            
        except Exception as e:
            logger.error(f"Ошибка обучения Random Forest: {str(e)}", exc_info=True)
            return float('inf')
    
    def predict(self, steps: int) -> np.ndarray:
        """Прогнозирование с сглаживанием первой точки"""
        if not self.trained or self.last_price is None:
            raise ValueError("Модель не обучена")
        
        try:
            predictions = []

            # Получение последних лагов
            last_prices = self.last_data['price'].values[-self.n_lags:].copy()
            
            for i in range(steps):
                # Подготовка данных
                features = last_prices[-self.n_lags:].reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                
                # Прогноз
                next_price = self.model.predict(features_scaled)[0]
                predictions.append(next_price)
                
                # Обновление лагов
                last_prices = np.append(last_prices[1:], next_price)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {str(e)}", exc_info=True)
            raise