"""
LSTM модель для прогнозирования временных рядов
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
import logging
from models.base_model import BaseModel
from config import config

logger = logging.getLogger(__name__)

class LSTMNetwork(nn.Module):
    """Архитектура LSTM сети"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMModel(BaseModel):
    """LSTM модель для временных рядов"""
    
    def __init__(self):
        super().__init__("LSTM")
        self.look_back = config.LSTM_LOOK_BACK
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = None
    
    def prepare_sequences(self, data: np.ndarray) -> tuple:
        """Подготовка последовательностей для LSTM"""
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i-self.look_back:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """Обучение LSTM"""
        try:
            # Сохранение данных для прогноза
            self.data = data.copy()
            
            # Масштабирование
            prices = self.data['price'].values.reshape(-1, 1)
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Разделение
            split_idx = int(len(data) * train_size)
            train_scaled = scaled_prices[:split_idx]
            test_scaled = scaled_prices[split_idx-self.look_back:]
            
            # Создание последовательностей
            X_train, y_train = self.prepare_sequences(train_scaled)
            X_test, y_test = self.prepare_sequences(test_scaled)
            
            if len(X_train) == 0 or len(X_test) == 0:
                return float('inf')
            
            # Преобразование в тензоры
            X_train = torch.FloatTensor(X_train).to(self.device)
            if X_train.dim() == 2:  # Если 2D - делаем 3D
                X_train = X_train.unsqueeze(-1)  # [batch, seq_len, 1]
            y_train = torch.FloatTensor(y_train).to(self.device)
            
            X_test = torch.FloatTensor(X_test).to(self.device)
            if X_test.dim() == 2:
                X_test = X_test.unsqueeze(-1)
            y_test = torch.FloatTensor(y_test).to(self.device)
            
            # DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.LSTM_BATCH_SIZE,
                shuffle=False
            )
            
            # Модель
            self.model = LSTMNetwork(
                input_size=1,
                hidden_size=config.LSTM_HIDDEN_SIZE,
                num_layers=config.LSTM_NUM_LAYERS
            ).to(self.device)
            
            # Обучение
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()
            for epoch in range(config.LSTM_EPOCHS):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    
                    # Корректировка размерности
                    if outputs.dim() > batch_y.dim():
                        batch_y = batch_y.unsqueeze(1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Оценка
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_test).cpu().numpy()
                predictions = self.scaler.inverse_transform(predictions)
                y_test_inv = self.scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
            
            rmse = math.sqrt(mean_squared_error(y_test_inv, predictions))
            self.trained = True
            return rmse
            
        except Exception as e:
            logger.error(f"Ошибка обучения LSTM: {str(e)}", exc_info=True)
            return float('inf')
    
    def predict(self, steps: int) -> np.ndarray:
        """Прогнозирование на будущее"""
        if not self.trained or self.model is None or self.data is None:
            raise ValueError("Модель не обучена или отсутствуют данные")
        
        try:
            self.model.eval()
            predictions = []
            
            # Подготовка последних данных
            prices = self.data['price'].values.reshape(-1, 1)
            scaled_prices = self.scaler.transform(prices)
            last_sequence = scaled_prices[-self.look_back:]
            
            with torch.no_grad():
                for _ in range(steps):
                    input_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
                    if input_seq.dim() == 2:
                        input_seq = input_seq.unsqueeze(-1)
                    
                    pred_scaled = self.model(input_seq).cpu().numpy()
                    pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
                    predictions.append(pred)
                    
                    # Обновление последовательности
                    last_sequence = np.vstack([last_sequence[1:], pred_scaled])
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования LSTM: {str(e)}", exc_info=True)
            raise