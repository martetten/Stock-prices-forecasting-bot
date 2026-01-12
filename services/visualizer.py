"""
Сервис для создания визуализаций прогнозов
"""

import matplotlib
matplotlib.use('Agg')  # Использование backend без GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
import os
import logging
import tempfile
from config import config

logger = logging.getLogger(__name__)

class Visualizer:
    """Класс для создания графиков прогнозов"""
    
    def create_forecast_plot(
        self,
        ticker: str,
        historical_data: pd.DataFrame,
        forecast: np.ndarray,
        buy_days: list,
        sell_days: list,
        forecast_days: int
    ) -> str:
        """
        Создание графика с историческими данными и прогнозом
        
        Args:
            ticker (str): Тикер компании
            historical_data (pd.DataFrame): Исторические данные с колонкой 'price'
            forecast (np.ndarray): Прогнозируемые цены
            buy_days (list): Индексы дней для покупки
            sell_days (list): Индексы дней для продажи
            forecast_days (int): Количество дней прогноза
            
        Returns:
            str: Путь к сохранённому файлу графика
        """
        try:
            # Проверка данных
            if historical_data.empty or len(forecast) == 0:
                raise ValueError("Некорректные данные для визуализации")
            
            # Создание временных меток для прогноза
            last_date = historical_data.index[-1]
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            # Создание фигуры
            plt.figure(figsize=config.PLOT_FIGSIZE)
            
            # Построение исторических данных
            plt.plot(
                historical_data.index,
                historical_data['price'],
                label='Исторические данные',
                linewidth=2.5,
                color='#1f77b4'
            )
            
            # Построение прогноза
            plt.plot(
                forecast_dates,
                forecast,
                label=f'Прогноз на {forecast_days} дней',
                linewidth=2.5,
                linestyle='--',
                color='#ff7f0e'
            )
            
            # Добавление точек покупки
            if buy_days:
                buy_dates = [forecast_dates[i] for i in buy_days if i < len(forecast_dates)]
                buy_prices = [forecast[i] for i in buy_days if i < len(forecast)]
                
                if buy_dates:
                    plt.scatter(
                        buy_dates,
                        buy_prices,
                        color='#2ca02c',
                        s=150,
                        marker='^',
                        label='Точки покупки',
                        zorder=5,
                        edgecolors='black',
                        linewidth=1.5
                    )
            
            # Добавление точек продажи
            if sell_days:
                sell_dates = [forecast_dates[i] for i in sell_days if i < len(forecast_dates)]
                sell_prices = [forecast[i] for i in sell_days if i < len(forecast)]
                
                if sell_dates:
                    plt.scatter(
                        sell_dates,
                        sell_prices,
                        color='#d62728',
                        s=150,
                        marker='v',
                        label='Точки продажи',
                        zorder=5,
                        edgecolors='black',
                        linewidth=1.5
                    )
            
            # Оформление графика
            plt.title(f'Прогноз цены акций {ticker} на {forecast_days} дней', fontsize=16, fontweight='bold')
            plt.xlabel('Дата', fontsize=12)
            plt.ylabel('Цена, USD', fontsize=12)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Создание временного файла с уникальным именем
            temp_dir = tempfile.gettempdir()
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_{ticker}_{timestamp}.png"
            filepath = os.path.join(temp_dir, filename)
            
            # Сохранение графика во временный файл
            plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"График сохранён во временный файл: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Ошибка при создании графика: {str(e)}", exc_info=True)
            raise ValueError(f"Не удалось создать график прогноза: {str(e)}") from e