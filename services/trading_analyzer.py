"""
Сервис для анализа торговых стратегий и расчёта прибыли
"""

import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple
import logging
from config import config

logger = logging.getLogger(__name__)

class TradingAnalyzer:
    """Класс для анализа торговых сигналов и формирования рекомендаций"""
    
    def __init__(self):
        self.extrema_order = max(1, min(3, config.FORECAST_DAYS // 10))
    
    def find_optimal_trades(self, predictions: np.ndarray) -> Tuple[List[int], List[int], bool]:
        """Поиск оптимальных точек для покупки и продажи"""
        if len(predictions) < 3:
            return [0], [len(predictions) - 1], predictions[-1] > predictions[0]
        
        try:
            # Поиск экстремумов
            local_min = argrelextrema(predictions, np.less, order=self.extrema_order)[0]
            local_max = argrelextrema(predictions, np.greater, order=self.extrema_order)[0]
            
            # Определение направления стратегии
            overall_trend = predictions[-1] - predictions[0]
            is_long_strategy = overall_trend > 0
            
            return local_min.tolist(), local_max.tolist(), is_long_strategy
            
        except Exception as e:
            logger.warning(f"Ошибка при поиске экстремумов: {str(e)}")
            # Fallback стратегия
            return [0], [len(predictions) - 1], predictions[-1] > predictions[0]
    
    def generate_recommendations(
        self, 
        forecast: np.ndarray,
        amount: float,
        current_price: float,
        buy_days: List[int],
        sell_days: List[int],
        is_long_strategy: bool
    ) -> tuple:
        """
        Формирование торговых рекомендаций с фильтрацией незначительных сделок
        
        Returns:
            tuple: (profit, strategy_text, roi)
        """
        strategy_details = []
        total_profit = 0.0
        valid_trades_found = False
        
        # Расчет для лонг-стратегии
        if is_long_strategy:
            for buy_idx in buy_days:
                sell_candidates = [d for d in sell_days if d > buy_idx]
                if sell_candidates:
                    sell_idx = min(sell_candidates)
                    buy_price = forecast[buy_idx]
                    sell_price = forecast[sell_idx]
                    
                    # Расчет прибыли для сделки
                    shares = amount / buy_price if amount > 0 else 1
                    trade_profit = shares * (sell_price - buy_price)
                    
                    # Фильтрация по порогу прибыли
                    if abs(trade_profit) >= config.MIN_PROFIT_THRESHOLD:
                        valid_trades_found = True
                        total_profit += trade_profit
                        
                        strategy_details.append(
                            f"🟢 <b>День {buy_idx+1}:</b> Покупка по ${buy_price:.2f}\n"
                            f"🔴 <b>День {sell_idx+1}:</b> Продажа по ${sell_price:.2f}\n"
                            f"💸 Потенциальная прибыль: ${trade_profit:.2f}"
                        )
        
        # Расчет для short-стратегии
        else:
            for sell_idx in sell_days:
                buy_candidates = [d for d in buy_days if d > sell_idx]
                if buy_candidates:
                    buy_idx = min(buy_candidates)
                    sell_price = forecast[sell_idx]
                    buy_price = forecast[buy_idx]
                    
                    # Расчет прибыли для short-сделки
                    shares = amount / sell_price if amount > 0 else 1
                    trade_profit = shares * (sell_price - buy_price)
                    
                    # Фильтрация по порогу прибыли
                    if abs(trade_profit) >= config.MIN_PROFIT_THRESHOLD:
                        valid_trades_found = True
                        total_profit += trade_profit
                        
                        strategy_details.append(
                            f"🔴 <b>День {sell_idx+1}:</b> Открыть SHORT по ${sell_price:.2f}\n"
                            f"🟢 <b>День {buy_idx+1}:</b> Закрыть SHORT по ${buy_price:.2f}\n"
                            f"💸 Потенциальная прибыль: ${trade_profit:.2f}"
                        )
        
        # Если нет значимых сделок - используем стратегию на весь период
        if not valid_trades_found:
            total_change = forecast[-1] - forecast[0]
            overall_profit = 0.0
            
            if is_long_strategy:
                shares = amount / forecast[0] if amount > 0 else 1
                overall_profit = shares * total_change
            else:
                shares = amount / forecast[0] if amount > 0 else 1
                overall_profit = shares * (-total_change)
            
            if abs(overall_profit) >= config.MIN_PROFIT_THRESHOLD / 2:  # Более низкий порог для общей стратегии
                total_profit = overall_profit
                strategy_type = "ПОКУПКА → ПРОДАЖА" if is_long_strategy else "SHORT → ЗАКРЫТИЕ"
                
                strategy_details.append(
                    f"🧞‍♂️ <b>Общая стратегия на весь период:</b>\n"
                    f"   • {strategy_type}\n"
                    f"   • Цена открытия: ${forecast[0]:.2f}\n"
                    f"   • Цена закрытия: ${forecast[-1]:.2f}\n"
                    f"   • Ожидаемая прибыль: ${total_profit:.2f}"
                )
            else:
                strategy_details.append(
                    f"🔸 <b>Недостаточно выгодных сделок:</b>\n"
                    f"   • Минимальный порог прибыли: ${config.MIN_PROFIT_THRESHOLD:.2f}\n"
                    f"   • Рекомендуется рассмотреть другие акции"
                )
        
        # Расчет ROI
        roi = (total_profit / amount) * 100 if amount > 0 else 0
        
        # Формирование текста стратегии
        if not strategy_details:
            strategy_text = "⚠️ Не удалось сформировать торговую стратегию"
        else:
            strategy_text = "\n\n".join(strategy_details)
        
        return total_profit, strategy_text, roi