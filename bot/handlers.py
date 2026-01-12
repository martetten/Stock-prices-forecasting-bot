# bot/handlers.py
"""
Обработчики команд Telegram-бота для прогнозирования акций
"""

from aiogram import Router, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime

from config import config
from services.data_loader import load_stock_data, validate_ticker
from services.model_selector import ModelSelector
from services.trading_analyzer import TradingAnalyzer
from services.visualizer import Visualizer
from utils.logger import log_user_request

logger = logging.getLogger(__name__)
router = Router()

# Определение состояний FSM
class StockAnalysis(StatesGroup):
    waiting_for_ticker = State()
    waiting_for_amount = State()

@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Начало диалога с пользователем"""
    # Сброс предыдущего состояния
    await state.clear()
    
    welcome_text = (
        "👋 <b>Привет, я бот-предсказатель 🔮 стоимости акций!</b>\n\n"
        "Я проанализирую исторические данные и построю прогноз на "
        f"<b>{config.FORECAST_DAYS}</b> дней с использованием трёх моделей:\n"
        "• 🌳 Random Forest\n"
        "• 📊 ARIMA\n"
        "• 🧠 LSTM\n\n"
        "Введите тикер компании (например: AAPL, MSFT, TSLA):"
    )
    await message.answer(welcome_text, parse_mode="HTML")
    await state.set_state(StockAnalysis.waiting_for_ticker)

@router.message(StateFilter(StockAnalysis.waiting_for_ticker), F.text)
async def process_ticker(message: Message, state: FSMContext):
    """Обработка введённого тикера"""
    ticker = message.text.strip().upper()
    
    # Валидация тикера
    if not validate_ticker(ticker):
        await message.answer(
            "❌ Некорректный тикер. Пожалуйста, используйте валидный формат:\n"
            "• Только буквы и цифры\n"
            "• Длина не более 10 символов\n\n"
            "Примеры: AAPL, MSFT, TSLA, GOOGL"
        )
        return
    
    # Сохранение тикера в состоянии
    await state.update_data(ticker=ticker)
    
    next_text = (
        f"Выбран тикер: <b>{ticker}</b>\n\n"
        "Введите сумму для условной инвестиции в долларах:\n"
        f"{config.INVESTMENT_THRESHOLD}"
    )
    await message.answer(next_text, parse_mode="HTML")
    await state.set_state(StockAnalysis.waiting_for_amount)

@router.message(StateFilter(StockAnalysis.waiting_for_amount), F.text)
async def process_amount(message: Message, state: FSMContext):
    """Обработка суммы и запуск анализа"""
    try:
        # Очистка ввода от лишних символов
        amount_str = message.text.strip().replace(',', '').replace(' ', '').replace('$', '')
        amount = float(amount_str)
        
        # Валидация суммы
        if amount < config.MIN_INVESTMENT:
            await message.answer(
                f"❌ Сумма должна быть не меньше ${config.MIN_INVESTMENT:,}. "
                "Пожалуйста, введите корректную сумму:"
            )
            return
        
        if amount > config.MAX_INVESTMENT:
            await message.answer(
                f"❌ Сумма не может превышать ${config.MAX_INVESTMENT:,}. "
                "Пожалуйста, введите корректную сумму:"
            )
            return
        
        # Получение тикера из состояния
        user_data = await state.get_data()
        ticker = user_data['ticker']
        
        # Уведомление о начале анализа
        await message.answer(
            f"🗿 <b>Начинаю анализ для {ticker}</b>\n\n"
            "⏱ Это займёт 1-2 минуты. Пожалуйста, подождите...",
            parse_mode="HTML"
        )
        
        # 1. Загрузка данных
        df = load_stock_data(ticker, days=config.HISTORY_DAYS)
        
        if df is None or df.empty:
            await message.answer(
                f"❌ <b>Ошибка загрузки данных</b>\n\n"
                f"Не удалось загрузить данные для тикера <b>{ticker}</b>.\n"
                "Возможные причины:\n"
                "• Неверный тикер\n"
                "• Проблемы с подключением к Yahoo Finance\n"
                "• Тикер не торгуется на бирже\n\n"
                "Используйте /start для новой попытки.",
                parse_mode='HTML'
            )
            await state.clear()
            return
        
        current_price = df['price'].iloc[-1]
        
        # 2. Обучение моделей и прогнозирование
        model_selector = ModelSelector()
        results = model_selector.train_and_evaluate(df, config.TRAIN_SIZE)
        forecast = model_selector.predict_best_model(config.FORECAST_DAYS)
        
        # Проверка корректности прогноза
        if forecast is None or len(forecast) == 0:
            raise ValueError("Не удалось сгенерировать прогноз")
        
        # 3. Анализ торговых стратегий
        analyzer = TradingAnalyzer()
        buy_days, sell_days, is_long_strategy = analyzer.find_optimal_trades(forecast)
        profit, strategy_text, roi = analyzer.generate_recommendations(
            forecast, 
            amount, 
            current_price,
            buy_days, 
            sell_days,
            is_long_strategy
        )
        
        # 4. Создание визуализации
        visualizer = Visualizer()
        plot_path = visualizer.create_forecast_plot(
            ticker,
            df,
            forecast,
            buy_days,
            sell_days,
            config.FORECAST_DAYS
        )
        
        # 5. Формирование отчёта
        predicted_price = forecast[-1]
        price_diff = predicted_price - current_price
        price_change = (price_diff / current_price) * 100

        # Определение направления стратегии
        is_long_strategy = price_diff > 0
        strategy_type = "Long" if is_long_strategy else "Short"

        report = (
            f"🧙‍♂️ <b>ОТЧЁТ ПО АКЦИЯМ {ticker}</b>\n"
            f"{'='*40}\n\n"
            f"🤖 <b>Результаты моделей (RMSE):</b>\n"
        )
        
        # Добавление результатов всех моделей
        for model_name, rmse in results['all_results'].items():
            if rmse == float('inf'):
                report += f"   • {model_name}: ❌ Ошибка обучения\n"
            else:
                best_mark = " ⭐" if model_name == results['best_model'] else ""
                report += f"   • {model_name}: {rmse:.2f}{best_mark}\n"
        
        report += (
            f"\n🎯 <b>Лучшая модель:</b> {results['best_model']}\n\n"
            f"{'='*40}\n"
            f"💵 <b>АНАЛИЗ ЦЕН:</b>\n"
            f"   • Текущая цена: <b>${current_price:.2f}</b>\n"
            f"   • Прогноз через {config.FORECAST_DAYS} дней: <b>${predicted_price:.2f}</b>\n"
            f"   • Изменение: <b>{price_change:.2f}%</b>\n\n"
            f"   • Рекомендуемая стратегия: <b>{strategy_type}</b>\n"
            f"   • Ожидаемый ROI: <b>{roi:+.2f}%</b>\n\n"
        )
        
        report += f"{'='*40}\n📍 <b>ТОРГОВЫЕ РЕКОМЕНДАЦИИ:</b>\n\n"
        
        if strategy_text:
            report += strategy_text
        else:
            report += "⚠️ Недостаточно четких сигналов для торговли"
        
        report += (
            f"\n\n{'='*40}\n"
            "⚠️ Прогноз создан для образовательных целей. "
            "Не является финансовой рекомендацией.\n\n"
            "Используйте /start для нового анализа."
        )
        
        # 6. Отправка результатов
        try:
            if os.path.exists(plot_path):
                # Сначала отправляем график БЕЗ подписи
                photo = FSInputFile(plot_path)
                await message.answer_photo(photo=photo)
                
                # Затем отправляем отчет как отдельное сообщение
                await message.answer(report, parse_mode="HTML")
            else:
                await message.answer(
                    "❌ <b>Ошибка генерации графика</b>\n\n"
                    "Не удалось создать график прогноза.",
                    parse_mode="HTML"
                )
                await message.answer(report, parse_mode="HTML")
        finally:
            # Гарантированное удаление временного файла после отправки
            try:
                if os.path.exists(plot_path):
                    os.remove(plot_path)
                    logger.info(f"Временный файл удалён: {plot_path}")
            except Exception as e:
                logger.error(f"Ошибка удаления временного файла {plot_path}: {str(e)}")
        
        # 7. Логирование запроса
        log_user_request(
            user_id=message.from_user.id,
            ticker=ticker,
            amount=amount,
            best_model=results['best_model'],
            metric=results['best_rmse'],
            profit=profit
        )
        
        # 8. Очистка временных файлов
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        await state.clear()
        
        logger.info(
            f"Успешный анализ: user={message.from_user.id}, "
            f"ticker={ticker}, amount=${amount:.2f}, profit=${profit:.2f}"
        )
        
    except ValueError as ve:
        await state.clear()
        logger.error(f"Ошибка валидации: {str(ve)}")
        await message.answer(
            f"❌ <b>Ошибка ввода</b>\n\n"
            f"{str(ve)}\n\n"
            "Используйте /start для новой попытки.",
            parse_mode="HTML"
        )
        
    except Exception as e:
        await state.clear()
        logger.error(f"Критическая ошибка при анализе: {str(e)}", exc_info=True)
        
        # Классификация ошибок для пользовательского сообщения
        error_lower = str(e).lower()
        if "yfinance" in error_lower or "download" in error_lower:
            error_text = (
                "❌ <b>Ошибка загрузки данных</b>\n\n"
                "Не удалось получить данные с биржи. Возможные причины:\n"
                "• Некорректный тикер\n"
                "• Сервер Yahoo Finance временно недоступен\n"
                "• Проблемы с интернет-соединением\n\n"
                "Попробуйте позже или используйте другой тикер."
            )
        elif "model" in error_lower or "train" in error_lower:
            error_text = (
                "❌ <b>Ошибка обучения моделей</b>\n\n"
                "Не удалось обучить модели прогнозирования. Возможные причины:\n"
                "• Недостаточно исторических данных\n"
                "• Проблемы с форматом данных\n"
                "• Ошибки в реализации моделей\n\n"
                "Попробуйте использовать популярные тикеры (AAPL, MSFT)."
            )
        else:
            error_text = (
                "❌ <b>Непредвиденная ошибка</b>\n\n"
                f"Детали: {str(e)[:150]}\n\n"
                "Пожалуйста, сообщите об ошибке разработчику.\n"
                "Используйте /start для новой попытки."
            )
        
        await message.answer(error_text, parse_mode="HTML")

@router.message(Command("help"))
async def cmd_help(message: Message):
    """Команда помощи"""
    help_text = (
        "📖 <b>ПОМОЩЬ</b>\n\n"
        "<b>Доступные команды:</b>\n"
        "/start - Начать анализ акций\n"
        "/help - Показать эту справку\n"
        "/cancel - Отменить текущую операцию\n\n"
        "<b>Как использовать бота:</b>\n"
        "1️⃣ Отправьте /start\n"
        "2️⃣ Введите тикер компании (например, AAPL)\n"
        "3️⃣ Введите сумму для условной инвестиции (например, 10000)\n"
        "4️⃣ Получите прогноз и рекомендации!\n\n"
        "<b>Популярные тикеры:</b>\n"
        "• AAPL - Apple\n"
        "• MSFT - Microsoft\n"
        "• GOOGL - Google\n"
        "• TSLA - Tesla\n"
        "• AMZN - Amazon\n"
        "• NVDA - NVIDIA\n\n"
        "<b>О моделях:</b>\n"
        "Бот использует три модели машинного обучения:\n"
        "🌳 Random Forest - ансамбль деревьев решений\n"
        "📊 ARIMA - статистическая модель временных рядов\n"
        "🧠 LSTM - рекуррентная нейронная сеть\n\n"
        "Лучшая модель выбирается автоматически по метрике RMSE."
    )
    await message.answer(help_text, parse_mode="HTML")

@router.message(Command("cancel"))
@router.message(F.text.lower().in_({"отмена", "cancel"}))
async def cmd_cancel(message: Message, state: FSMContext):
    """Отмена текущей операции"""
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Нет активной операции для отмены.")
        return
    
    await state.clear()
    await message.answer(
        "❌ Операция отменена.\n\n"
        "Используйте /start для начала нового анализа."
    )

# Регистрация обработчиков
def register_handlers(dp):
    dp.include_router(router)