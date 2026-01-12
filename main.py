"""
Точка входа для запуска Telegram-бота
"""

import asyncio
import logging
import os
import sys
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from bot.handlers import register_handlers
from utils.logger import setup_logging
from config import config

# Настройка системного логирования
setup_logging()
logger = logging.getLogger(__name__)

async def on_startup(bot: Bot):
    """Действия при запуске бота"""
    # Получение информации о боте
    bot_info = await bot.get_me()
    
    logger.info("=" * 60)
    logger.info(f"Запуск бота: @{bot_info.username}")
    logger.info(f"ID бота: {bot_info.id}")
    logger.info(f"Прогноз на {config.FORECAST_DAYS} дней")
    logger.info(f"Исторические данные: {config.HISTORY_DAYS} дней")
    logger.info(f"Лог-файл: {config.LOG_FILE}")
    logger.info("=" * 60)
    
    # Установка команд в интерфейсе Telegram
    await bot.set_my_commands([
        {"command": "start", "description": "Начать анализ акций"},
        {"command": "help", "description": "Помощь по использованию"},
        {"command": "cancel", "description": "Отменить текущую операцию"}
    ])
    
    # Установка описания бота
    await bot.set_my_description(
        "🎓 Учебный бот для прогнозирования акций\n"
        "🧙‍♂️ Использует 3 модели ML: Random Forest, ARIMA, LSTM\n"
        "⚠️ Результаты носят исключительно учебный характер"
    )

async def on_shutdown(bot: Bot):
    """Действия при остановке бота"""
    logger.info("=" * 60)
    logger.info("Бот остановлен")
    logger.info(f"Статистика: {getattr(bot, 'requests_count', 0)} обработанных запросов")
    logger.info("=" * 60)

async def main():
    """Основная функция запуска бота"""
    # Инициализация бота с настройками
    bot = Bot(
        token=config.TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    
    # Регистрация обработчиков
    register_handlers(dp)
    
    # Настройка обработчиков событий
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    # Инициализация счётчика запросов
    bot.requests_count = 0
    
    try:
        logger.info("Запуск polling...")
        await dp.start_polling(
            bot, 
            allowed_updates=dp.resolve_used_update_types(),
            close_bot_session=True
        )
    except KeyboardInterrupt:
        logger.info("Остановка по запросу пользователя")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.critical(f"Необработанная ошибка: {e}", exc_info=True)
        sys.exit(1)