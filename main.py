import json
import os
import random
import signal
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Dict, Optional, TypedDict, cast, Any, Literal, Tuple, NoReturn
from contextlib import suppress

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

# Load environment variables
load_dotenv()

# Configuration
class Config:
    BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")
    LOG_FILE: str = "bot.log"
    MAX_LOG_SIZE: int = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT: int = 3
    RATE_LIMIT_CALLS: int = 1
    RATE_LIMIT_PERIOD: int = 300  # 5 minutes

# Set up logging
def setup_logging() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        Config.LOG_FILE, 
        maxBytes=Config.MAX_LOG_SIZE, 
        backupCount=Config.BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

# Type definitions
class MealInfo(TypedDict):
    time: str
    details: str
    instructions: str

class MealPlan(TypedDict):
    name: str
    meals: Dict[str, MealInfo]

# Custom exceptions
class MealPlanError(Exception):
    """Base exception for meal plan related errors."""

class MealPlanNotFoundError(MealPlanError):
    """Raised when a meal plan is not found."""

class InvalidMealTimeError(MealPlanError):
    """Raised when a meal time is invalid."""

class MealNotFoundError(MealPlanError):
    """Raised when a meal is not found in a plan."""

class InvalidJobDataError(Exception):
    """Raised when job data is invalid."""

class ChatNotFoundError(Exception):
    """Raised when chat is not found in update."""

class UserNotFoundError(Exception):
    """Raised when user is not found in update."""

# Meal plan loader
class MealPlanLoader:
    @staticmethod
    def load_meal_plans() -> Dict[str, MealPlan]:
        meal_plans: Dict[str, MealPlan] = {}
        for char in range(ord('a'), ord('z') + 1):
            file_name = f"plan_{chr(char)}.json"
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'r') as file:
                        data = json.load(file)
                        if not isinstance(data, dict) or 'name' not in data or 'meals' not in data:
                            raise ValueError(f"Invalid meal plan structure in {file_name}")
                        meal_plans[chr(char)] = cast(MealPlan, data)
                except (json.JSONDecodeError, IOError, ValueError) as e:
                    logger.error(f"Error reading {file_name}: {e}")
        return meal_plans

# Bot handlers
class MealPlanBot:
    def __init__(self):
        self.meal_plans: Dict[str, MealPlan] = MealPlanLoader.load_meal_plans()

    async def send_meal_message(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        job = context.job
        if job is None or not isinstance(job.data, dict):
            logger.error("Invalid job data")
            raise InvalidJobDataError("Job data is invalid or None")

        job_data: Dict[str, Any] = job.data
        chat_id = job_data.get('chat_id')
        meal = job_data.get('meal')
        meal_plan = job_data.get('meal_plan')

        if not all(isinstance(i, str) for i in [meal, meal_plan] if i is not None) or not isinstance(chat_id, (int, str)):
            logger.error(f"Invalid job data: chat_id={chat_id}, meal={meal}, meal_plan={meal_plan}")
            raise InvalidJobDataError(f"Invalid job data types: chat_id={type(chat_id)}, meal={type(meal)}, meal_plan={type(meal_plan)}")

        try:
            plan = self.get_meal_plan(cast(str, meal_plan))
            meal_info = self.get_meal_info(plan, cast(str, meal))
            message = self.format_meal_message(cast(str, meal), meal_info)
            await context.bot.send_message(chat_id=chat_id, text=message)
        except MealPlanError as e:
            logger.error(f"Error processing meal plan: {e}")
            await context.bot.send_message(chat_id=chat_id, text="An error occurred while processing your meal plan. Please try again later.")
        except Exception as e:
            logger.error(f"Unexpected error sending meal message: {e}")
            await context.bot.send_message(chat_id=chat_id, text="An unexpected error occurred. Please try again later.")

    def get_meal_plan(self, meal_plan_key: str) -> MealPlan:
        plan = self.meal_plans.get(meal_plan_key)
        if not plan:
            raise MealPlanNotFoundError(f"Meal plan {meal_plan_key} not found")
        return plan

    def get_meal_info(self, plan: MealPlan, meal: str) -> MealInfo:
        meal_info = plan['meals'].get(meal)
        if not meal_info:
            raise MealNotFoundError(f"Meal {meal} not found in plan")
        return meal_info

    def format_meal_message(self, meal: str, meal_info: MealInfo) -> str:
        return f"{meal} ({meal_info.get('time', 'Time not specified')}):\n{meal_info.get('details', 'No details available')}\nInstructions: {meal_info.get('instructions', 'No instructions available')}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat:
            logger.error("No effective chat in update")
            raise ChatNotFoundError("No effective chat found in update")

        chat_id = chat.id

        try:
            await context.bot.send_message(
                chat_id=chat_id, 
                text="Welcome to your personalized meal plan! You will receive meal instructions at the appropriate times."
            )

            if not self.meal_plans:
                await context.bot.send_message(chat_id=chat_id, text="Sorry, no meal plans are available at the moment.")
                return

            meal_plan_key = random.choice(list(self.meal_plans.keys()))
            meal_plan = self.meal_plans[meal_plan_key]
            plan_name = meal_plan['name']
            await context.bot.send_message(chat_id=chat_id, text=f"You've been assigned meal plan: {plan_name}")

            self.schedule_meals(context, chat_id, meal_plan_key, meal_plan)

            # Display a random meal close to the current time
            current_time = datetime.now()
            closest_meal = self.get_closest_meal(meal_plan, current_time)
            if closest_meal:
                meal_name, meal_info = closest_meal
                message = self.format_meal_message(meal_name, meal_info)
                await context.bot.send_message(chat_id=chat_id, text=f"Here's a meal close to your current time:\n\n{message}")
            else:
                await context.bot.send_message(chat_id=chat_id, text="No meals found for the current time.")

        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await context.bot.send_message(chat_id=chat_id, text="An error occurred while setting up your meal plan. Please try again later.")

    def schedule_meals(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, meal_plan_key: str, meal_plan: MealPlan) -> None:
        job_queue = context.job_queue
        if job_queue is None:
            logger.error("Job queue is None")
            raise ValueError("Job queue is None")

        for meal, details in meal_plan['meals'].items():
            try:
                meal_time = self.parse_meal_time(details['time'])
                meal_datetime = self.get_next_meal_datetime(meal_time)

                job_queue.run_once(
                    self.send_meal_message,
                    meal_datetime,
                    data={'chat_id': chat_id, 'meal': meal, 'meal_plan': meal_plan_key},
                    name=f"{chat_id}_{meal}"
                )
            except InvalidMealTimeError as e:
                logger.error(f"Invalid time format for meal '{meal}' in plan '{meal_plan_key}': {e}")

    @staticmethod
    def parse_meal_time(time_str: str) -> datetime:
        try:
            return datetime.strptime(time_str, '%H:%M')
        except ValueError as e:
            raise InvalidMealTimeError(f"Invalid time format: {time_str}") from e

    @staticmethod
    def get_next_meal_datetime(meal_time: datetime) -> datetime:
        now = datetime.now()
        meal_datetime = now.replace(hour=meal_time.hour, minute=meal_time.minute, second=0, microsecond=0)
        if meal_datetime < now:
            meal_datetime += timedelta(days=1)
        return meal_datetime

    def get_closest_meal(self, meal_plan: MealPlan, current_time: datetime) -> Optional[Tuple[str, MealInfo]]:
        closest_meal: Optional[Tuple[str, MealInfo]] = None
        smallest_time_diff = timedelta.max

        for meal_name, meal_info in meal_plan['meals'].items():
            try:
                meal_time = self.parse_meal_time(meal_info['time'])
                time_diff = abs(current_time.replace(year=1, month=1, day=1) - meal_time.replace(year=1, month=1, day=1))
                if time_diff < smallest_time_diff:
                    smallest_time_diff = time_diff
                    closest_meal = (meal_name, meal_info)
            except InvalidMealTimeError:
                continue

        return closest_meal

    @limits(calls=Config.RATE_LIMIT_CALLS, period=Config.RATE_LIMIT_PERIOD)
    @on_exception(expo, RateLimitException, max_tries=1)
    async def _rate_limited_random_meal(self, chat_id: int) -> str:
        if not self.meal_plans:
            return "Sorry, no meal plans are available at the moment."

        meal_plan_key = random.choice(list(self.meal_plans.keys()))
        meal_plan = self.meal_plans[meal_plan_key]
        meal_name, meal_info = random.choice(list(meal_plan['meals'].items()))

        message = self.format_meal_message(meal_name, meal_info)
        return f"Here's a random meal from plan {meal_plan['name']}:\n\n{message}"

    async def random_meal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat:
            logger.error("No effective chat in update")
            raise ChatNotFoundError("No effective chat found in update")

        chat_id = chat.id
        user = update.effective_user
        if not user:
            logger.error("No effective user in update")
            raise UserNotFoundError("No effective user found in update")

        user_id = user.id

        try:
            message = await self._rate_limited_random_meal(chat_id)
            await context.bot.send_message(chat_id=chat_id, text=message)
        except RateLimitException:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"You're using this command too frequently. Please wait {Config.RATE_LIMIT_PERIOD} seconds before trying again."
            )
        except Exception as e:
            logger.error(f"Error in random_meal command: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="An error occurred while processing your request. Please try again later."
            )

async def main() -> None:
    bot = MealPlanBot()
    application = Application.builder().token(Config.BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("random_meal", bot.random_meal))

    try:
        await application.initialize()
        await application.start()
        logger.info("Bot started. Press Ctrl+C to stop.")

        # Set up signal handlers
        stop_event = asyncio.Event()

        def signal_handler() -> None:
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        # Run the bot until the stop event is set
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await stop_event.wait()
    except Exception as e:
        logger.error(f"Error running the bot: {e}")
    finally:
        with suppress(Exception):
            await application.stop()
        logger.info("Bot stopped.")

def run_bot() -> NoReturn:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Ensure the script exits
        os._exit(0)

if __name__ == '__main__':
    run_bot()

# Print python-telegram-bot version
import telegram
print(f"python-telegram-bot version: {telegram.__version__}")