import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GIGA_CHAT_CREDENTIALS: str = os.getenv("GIGA_CHAT_CREDENTIALS")
    GIGA_CHAT_MODEL: str = os.getenv("GIGA_CHAT_MODEL")
