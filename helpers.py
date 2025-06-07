import os
from dotenv import load_dotenv

def get_env_var(key: str) -> str:
    load_dotenv()
    return os.getenv(key)
