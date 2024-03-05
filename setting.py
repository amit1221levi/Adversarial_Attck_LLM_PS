import os
from dotenv import load_dotenv

load_dotenv()

class Setting:
    def __init__(self):
        self.huggingface_key = os.getenv("HUGGINGFACE_API_KEY")