import os
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.1-8b-instant"


def groq_client() -> AsyncGroq:
    return AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
