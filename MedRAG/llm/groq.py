import os
import dotenv
from .base import BaseLLM
from llama_index.llms.groq import Groq

load_dotenv()

class GroqLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(
            model="llama3-70b-8192",  # or "mixtral-8x7b-32768", etc.
            api_key=self.api_key
        )

    def generate(self, query: str) -> str:
        return self.llm.complete(query).text