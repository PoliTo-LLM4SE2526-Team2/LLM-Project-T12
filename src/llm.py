from openai import OpenAI
from abc import ABC, abstractmethod

class baseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list, temperature: float = 0.1) -> str:
        pass

class chatLLM(baseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages, temperature = 0.1) -> str:
        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return ""