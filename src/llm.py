from openai import OpenAI
from abc import ABC, abstractmethod
import time

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list, temperature: float, top_p: float) -> str:
        pass

class ChatLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str, max_retries: int = 3):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries

    def generate(self, messages, temperature=0, top_p=1, timeout=60) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout
                )
                content = response.choices[0].message.content
                
                # 检查是否为空响应
                if content and content.strip():
                    return content
                else:
                    print(f"Empty response from API, retry {attempt + 1}/{self.max_retries}...")
                    if attempt < self.max_retries - 1:  # 不是最后一次才等待
                        time.sleep(1)
                    continue  # 继续下一次重试
                    
            except Exception as e:
                print(f"API Error: {e}, retry {attempt + 1}/{self.max_retries}...")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # 出错等待2秒再重试
        
        # 所有重试都失败了
        print(f"All {self.max_retries} retries failed, returning empty string")
        return ""