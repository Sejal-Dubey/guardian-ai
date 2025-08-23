# shared/llm.py
from typing import List
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from shared.config import get_settings

settings = get_settings()
client = Groq(api_key=settings.groq_api_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=10))
def chat_completion(
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content