import os
from typing import Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def chat(
    prompt: str,
    system: str = "You are a helpful assistant",
    model: str = "deepseek-v4-flash",
    thinking: Optional[Any] = None,
) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    
    # Resolve reasoning effort
    effort = None
    if thinking is True:
        effort = "high"
    elif isinstance(thinking, str) and thinking in ["high", "max"]:
        effort = thinking
    elif thinking is None:
        if "pro" in model or "reasoner" in model:
            effort = "high"

    if effort is not None:
        kwargs["reasoning_effort"] = effort

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


if __name__ == "__main__":
    result = chat("Hello")
    print(result)
