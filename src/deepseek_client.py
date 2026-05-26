import os
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
) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    if "pro" in model or "reasoner" in model:
        kwargs["reasoning_effort"] = "high"
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


if __name__ == "__main__":
    result = chat("Hello")
    print(result)
