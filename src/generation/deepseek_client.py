import os
from typing import Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

from src.token_tracker import log_response

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
    """
    Call the DeepSeek chat API and return the assistant reply text.

    Token usage (prompt, completion, reasoning) and both *reasoning_content*
    and *content* are automatically appended to today's log file via
    ``src.token_tracker.log_response``.
    """
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

    # ── log token usage + content ─────────────────────────────────────────
    log_response(response, model=model)

    return response.choices[0].message.content


if __name__ == "__main__":
    result = chat("Hello")
    print(result)
