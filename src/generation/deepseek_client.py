import os
from typing import Optional, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.token_tracker import log_response

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

# ── setup clients ─────────────────────────────────────────────────────────────
deepseek_client = None
if os.environ.get("DEEPSEEK_API_KEY"):
    deepseek_client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

nvidia_client = None
if os.environ.get("NVIDIA_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"):
    nvidia_client = OpenAI(
        api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
    )

vilao_client = None
if os.environ.get("LLM_API_KEY"):
    vilao_client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY"),
        base_url="https://api.vilao.ai/v1",
    )

# Alias client for test mocking compatibility
client = deepseek_client


def chat(
    prompt: str,
    system: str = "You are a helpful assistant",
    model: str = "deepseek-v4-flash",
    thinking: Optional[Any] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Call the DeepSeek (or NVIDIA NIM DeepSeek, or Vilao.ai) chat API and return the assistant reply text.

    Token usage (prompt, completion, reasoning) and both *reasoning_content*
    and *content* are automatically appended to today's log file via
    ``src.token_tracker.log_response``.
    """
    # Decide which client to route to
    is_nvidia = False
    use_vilao = False
    
    if provider == "nvidia":
        is_nvidia = True
    elif provider == "vilao":
        use_vilao = True
    elif provider == "deepseek":
        pass
    else:
        # Auto-detect routing
        is_nvidia = model in ["deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"]
        
        # Check if we should route to Vilao.ai
        if "/" in model:
            use_vilao = True
        elif "minimax" in model.lower() or model.startswith("mn/"):
            use_vilao = True
        elif not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("LLM_API_KEY"):
            use_vilao = True

    kwargs = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    
    if is_nvidia:
        if nvidia_client is None:
            raise ValueError("Error: Provider requires NVIDIA_API_KEY or DEEPSEEK_API_KEY but neither is set.")
        active_client = nvidia_client
        target_model = model
        if model in ["deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"]:
            target_model = "deepseek-ai/deepseek-v4-pro"
        kwargs["model"] = target_model
        
        # Resolve thinking for NVIDIA provider
        thinking_bool = False
        if thinking is True or (isinstance(thinking, str) and thinking in ["high", "max"]):
            thinking_bool = True
        elif thinking is None:
            # Default thinking to True for the Pro/Reasoner model
            thinking_bool = True
            
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": thinking_bool}}
        
    elif use_vilao:
        if vilao_client is None:
            if deepseek_client is not None:
                # Fallback to direct DeepSeek API
                active_client = deepseek_client
                kwargs["model"] = model
            else:
                raise ValueError("Error: Model routes to Vilao.ai but LLM_API_KEY is not set, and DEEPSEEK_API_KEY is also not set.")
        else:
            active_client = vilao_client
            # Format the model name with provider prefix if missing
            final_model = model
            if "/" not in final_model:
                if "minimax" in final_model.lower():
                    final_model = f"mn/{final_model}"
                elif "deepseek" in final_model.lower():
                    final_model = f"deepseek/{final_model}"
            kwargs["model"] = final_model
            
    else:
        if deepseek_client is None:
            raise ValueError("Error: DEEPSEEK_API_KEY is not set.")
        active_client = deepseek_client
        kwargs["model"] = model
        
        # Resolve reasoning effort for DeepSeek direct API
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

    response = active_client.chat.completions.create(**kwargs)

    # ── log token usage + content ─────────────────────────────────────────
    # Log usage under the requested/original model name for consistency in tracker
    log_response(response, model=model)

    return response.choices[0].message.content


if __name__ == "__main__":
    result = chat("Hello")
    print(result)
