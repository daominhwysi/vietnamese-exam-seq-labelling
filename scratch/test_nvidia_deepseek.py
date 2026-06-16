import os
import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generation.deepseek_client import chat

def main():
    print("Testing chat function with deepseek-v4-pro (NVIDIA NIM)...")
    try:
        # Simple test prompt
        result = chat(
            prompt="Hello, respond with exactly 'NVIDIA NIM deepseek-v4-pro works!'",
            model="deepseek-v4-pro",
            thinking=False
        )
        print("Response received:")
        print(repr(result))
    except Exception as e:
        print(f"Error occurred during test: {e}")

if __name__ == "__main__":
    main()
