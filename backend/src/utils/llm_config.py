"""
Universal LLM Configuration
Uses Groq (Llama 3.3 70B) for LLM inference

Usage:
    from src.utils.llm_config import llm, invoke_with_retry
    response = llm.invoke("your prompt")

    # Or with structured output
    structured_llm = llm.with_structured_output(YourPydanticModel)
    result = structured_llm.invoke("your prompt")
"""

import os
import time
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ==============================================================================
# Global LLM Instance
# ==============================================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # switch to llama-3.1-8b-instant for faster dev
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ==============================================================================
# Retry Utility (replaces all the time.sleep rate limiting)
# ==============================================================================

def invoke_with_retry(messages, max_retries: int = 3, structured_output_class=None):
    """
    Invoke LLM with exponential backoff on rate limit errors.
    Use this instead of llm.invoke() everywhere in the codebase.
    """
    active_llm = llm.with_structured_output(structured_output_class) if structured_output_class else llm
    for attempt in range(max_retries):
        try:
            return active_llm.invoke(messages)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s → 2s → 4s
                logger.warning(f"Rate limited. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"LLM call failed: {e}")
                raise


# ==============================================================================
# Smoke Test
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING GROQ LLM CONFIGURATION")
    print("=" * 60)

    # Test 1: Basic invocation
    print("\n[Test 1] Basic invocation...")
    response = llm.invoke("Say exactly: 'Groq is working.'")
    print(f"Response: {response.content}")

    # Test 2: Structured output
    print("\n[Test 2] Structured output...")
    from pydantic import BaseModel

    class TestModel(BaseModel):
        message: str
        status: str

    structured_llm = llm.with_structured_output(TestModel)
    result = structured_llm.invoke("Return a JSON with message='test' and status='ok'")
    print(f"message: {result.message}, status: {result.status}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60 + "\n")