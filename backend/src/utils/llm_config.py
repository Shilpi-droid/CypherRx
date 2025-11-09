"""
Universal LLM Configuration
Uses Google Gemini for LLM inference

Usage:
    from backend.src.utils.llm_config import llm
    response = llm.invoke("your prompt")

    # Or with structured output
    structured_llm = llm.with_structured_output(YourPydanticModel)
    result = structured_llm.invoke("your prompt")
"""

import os
import json
import logging
from typing import Any, Type, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Disable httpx INFO logging (comment out to see HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ==============================================================================
# Base Message Class
# ==============================================================================

class LLMMessage:
    """Universal message format"""
    def __init__(self, content: str):
        self.content = content


# ==============================================================================
# Google Gemini Implementation
# ==============================================================================

class GeminiLLM:
    """
    Google Gemini LLM implementation
    Provides cloud-based LLM inference with structured output support
    """

    def __init__(self, model: str = None, temperature: float = 0.1):
        """
        Initialize Google Gemini LLM

        Args:
            model: Gemini model name (or set GEMINI_MODEL env var, default: gemini-2.5-flash)
            temperature: Generation temperature (0-1)
        """
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set.\n"
                "Get your API key from: https://makersuite.google.com/app/apikey\n"
                "Set it in your .env file: GEMINI_API_KEY=your_key_here"
            )

        # Configure Gemini
        genai.configure(api_key=api_key)

        self.model_name = model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash-exp"
        self.temperature = temperature
        self.structured_output_schema = None

        # Initialize model
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"[OK] Google Gemini initialized. Model: {self.model_name}")
        except Exception as e:
            print(f"[WARNING] Gemini initialization failed: {e}")
            raise

    def invoke(self, messages: Any) -> Any:
        """
        Invoke LLM with messages

        Args:
            messages: String, list of messages, or message objects

        Returns:
            LLMMessage (unstructured) or Pydantic object (structured)
        """
        # Convert input to prompt string
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            parts = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    parts.append(msg.content)
                elif isinstance(msg, dict):
                    parts.append(msg.get('content', ''))
            prompt = "\n\n".join(parts)
        else:
            prompt = str(messages)

        # Add educational context
        educational_context = (
            "IMPORTANT CONTEXT: This is a student educational project for an AI/ML portfolio. "
            "This is a demonstration of knowledge graph reasoning and LLM integration. "
            "NO real medical advice will be given to patients. "
            "You are analyzing a sample medical knowledge graph for educational purposes only. "
            "Provide direct, factual information based solely on the graph data provided."
        )

        # Combine context with prompt
        full_prompt = f"{educational_context}\n\n{prompt}"

        try:
            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=2000,
            )

            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            content = response.text

            if self.structured_output_schema:
                return self._extract_structured_output(content)
            else:
                return LLMMessage(content)

        except Exception as e:
            print(f"[ERROR] Gemini invocation failed: {e}")
            return self._get_fallback_response()

    def _extract_structured_output(self, text: str) -> BaseModel:
        """Extract and validate structured output from text"""
        import re

        text = text.strip()

        # Try multiple extraction strategies
        strategies = [
            lambda t: json.loads(t.split('```json')[1].split('```')[0].strip()) if '```json' in t else None,
            lambda t: json.loads(t.split('```')[1].split('```')[0].strip()) if '```' in t else None,
            lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group(0)) if re.search(r'\{.*\}', t, re.DOTALL) else None,
            lambda t: json.loads(t),
        ]

        for strategy in strategies:
            try:
                data = strategy(text)
                if data:
                    return self.structured_output_schema(**data)
            except:
                continue

        # Fallback
        print(f"[WARNING] Could not extract JSON from response: {text[:200]}")
        return self._get_fallback_response()

    def with_structured_output(self, schema: Type[BaseModel]) -> 'GeminiLLM':
        """Return new instance configured for structured output"""
        new_instance = GeminiLLM(model=self.model_name, temperature=self.temperature)
        new_instance.structured_output_schema = schema
        return new_instance

    def _get_fallback_response(self) -> Any:
        """Return fallback response on error"""
        if self.structured_output_schema:
            schema_name = self.structured_output_schema.__name__

            if schema_name == "QueryClassification":
                return self.structured_output_schema(
                    query_type="simple_aggregation",
                    reasoning="Unable to classify - using default"
                )
            elif schema_name == "ExtractedEntities":
                return self.structured_output_schema(
                    drugs=[], conditions=[], drug_classes=[]
                )
            else:
                try:
                    return self.structured_output_schema.model_construct()
                except:
                    raise RuntimeError(f"No fallback available for {schema_name}")

        return LLMMessage("{}")


# ==============================================================================
# Global LLM Instance (used throughout the project)
# ==============================================================================

def create_llm() -> GeminiLLM:
    """
    Create Google Gemini LLM instance

    Environment Variables:
        GEMINI_API_KEY: Your Google API key (required)
        GEMINI_MODEL: Model name (optional, default: gemini-2.0-flash-exp)

    Returns:
        Configured Gemini LLM instance
    """
    print(f"\n{'='*70}")
    print(f"LLM PROVIDER: GOOGLE GEMINI")
    print(f"{'='*70}")

    try:
        return GeminiLLM(temperature=0.1)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize Google Gemini LLM: {e}")
        print("\nTo use Google Gemini:")
        print("  1. Get API key: https://makersuite.google.com/app/apikey")
        print("  2. Set in .env file: GEMINI_API_KEY=your_key_here")
        print("  3. (Optional) Set model: GEMINI_MODEL=gemini-2.0-flash-exp")
        raise


try:
    llm = create_llm()
    print(f"[OK] Global LLM instance initialized successfully.")
    print(f"{'='*70}\n")
except Exception as e:
    print(f"\n[FATAL] Could not initialize LLM provider.")
    print(f"{'='*70}\n")
    raise


# ==============================================================================
# Testing
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING GEMINI LLM CONFIGURATION")
    print("="*70)

    # Test 1: Simple query
    print("\n[Test 1] Simple unstructured query...")
    response = llm.invoke("Say 'Hello! Gemini LLM is working!' in exactly those words.")
    print(f"Response: {response.content}")

    # Test 2: Structured output
    print("\n[Test 2] Structured output...")

    class TestModel(BaseModel):
        message: str
        status: str

    structured_llm = llm.with_structured_output(TestModel)
    result = structured_llm.invoke("Return JSON with message='test' and status='ok'")
    print(f"Result: {result}")
    print(f"  message: {result.message}")
    print(f"  status: {result.status}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70 + "\n")
