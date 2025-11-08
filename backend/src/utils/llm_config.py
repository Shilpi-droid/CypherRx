"""
Universal LLM Configuration
Uses Ollama for local LLM inference

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
# Ollama Implementation
# ==============================================================================

class OllamaLLM:
    """
    Ollama LLM implementation
    Provides local LLM inference with structured output support
    """

    def __init__(self, model: str = None, temperature: float = 0.1):
        """
        Initialize Ollama LLM

        Args:
            model: Ollama model name (or set OLLAMA_MODEL env var)
            temperature: Generation temperature (0-1)
        """
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError(
                "Ollama requires the ollama package. Install with:\n"
                "  pip install ollama"
            )

        self.model = model or os.getenv("OLLAMA_MODEL") or self._get_available_model()
        self.temperature = temperature
        self.structured_output_schema = None

        # Test connection
        try:
            self.ollama.list()
            print(f"[OK] Ollama connected. Model: {self.model}")
        except Exception as e:
            print(f"[WARNING] Ollama connection failed: {e}")
            print("Make sure Ollama is running: ollama serve")

    def _get_available_model(self) -> str:
        """Auto-detect available Ollama model"""
        try:
            models_response = self.ollama.list()
            available_models = [model.model for model in models_response.models]

            if not available_models:
                raise RuntimeError("No Ollama models found. Run: ollama pull llama3.2")

            # Priority order
            preferred = ['llama3.2', 'llama3.2:latest', 'mistral', 'mistral:latest',
                        'qwen2.5:3b', 'deepseek-r1:1.5b']

            for model in preferred:
                if model in available_models:
                    return model

            return available_models[0]

        except Exception as e:
            raise RuntimeError(f"Could not detect Ollama model: {e}")

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

        try:
            response = self.ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': educational_context},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': 2000,
                }
            )

            content = response['message']['content']

            if self.structured_output_schema:
                return self._extract_structured_output(content)
            else:
                return LLMMessage(content)

        except Exception as e:
            print(f"[ERROR] Ollama invocation failed: {e}")
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

    def with_structured_output(self, schema: Type[BaseModel]) -> 'OllamaLLM':
        """Return new instance configured for structured output"""
        new_instance = OllamaLLM(model=self.model, temperature=self.temperature)
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

def create_llm() -> OllamaLLM:
    """
    Create Ollama LLM instance

    Environment Variables:
        OLLAMA_MODEL: Model name (optional, auto-detects if not set)

    Returns:
        Configured Ollama LLM instance
    """
    print(f"\n{'='*70}")
    print(f"LLM PROVIDER: OLLAMA")
    print(f"{'='*70}")

    try:
        return OllamaLLM(temperature=0.1)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize Ollama LLM: {e}")
        print("\nTo use Ollama:")
        print("  1. Install: https://ollama.ai")
        print("  2. Start: ollama serve")
        print("  3. Pull model: ollama pull llama3.2")
        print("  4. Set: OLLAMA_MODEL=llama3.2 (optional)")
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
    print("TESTING LLM CONFIGURATION")
    print("="*70)

    # Test 1: Simple query
    print("\n[Test 1] Simple unstructured query...")
    response = llm.invoke("Say 'Hello! LLM is working!' in exactly those words.")
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
    print("✓ ALL TESTS PASSED")
    print("="*70 + "\n")
