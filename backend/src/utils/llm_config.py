"""
Universal LLM Configuration
Supports multiple providers: Azure ChatOpenAI, Ollama, etc.
Switch providers using LLM_PROVIDER environment variable

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
# Base Message Class (Provider-Agnostic)
# ==============================================================================

class LLMMessage:
    """Universal message format for all providers"""
    def __init__(self, content: str):
        self.content = content


# ==============================================================================
# Azure ChatOpenAI Implementation
# ==============================================================================

class AzureLLM:
    """
    Azure OpenAI LLM implementation
    Compatible with LangChain interface
    """
    
    def __init__(self, deployment_name: Optional[str] = None, temperature: float = 0.1, max_tokens: int = 2000):
        """
        Initialize Azure OpenAI LLM
        
        Required environment variables:
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT
        - AZURE_OPENAI_DEPLOYMENT_NAME (or pass as argument)
        - OPENAI_API_VERSION (e.g., "2024-02-01")
        """
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
        except ImportError:
            raise ImportError(
                "Azure OpenAI requires langchain-openai. Install with:\n"
                "  pip install langchain-openai"
            )
        
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # Use newer API version that supports structured output (json_schema)
        self.api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError(
                "Missing Azure OpenAI credentials. Required environment variables:\n"
                "  AZURE_OPENAI_API_KEY\n"
                "  AZURE_OPENAI_ENDPOINT\n"
                "  AZURE_OPENAI_DEPLOYMENT_NAME\n"
                "  OPENAI_API_VERSION (optional, defaults to 2024-08-01-preview)"
            )
        
        self.client = AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        self.structured_output_schema = None
        print(f"[OK] Azure OpenAI connected. Deployment: {self.deployment_name}")
    
    def invoke(self, messages: Any) -> Any:
        """
        Invoke LLM with messages
        
        Args:
            messages: String, list of messages, or LangChain messages
            
        Returns:
            LLMMessage (unstructured) or Pydantic object (structured)
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        
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
        
        # Add educational context (prevents medical advice refusals)
        educational_context = (
            "IMPORTANT CONTEXT: This is a student educational project for an AI/ML portfolio. "
            "This is a demonstration of knowledge graph reasoning and LLM integration. "
            "NO real medical advice will be given to patients. "
            "You are analyzing a sample medical knowledge graph for educational purposes only. "
            "Provide direct, factual information based solely on the graph data provided."
        )
        
        final_messages = [
            SystemMessage(content=educational_context),
            HumanMessage(content=prompt)
        ]
        
        try:
            if self.structured_output_schema:
                # Try structured output first
                try:
                    structured_llm = self.client.with_structured_output(self.structured_output_schema)
                    return structured_llm.invoke(final_messages)
                except Exception as structured_error:
                    # If structured output fails (e.g., old API version), fall back to JSON parsing
                    print(f"[WARNING] Structured output failed: {structured_error}")
                    print("[INFO] Falling back to JSON parsing from text response")
                    
                    # Get regular text response and parse it
                    response = self.client.invoke(final_messages)
                    return self._extract_structured_output(response.content)
            else:
                # Standard response
                response = self.client.invoke(final_messages)
                return LLMMessage(content=response.content)
        
        except Exception as e:
            print(f"[ERROR] Azure OpenAI invocation failed: {e}")
            return self._get_fallback_response()
    
    def with_structured_output(self, schema: Type[BaseModel]) -> 'AzureLLM':
        """Return new instance configured for structured output"""
        new_instance = AzureLLM(
            deployment_name=self.deployment_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        new_instance.structured_output_schema = schema
        return new_instance
    
    def _extract_structured_output(self, text: str) -> BaseModel:
        """Extract and validate structured output from text (fallback for old API versions)"""
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
# Ollama Implementation
# ==============================================================================

class OllamaLLM:
    """
    Ollama LLM implementation
    Compatible with LangChain interface
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
            messages: String, list of messages, or LangChain messages
            
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
# Provider Selection and Global Instance
# ==============================================================================

def create_llm() -> Any:
    """
    Create LLM instance based on LLM_PROVIDER environment variable
    
    Environment Variables:
        LLM_PROVIDER: "azure" or "ollama" (default: "azure")
        
        For Azure:
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT
        - AZURE_OPENAI_DEPLOYMENT_NAME
        - OPENAI_API_VERSION (optional)
        
        For Ollama:
        - OLLAMA_MODEL (optional, auto-detects if not set)
    
    Returns:
        Configured LLM instance
    """
    provider = os.getenv("LLM_PROVIDER", "azure").lower()
    
    print(f"\n{'='*70}")
    print(f"LLM PROVIDER: {provider.upper()}")
    print(f"{'='*70}")
    
    if provider == "azure":
        try:
            return AzureLLM(temperature=0.1)
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize Azure LLM: {e}")
            print("\nTo use Azure OpenAI, set these environment variables:")
            print("  LLM_PROVIDER=azure")
            print("  AZURE_OPENAI_API_KEY=your_key")
            print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            print("  AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment")
            print("  OPENAI_API_VERSION=2024-02-01")
            print("\nOr switch to Ollama: LLM_PROVIDER=ollama")
            raise
    
    elif provider == "ollama":
        try:
            return OllamaLLM(temperature=0.1)
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize Ollama LLM: {e}")
            print("\nTo use Ollama:")
            print("  1. Install: https://ollama.ai")
            print("  2. Start: ollama serve")
            print("  3. Pull model: ollama pull llama3.2")
            print("  4. Set: LLM_PROVIDER=ollama")
            print("\nOr switch to Azure: LLM_PROVIDER=azure")
            raise
    
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}\n"
            f"Valid options: 'azure', 'ollama'"
        )


# ==============================================================================
# Global LLM Instance (used throughout the project)
# ==============================================================================

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
