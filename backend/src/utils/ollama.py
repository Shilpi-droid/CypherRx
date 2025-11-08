# """
# LLM Configuration for Medical Data Extraction
# Uses: Google Gemini 2.5 Flash (Free tier: 1500 requests/day)
# """

# import os
# from typing import Optional
# import google.generativeai as genai
# from dotenv import load_dotenv
# load_dotenv()
# class GeminiLLM:
#     """Wrapper for Google Gemini 2.5 Flash model"""
    
#     def __init__(self, api_key: Optional[str] = None):
#         """
#         Initialize Gemini LLM
        
#         Args:           
#             api_key: Google API key (get free at: https://aistudio.google.com/app/apikey)
#         """       
    
#         # Configure Gemini
#         try:
#             token = api_key or os.getenv("GEMINI_API_KEY")
#             if not token:
#                 raise ValueError(
#                     "GEMINI_API_KEY required!\n"
#                     "Get free API key at: https://aistudio.google.com/app/apikey\n"
#                     "Set as: export GEMINI_API_KEY='your_key' or pass as argument"
#                 )
            
#             genai.configure(api_key=token)
#             self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
#             print("✓ Using Google Gemini 2.0 Flash Exp (Free tier: 10 RPM, 1500 requests/day)")
            
#         except Exception as e:
#             raise RuntimeError(f"Gemini API error: {e}")

    
#     def invoke(self, prompt: str, max_tokens: int = 2000) -> dict:
#         """
#         Invoke LLM with prompt and return response
        
#         Args:
#             prompt: The prompt text
#             max_tokens: Maximum tokens to generate
            
#         Returns:
#             dict with 'content' key containing the response
#         """
#         try:
#             response = self.model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     max_output_tokens=max_tokens,
#                     temperature=0.1,  # Low temp for structured extraction
#                 )
#             )
#             return {"content": response.text}
                
#         except Exception as e:
#             print(f"LLM Error: {e}")
#             return {"content": "{}"}


# def create_llm():
#     """Create LLM instance based on environment"""
#     return GeminiLLM()
    

# llm = create_llm()


# def main():
#     """Test Gemini LLM configuration"""
#     print("\n" + "="*70)
#     print("TESTING GEMINI LLM CONFIGURATION")
#     print("="*70)
    
#     # Test 1: Simple query
#     print("\n[Test 1] Simple query...")
#     response = llm.invoke("Say 'Hello! Gemini is working!' in exactly those words.")
#     print(f"Response: {response['content']}")
    
#     # Test 2: JSON extraction
#     print("\n[Test 2] Structured JSON extraction...")
#     prompt = """
#     Extract the following information and return ONLY valid JSON:
    
#     Text: "Aspirin 81mg is used for pain relief. Take 1-2 tablets every 4-6 hours. 
#     Do not use if allergic to NSAIDs. Common side effects include stomach upset."
    
#     Return JSON with these fields:
#     {
#         "drug_name": "...",
#         "dosage": "...",
#         "indication": "...",
#         "contraindication": "...",
#         "side_effects": ["..."]
#     }
    
#     Return ONLY the JSON, no other text.
#     """
#     response = llm.invoke(prompt, max_tokens=500)
#     print(f"Response:\n{response['content']}")
    
#     # Test 3: Verify it's valid JSON
#     print("\n[Test 3] Validating JSON...")
#     try:
#         import json
#         data = json.loads(response['content'].strip().replace('```json', '').replace('```', ''))
#         print(f"✓ Valid JSON!")
#         print(f"  Drug: {data.get('drug_name', 'N/A')}")
#         print(f"  Dosage: {data.get('dosage', 'N/A')}")
#         print(f"  Side effects: {data.get('side_effects', [])}")
#     except Exception as e:
#         print(f"✗ JSON parsing failed: {e}")
    
#     print("\n" + "="*70)
#     print("✓ ALL TESTS COMPLETE")
#     print("="*70)
#     print("\nGemini is ready to use!")
#     print("Usage: from src.utils.llm_config import llm")
#     print("       response = llm.invoke('your prompt here')")
#     print("="*70 + "\n")


# if __name__ == "__main__":
#     main()


# llm_config.py
"""
LLM Configuration - DeepSeek via Ollama
Supports both unstructured and structured output (Pydantic)
"""

import ollama
import json
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Disable httpx INFO logging (comment out to see HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)


class OllamaMessage:
    """Message format compatible with LangChain"""
    def __init__(self, content: str):
        self.content = content


class OllamaLLM:
    """
    Ollama wrapper compatible with LangChain interface
    Supports .invoke() and .with_structured_output()
    """
    
    def __init__(self, model: str = "deepseek-r1:1.5b", temperature: float = 0.1):
        """
        Initialize Ollama LLM
        
        Args:
            model: Ollama model name (e.g., 'deepseek-r1:1.5b', 'llama3.2', 'mistral')
            temperature: Generation temperature (0-1)
        
        Note: Make sure model is pulled first: ollama pull deepseek-r1:1.5b
        """
        self.model = model
        self.temperature = temperature
        self.structured_output_schema = None
        
        # Test connection
        try:
            ollama.list()
            # print(f"[OK] Ollama connected. Using model: {model}")
        except Exception as e:
            print(f"[WARNING] Ollama connection failed: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def invoke(self, messages: Any) -> OllamaMessage:
        """
        Invoke LLM with messages (compatible with LangChain)
        
        Args:
            messages: Either a string prompt or list of message dicts
            
        Returns:
            OllamaMessage with content attribute
        """
        # Handle different input formats
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            # Convert LangChain message format to Ollama format
            ollama_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = 'system' if msg.__class__.__name__ == 'SystemMessage' else 'user'
                    ollama_messages.append({
                        'role': role,
                        'content': msg.content
                    })
                elif isinstance(msg, dict):
                    ollama_messages.append(msg)
            
            # Combine into single prompt for simpler models
            prompt = "\n\n".join([m.get('content', '') for m in ollama_messages])
        else:
            prompt = str(messages)
        
        # CRITICAL: Add educational disclaimer to avoid model refusals
        educational_context = """IMPORTANT CONTEXT: This is a student educational project for an AI/ML portfolio.
This is a demonstration of knowledge graph reasoning and LLM integration.
NO real medical advice will be given to patients.
You are analyzing a sample medical knowledge graph for educational purposes only.
Provide direct, factual information based solely on the graph data provided."""
        
        full_prompt = f"{educational_context}\n\n{prompt}"
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': educational_context},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': 2000,  # Max tokens
                }
            )
            
            content = response['message']['content']
            
            # If structured output requested, return Pydantic object directly
            if self.structured_output_schema:
                return self._extract_and_validate_json(content, self.structured_output_schema)
            
            return OllamaMessage(content)
            
        except Exception as e:
            print(f"[ERROR] Ollama invocation failed: {e}")
            
            # If structured output requested, return default instance with explicit defaults
            if self.structured_output_schema:
                schema_name = self.structured_output_schema.__name__
                print(f"[INFO] Creating fallback default for {schema_name}")
                
                # Hardcoded defaults for known schemas
                if schema_name == "QueryClassification":
                    return self.structured_output_schema(
                        query_type="simple_aggregation",
                        reasoning="Unable to classify - using default"
                    )
                elif schema_name == "ExtractedEntities":
                    return self.structured_output_schema(
                        drugs=[],
                        conditions=[],
                        drug_classes=[]
                    )
                else:
                    # Generic fallback - try to construct empty version
                    try:
                        return self.structured_output_schema.model_construct()
                    except:
                        print(f"[ERROR] Could not create fallback for {schema_name}")
                        raise RuntimeError(f"Ollama failed and no fallback available for {schema_name}")
            
            # Return empty response for unstructured
            return OllamaMessage("{}")
    
    def _extract_and_validate_json(self, text: str, schema: Type[BaseModel]) -> BaseModel:
        """Extract and validate JSON against Pydantic schema, return Pydantic object"""
        import re
        
        original_text = text
        text = text.strip()
        
        # Strategy 1: Look for JSON in markdown code blocks
        if '```json' in text:
            try:
                json_text = text.split('```json')[1].split('```')[0].strip()
                data = json.loads(json_text)
                return schema(**data)
            except:
                pass
        
        # Strategy 2: Look for any code blocks
        if '```' in text:
            try:
                json_text = text.split('```')[1].split('```')[0].strip()
                # Remove language identifier if present
                if json_text.startswith('json'):
                    json_text = json_text[4:].strip()
                data = json.loads(json_text)
                return schema(**data)
            except:
                pass
        
        # Strategy 3: Look for JSON object with regex
        try:
            # Find content between first { and last }
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                data = json.loads(json_text)
                return schema(**data)
        except:
            pass
        
        # Strategy 4: Try parsing the entire text as JSON
        try:
            data = json.loads(text)
            return schema(**data)
        except:
            pass
        
        # Strategy 5: If it's a classification response, try to extract from plain text
        schema_name = schema.__name__
        if schema_name == "QueryClassification":
            try:
                # Look for query type keywords
                text_lower = text.lower()
                if "interaction_check" in text_lower or "interaction check" in text_lower:
                    return schema(query_type="interaction_check", reasoning=text[:100])
                elif "filtered_aggregation" in text_lower or "filtered aggregation" in text_lower:
                    return schema(query_type="filtered_aggregation", reasoning=text[:100])
                elif "simple_aggregation" in text_lower or "simple aggregation" in text_lower:
                    return schema(query_type="simple_aggregation", reasoning=text[:100])
                elif "path_traversal" in text_lower or "path traversal" in text_lower:
                    return schema(query_type="path_traversal", reasoning=text[:100])
            except:
                pass
        
        # Strategy 6: If it's entity extraction, try to parse from plain text
        if schema_name == "ExtractedEntities":
            try:
                # Look for drug/condition mentions
                drugs = []
                conditions = []
                
                # Try to find lists in the text
                if "drugs:" in text.lower():
                    drugs_section = text.lower().split("drugs:")[1].split("\n")[0]
                    drugs = [d.strip() for d in drugs_section.split(",") if d.strip()]
                
                if "conditions:" in text.lower():
                    cond_section = text.lower().split("conditions:")[1].split("\n")[0]
                    conditions = [c.strip() for c in cond_section.split(",") if c.strip()]
                
                return schema(drugs=drugs, conditions=conditions, drug_classes=[])
            except:
                pass
        
        # All strategies failed - use hardcoded fallbacks
        print(f"[WARNING] All JSON extraction strategies failed for {schema_name}")
        print(f"[DEBUG] Original text (first 300 chars): {original_text[:300]}")
        
        if schema_name == "QueryClassification":
            return schema(
                query_type="simple_aggregation",
                reasoning="Failed to parse LLM response - using default"
            )
        elif schema_name == "ExtractedEntities":
            return schema(
                drugs=[],
                conditions=[],
                drug_classes=[]
            )
        else:
            # Generic fallback
            try:
                return schema.model_construct()
            except:
                raise RuntimeError(f"Could not create fallback for {schema_name}")
    
    def with_structured_output(self, schema: Type[BaseModel]) -> 'OllamaLLM':
        """
        Return a version of this LLM that outputs structured data
        
        Args:
            schema: Pydantic model class
            
        Returns:
            New OllamaLLM instance configured for structured output
        """
        new_instance = OllamaLLM(model=self.model, temperature=self.temperature)
        new_instance.structured_output_schema = schema
        return new_instance


# Initialize global LLM instance with auto-detection
def get_available_model():
    """Detect and return an available Ollama model"""
    try:
        # Get list of available models
        models_response = ollama.list()
        # Extract model names from ListResponse.Model objects
        available_models = [model.model for model in models_response.models]
        
        if not available_models:
            return None, []
        
        # Priority order of preferred models
        preferred_models = [
            'llama3.2',
            'llama3.2:latest', 
            'mistral',
            'mistral:latest',
            'qwen2.5:3b',
            'deepseek-coder:1.3b',
            'deepseek-r1:1.5b',
        ]
        
        # Try to find a preferred model
        for preferred in preferred_models:
            if preferred in available_models:
                return preferred, available_models
        
        # If no preferred model, use first available
        return available_models[0], available_models
        
    except Exception as e:
        print(f"[WARNING] Could not list Ollama models: {e}")
        import traceback
        traceback.print_exc()
        return None, []

# Try to get model from environment or auto-detect
import os
manual_model = os.getenv("OLLAMA_MODEL")

if manual_model:
    model_name = manual_model
    print(f"[INFO] Using manually specified model: {model_name}")
else:
    model_name, available = get_available_model()
    if model_name:
        print(f"[INFO] Auto-detected model: {model_name}")
        print(f"[INFO] Available models: {', '.join(available)}")
    else:
        print("[ERROR] No Ollama models found!")
        print("\n" + "="*70)
        print("OLLAMA SETUP REQUIRED:")
        print("="*70)
        print("1. Install Ollama from: https://ollama.ai")
        print("2. Start Ollama server: ollama serve")
        print("3. Pull a model:")
        print("   - ollama pull llama3.2  (Recommended for accuracy)")
        print("   - ollama pull mistral   (Fast and reliable)")
        print("   - ollama pull qwen2.5:3b (Very fast)")
        print("4. Restart this application")
        print("="*70 + "\n")
        raise RuntimeError("No Ollama models available")

try:
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1
    )
    print(f"[OK] LLM provider initialized successfully (Ollama + {model_name}).")
    
    # Warn if using coding model for medical queries
    if 'coder' in model_name.lower() or 'code' in model_name.lower():
        print("\n" + "="*70)
        print("WARNING: You're using a CODING model for MEDICAL queries!")
        print("This may result in poor accuracy and refusals.")
        print("")
        print("Recommended: Pull a better model for medical reasoning:")
        print("  ollama pull llama3.2      (Best accuracy)")
        print("  ollama pull mistral       (Good balance)")
        print("  ollama pull qwen2.5:3b    (Fastest)")
        print("="*70 + "\n")
        
except Exception as e:
    print(f"[ERROR] Failed to initialize Ollama LLM with {model_name}: {e}")
    print("\nTry one of these commands:")
    print(f"  ollama pull {model_name}")
    print("  ollama pull llama3.2")
    print("  ollama pull mistral")
    raise
