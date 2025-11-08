# LLM Configuration Guide

This project supports **multiple LLM providers** with a unified interface. Switch between providers using the `LLM_PROVIDER` environment variable.

## 🚀 Quick Start

### Option 1: Azure OpenAI (Recommended for Production)

1. **Set up Azure OpenAI** (if you don't have it already):
   - Go to [Azure Portal](https://portal.azure.com)
   - Create an Azure OpenAI resource
   - Deploy a model (e.g., GPT-4o, GPT-4, GPT-3.5-turbo)
   - Get your API key, endpoint, and deployment name

2. **Configure environment variables** in your `.env` file:

```bash
# Choose Azure as provider
LLM_PROVIDER=azure

# Azure credentials
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
OPENAI_API_VERSION=2024-08-01-preview
# Note: Use 2024-08-01-preview or later for structured output support
```

3. **Install dependencies**:

```bash
pip install langchain-openai
```

4. **Test it**:

```bash
cd backend
python -m src.utils.llm_config
```

---

### Option 2: Ollama (Free, Local)

1. **Install Ollama**:
   - Download from [https://ollama.ai](https://ollama.ai)
   - Or: `curl https://ollama.ai/install.sh | sh` (Linux/Mac)

2. **Start Ollama and pull a model**:

```bash
ollama serve
ollama pull llama3.2
```

3. **Configure environment variables** in your `.env` file:

```bash
# Choose Ollama as provider
LLM_PROVIDER=ollama

# Optional: Specify model (auto-detects if not set)
OLLAMA_MODEL=llama3.2
```

4. **Install dependencies**:

```bash
pip install ollama
```

5. **Test it**:

```bash
cd backend
python -m src.utils.llm_config
```

---

## 📝 Usage in Your Code

The LLM interface is **identical** regardless of provider:

```python
from backend.src.utils.llm_config import llm

# Simple query
response = llm.invoke("What is diabetes?")
print(response.content)

# Structured output with Pydantic
from pydantic import BaseModel

class DrugInfo(BaseModel):
    name: str
    drug_class: str
    indication: str

structured_llm = llm.with_structured_output(DrugInfo)
result = structured_llm.invoke("Extract drug info from: Metformin is a biguanide used for diabetes")
print(result.name)  # "Metformin"
print(result.drug_class)  # "Biguanide"
```

---

## 🔄 Switching Providers

Just change one environment variable:

```bash
# Switch to Azure
LLM_PROVIDER=azure

# Switch to Ollama
LLM_PROVIDER=ollama
```

**No code changes needed!** The interface is identical.

---

## 🛠️ Available Models

### Azure OpenAI
- `gpt-4o` (GPT-4 Omni) - Best overall
- `gpt-4` (GPT-4) - High quality
- `gpt-35-turbo` (GPT-3.5 Turbo) - Fast and cheap

### Ollama (Local)
- `llama3.2` - Best for medical reasoning ✅
- `mistral` - Fast and reliable
- `qwen2.5:3b` - Very fast, lightweight
- `deepseek-r1:1.5b` - Good for code

**Recommendation**: Use `llama3.2` for medical queries with Ollama.

---

## 🐛 Troubleshooting

### Azure OpenAI

**Error: Missing credentials**
```bash
# Make sure all these are set in .env:
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
OPENAI_API_VERSION=...
```

**Error: Deployment not found**
- Check your deployment name matches exactly what's in Azure Portal
- Make sure the model is deployed (not just created)

### Ollama

**Error: Connection refused**
```bash
# Start Ollama server:
ollama serve

# In another terminal, check if it's running:
ollama list
```

**Error: No models found**
```bash
# Pull a model:
ollama pull llama3.2
```

---

## 📊 Comparison

| Feature | Azure OpenAI | Ollama |
|---------|--------------|---------|
| **Cost** | Paid (per token) | Free |
| **Speed** | Fast (cloud) | Medium (local) |
| **Quality** | Excellent | Good |
| **Privacy** | Data sent to Azure | Fully local |
| **Setup** | Requires Azure account | Just install |
| **Internet** | Required | Not required |
| **Best For** | Production | Development, Privacy |

---

## 🎯 Recommendations

- **Development/Testing**: Use Ollama (free, local, private)
- **Production**: Use Azure OpenAI (better quality, scalable)
- **Student Projects**: Use Ollama (free tier limits on Azure)
- **Enterprise**: Use Azure OpenAI (SLA, support, compliance)

---

## 🔧 Advanced Configuration

### Custom Temperature

```python
# Lower temperature = more deterministic
# Higher temperature = more creative

# Edit llm_config.py:
llm = create_llm()  # Uses default temperature=0.1
```

### Custom Max Tokens

```python
# Edit the create_llm() function in llm_config.py:
return AzureLLM(temperature=0.1, max_tokens=4000)  # Default is 2000
```

### Adding New Providers

To add support for another provider (e.g., OpenAI, Anthropic):

1. Create a new class (e.g., `OpenAILLM`) in `llm_config.py`
2. Implement these methods:
   - `__init__()`
   - `invoke(messages)`
   - `with_structured_output(schema)`
3. Add to `create_llm()` function:
   ```python
   elif provider == "openai":
       return OpenAILLM(temperature=0.1)
   ```

---

## 📚 Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

---

## 🤝 Support

If you encounter issues:

1. Check the error message carefully
2. Verify your `.env` file settings
3. Test with the included test script: `python -m src.utils.llm_config`
4. Check provider-specific troubleshooting above

---

**Made with ❤️ for easy LLM provider switching**

