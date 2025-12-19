# Model Configuration Guide

## Two Models, Two Purposes

1. **Embedding Model** - Converts code to vectors for search
2. **Chat Model** - Powers AI conversations

Configure them independently in `.env`:

---

## Common Configurations

### 1. All Local (Free, Private)
```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large
CHAT_PROVIDER=ollama
CHAT_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

**Setup**: `ollama pull mxbai-embed-large && ollama pull qwen2.5:7b`

### 2. All Cloud (Best Quality)
```bash
OPENAI_API_KEY=sk-your-key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
CHAT_PROVIDER=openai
CHAT_MODEL=gpt-4o
```

### 3. Hybrid (Cost Effective)
```bash
GEMINI_API_KEY=your-key
EMBEDDING_PROVIDER=default  # Free, local
CHAT_PROVIDER=gemini
CHAT_MODEL=gemini-1.5-flash
```

### 4. Testing (No Cost)
```bash
EMBEDDING_PROVIDER=default
# Chat still requires API key - use Gemini free tier
```

---

## Options Summary

| Provider | Embedding Models | Chat Models |
|----------|-----------------|-------------|
| **openai** | text-embedding-3-small | gpt-4o, gpt-3.5-turbo |
| **gemini** | (automatic) | gemini-1.5-flash, gemini-1.5-pro |
| **ollama** | mxbai-embed-large, nomic-embed-text | qwen2.5, llama3.2, deepseek-r1 |
| **default** | Sentence Transformers | N/A |

---

## Troubleshooting

**Ollama not working?**
```bash
ollama serve  # Start server
curl http://localhost:11434/api/tags  # Test connection
```

**Embedding conflict error?**
```bash
rm -rf db/  # Delete old database
```

**Gemini rate limit?**
```bash
EMBEDDING_PROVIDER=default  # Switch to local
```

---

## Recommendations

- **Development**: Ollama (free, private)
- **Production**: OpenAI (best quality)
- **Testing**: Default + Gemini (free)

