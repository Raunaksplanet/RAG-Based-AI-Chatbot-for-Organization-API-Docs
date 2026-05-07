# RAG API Docs Chatbot

Simple AI chatbot for API documentation using RAG + local LLMs with Ollama. Fully self-hosted, no cloud APIs.

## Stack

* Ollama
* `deepseek-r1:1.5b`
* `nomic-embed-text`
* ChromaDB
* FastAPI
* LangChain / Python

## Flow

```text
OpenAPI/Swagger JSON
        ↓
Chunk + Embed
        ↓
Store in ChromaDB
        ↓
Retrieve relevant docs
        ↓
Generate answer with DeepSeek
```

## Install

```bash
brew install ollama

ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text

pip install chromadb langchain langchain-community jsonref fastapi uvicorn requests
```

## Usage

### 1. Add your API spec

```bash
cp your-api-spec.json swagger.json
```

### 2. Ingest docs

```bash
python main.py
```

### 3. Run API server

```bash
uvicorn main:app --reload
```

Server runs on:

```text
http://localhost:8000
```

## Query Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I authenticate?"}'
```
