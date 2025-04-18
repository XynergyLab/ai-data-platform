# LLM Inference Service

This service provides LLM (Large Language Model) inference capabilities using both Ollama and Hugging Face's Text Generation Inference (TGI) servers.

## Components

1. Ollama Server
   - Runs on port 11434
   - Supports multiple models (llama2, mistral, codellama)
   - Optimized for CPU and GPU inference

2. Text Generation Inference (TGI)
   - Runs on port 8080
   - Optimized for Hugging Face models
   - Supports batched inference

## Configuration

1. Copy `.env.example` to `.env` and adjust settings
2. Place models in the appropriate directories:
   - Ollama models: `./models`
   - Hugging Face models: `./models/hf`

## Usage

Start the service:
```bash
podman-compose up -d
```

Health check:
```bash
curl http://localhost:11434/health  # Ollama
curl http://localhost:8080/health   # TGI
```

## Resource Requirements

- Minimum 32GB RAM
- 4+ CPU cores
- NVIDIA GPU (optional but recommended)
- 100GB+ storage for models

## Network

- Part of the global_network for service mesh integration
- Dedicated ai_network for AI service communication
