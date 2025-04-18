# Embedding Services

This service provides vector embedding capabilities for both text and images, supporting various downstream tasks like similarity search, clustering, and classification.

## Components

1. Text Embedding Service
   - Based on Hugging Face's text-embeddings-inference
   - Uses BAAI/bge-large-en-v1.5 by default
   - Optimized for high-throughput batch processing
   - Runs on port 8081

2. CLIP Embedding Service
   - Uses OpenAI's CLIP model for image and text embeddings
   - Supports cross-modal similarity computation
   - Runs on port 8082

## Configuration

1. Copy `.env.example` to `.env` and adjust settings
2. Download models:
   ```bash
   # For text embeddings
   huggingface-cli download BAAI/bge-large-en-v1.5
   # For CLIP
   huggingface-cli download openai/clip-vit-large-patch14
   ```

## Usage

Start the services:
```bash
podman-compose up -d
```

Example API calls:

1. Text Embeddings:
```bash
curl -X POST http://localhost:8081/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello, world!", "Another text"]}'
```

2. Image Embeddings:
```bash
curl -X POST http://localhost:8082/embed_image \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/image.jpg"
```

## Resource Requirements

- Minimum 16GB RAM total
- 4+ CPU cores
- GPU optional but recommended for higher throughput
- 20GB+ storage for models

## Integration

- Connects with vector stores (Milvus/Qdrant) for similarity search
- Supports caching via Redis
- Part of the global service mesh
