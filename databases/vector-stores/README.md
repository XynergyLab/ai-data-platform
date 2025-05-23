# Vector Store Services

This configuration sets up a production-ready vector database environment using both Milvus and Qdrant, with supporting services for distributed coordination and object storage.

## Components

1. Milvus
   - Primary vector database
   - Scalable similarity search
   - Ports: 19530 (API), 9091 (metrics)

2. Qdrant
   - Alternative vector database
   - Filtering and payload support
   - Ports: 6333 (HTTP), 6334 (gRPC)

3. Supporting Services
   - etcd: Metadata storage and coordination
   - MinIO: Object storage for Milvus

## Configuration

1. Copy `.env.example` to `.env` and adjust settings
2. Create necessary directories:
   ```bash
   mkdir -p milvus/data milvus/config
   mkdir -p qdrant/storage
   mkdir -p minio
   mkdir -p etcd
   ```

## Usage

Start the services:
```bash
podman-compose up -d
```

### Milvus Example
```python
from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")
collection = Collection("example")
```

### Qdrant Example
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
```

## Resource Requirements

- Minimum 32GB RAM total
- 8+ CPU cores
- Fast SSD storage (500GB+ recommended)
- High I/O capacity for concurrent operations

## Integration

- Works with embedding services for vector generation
- Supports both text and image vector storage
- Horizontal scaling capabilities
- Backup and recovery procedures
