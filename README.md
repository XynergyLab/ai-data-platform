# AI and Data Processing Platform

A comprehensive platform for AI services, data processing, and monitoring using Podman containers.

## Architecture Overview

```
AI Platform
├── ai-services/
│   ├── llm-inference/      # LLM inference services
│   ├── embedding-services/ # Vector embedding generation
│   └── multimodal/        # Image, audio, video processing
├── data-processing/
│   ├── pipelines/         # Airflow DAGs and ETL
│   ├── batch-processing/  # Spark batch jobs
│   └── preprocessing/     # Data validation and prep
├── databases/
│   ├── vector-stores/     # Milvus and Qdrant
│   └── time-series/      # Time-series data
├── monitoring/
│   ├── prometheus/       # Metrics collection
│   ├── grafana/         # Visualization
│   └── loki/           # Log aggregation
└── shared/
    ├── configs/        # Shared configurations
    ├── networks/       # Network definitions
    └── volumes/        # Persistent storage
```

## Quick Start

1. Prerequisites:
   - Podman installed and configured
   - WSL2 with Fedora 41
   - Minimum 64GB RAM
   - 500GB+ storage
   - NVIDIA GPU (recommended)

2. Environment Setup:
   ```powershell
   # Clone configuration files
   cd C:\Users\drew\Documents\Podman_Compose
   
   # Copy environment files
   Get-ChildItem -Recurse -Filter ".env.example" | 
   ForEach-Object {
       Copy-Item $_.FullName ($_.FullName -replace '\.example$','')
   }
   ```

3. Start Services:
   ```powershell
   # Start core services
   .\deploy.ps1 start-core
   
   # Start AI services
   .\deploy.ps1 start-ai
   
   # Start monitoring
   .\deploy.ps1 start-monitoring
   ```

## Service Access Points

- LLM Inference: http://localhost:11434
- Embedding Service: http://localhost:8081
- Multimodal Processing: http://localhost:8090-8092
- Airflow UI: http://localhost:8080
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Vector Stores:
  - Milvus: localhost:19530
  - Qdrant: localhost:6333

## Component Documentation

- [AI Services](./ai-services/README.md)
- [Data Processing](./data-processing/README.md)
- [Vector Stores](./databases/vector-stores/README.md)
- [Monitoring](./monitoring/README.md)

## Resource Requirements

Total system requirements:
- RAM: 64GB minimum
- CPU: 16+ cores recommended
- Storage: 500GB+ SSD
- GPU: NVIDIA with 8GB+ VRAM recommended

## Network Configuration

- Each service group has its own network
- Global network for inter-service communication
- Traefik for service mesh and routing

## Maintenance

1. Backup Procedures:
   ```powershell
   .\deploy.ps1 backup-all
   ```

2. Updates:
   ```powershell
   .\deploy.ps1 update-services
   ```

3. Monitoring:
   - Check Grafana dashboards
   - Monitor resource usage
   - Review logs in Loki

## Security Notes

- All services run in isolated networks
- Default credentials must be changed
- Regular security updates required
- Access control implemented via Traefik
