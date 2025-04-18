# Data Processing Services

This service provides comprehensive data processing capabilities including ETL pipelines, batch processing, and data validation using Apache Airflow, Apache Spark, and custom validation services.

## Components

1. Apache Airflow
   - Workflow orchestration
   - DAG management
   - Task scheduling
   - Runs on port 8080

2. Apache Spark
   - Distributed data processing
   - Batch processing
   - Data transformation
   - Master UI on port 8181

3. Data Validator
   - Custom validation service
   - Schema validation
   - Data quality checks
   - Real-time validation API

## Configuration

1. Copy `.env.example` to `.env` and adjust settings
2. Set up necessary directories:
   ```bash
   mkdir -p pipelines/dags
   mkdir -p pipelines/plugins
   mkdir -p pipelines/logs
   mkdir -p batch-processing/spark
   mkdir -p preprocessing/config
   ```

3. Configure validation rules in `preprocessing/config/validation_rules.json`

## Usage

Start the services:
```bash
podman-compose up -d
```

Access points:
- Airflow UI: http://localhost:8080
- Spark Master UI: http://localhost:8181
- Data Validator API: http://localhost:5000

Example validation API call:
```bash
curl -X POST http://localhost:5000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...],
    "schema": "customer_data_v1"
  }'
```

## Resource Requirements

- Minimum 16GB RAM total
- 8+ CPU cores
- Fast storage for data processing
- Network capacity for distributed processing

## Integration

- Connects with AI services for data preprocessing
- Supports vector store data preparation
- Integrates with monitoring stack
- Part of the global service mesh
