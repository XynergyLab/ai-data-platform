# ML Infrastructure

This directory contains the machine learning infrastructure components for training models and serving them in production. The infrastructure is built using containerized services managed with Podman Compose.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Starting the Infrastructure](#starting-the-infrastructure)
  - [Stopping the Infrastructure](#stopping-the-infrastructure)
- [Using Model Serving Endpoints](#using-model-serving-endpoints)
  - [Triton Inference Server](#triton-inference-server)
  - [TensorFlow Serving](#tensorflow-serving)
  - [TorchServe](#torchserve)
- [Working with Training Clusters](#working-with-training-clusters)
  - [Submitting Training Jobs](#submitting-training-jobs)
  - [Tracking Experiments with MLflow](#tracking-experiments-with-mlflow)
  - [Using Distributed Training](#using-distributed-training)
- [Monitoring and Observability](#monitoring-and-observability)
  - [Grafana Dashboards](#grafana-dashboards)
  - [Prometheus Metrics](#prometheus-metrics)
  - [Logs](#logs)
- [Maintenance](#maintenance)
  - [Upgrading Components](#upgrading-components)
  - [Backup and Restore](#backup-and-restore)
  - [Health Checks](#health-checks)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Diagnostic Tools](#diagnostic-tools)
  - [Support](#support)

## Overview

This ML infrastructure provides end-to-end capabilities for model development, training, and deployment. It includes:

- Model serving platforms for inference in production
- Distributed training clusters for model development
- Experiment tracking and model registry
- Monitoring and observability
- Vector databases for efficient similarity search

The infrastructure uses containerization to ensure consistency across environments and scalability for production workloads.

## Components

The ML infrastructure consists of two main components:

### Model Serving

Located in the `model-serving` directory, this component provides:

- **Triton Inference Server**: High-performance model serving for multiple frameworks
- **TensorFlow Serving**: Optimized for TensorFlow models
- **TorchServe**: PyTorch model deployment
- **Model Gateway**: API gateway for model endpoints with load balancing
- **Metrics Exporter**: Prometheus metrics collection for model performance

### Training Clusters

Located in the `training-clusters` directory, this component provides:

- **PyTorch Distributed**: Multi-node training with PyTorch
- **Ray**: Distributed computing framework for ML
- **MLflow**: Experiment tracking and model registry
- **Kubeflow Pipelines**: ML workflow orchestration
- **Airflow**: Job scheduling and pipeline automation
- **MinIO**: Object storage for training artifacts

## Getting Started

### Prerequisites

Before starting the ML infrastructure, ensure you have:

1. Podman installed (version 3.0 or later)
2. At least 32GB of RAM available
3. GPU with CUDA support (recommended)
4. At least 100GB of free disk space
5. Network access to container registries

### Starting the Infrastructure

To start the entire ML infrastructure:

```powershell
# Navigate to the Podman_Compose directory
cd C:\Users\drew\Documents\Podman_Compose

# Start the model serving components
podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml up -d

# Start the training cluster components
podman-compose -f ml-infrastructure/training-clusters/podman-compose.yaml up -d
```

You can also start individual components:

```powershell
# Start only the model serving infrastructure
podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml up -d

# Start specific services
podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml up -d triton-inference-server model-gateway
```

### Stopping the Infrastructure

To stop the infrastructure:

```powershell
# Stop the model serving components
podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml down

# Stop the training cluster components
podman-compose -f ml-infrastructure/training-clusters/podman-compose.yaml down

# Stop and remove volumes (caution: this will delete all data)
podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml down -v
```

## Using Model Serving Endpoints

The model serving infrastructure exposes multiple endpoints for model inference.

### Triton Inference Server

Triton Inference Server is available at `http://localhost:8000`.

#### Example: Making an inference request

```python
import requests
import json
import numpy as np

# Prepare the input data
input_data = np.random.rand(1, 768).astype(np.float32).tolist()

# Create the request payload
payload = {
    "inputs": [
        {
            "name": "input",
            "shape": [1, 768],
            "datatype": "FP32",
            "data": input_data
        }
    ]
}

# Send the request to Triton
url = "http://localhost:8000/v2/models/text_classification_model/infer"
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Parse the response
result = response.json()
print("Prediction:", result)
```

### TensorFlow Serving

TensorFlow Serving is available at `http://localhost:8501`.

#### Example: Making an inference request

```python
import requests
import json
import numpy as np

# Prepare input data
input_data = np.random.rand(1, 10).astype(np.float32).tolist()

# Create request payload
payload = {
    "instances": input_data
}

# Send request
url = "http://localhost:8501/v1/models/default:predict"
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Parse response
result = response.json()
print("Prediction:", result)
```

### TorchServe

TorchServe is available at `http://localhost:8080`.

#### Example: Making an inference request

```python
import requests
import json

# Prepare input text
input_text = "This is a sample text for classification."

# Send request
url = "http://localhost:8080/predictions/text_classifier"
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps({"text": input_text}))

# Parse response
result = response.json()
print("Prediction:", result)
```

## Working with Training Clusters

### Submitting Training Jobs

Training jobs can be submitted through Airflow DAGs or Kubeflow Pipelines.

#### Using Airflow

1. Access the Airflow UI at `http://localhost:8080`
2. Log in with the default credentials (username: `admin`, password: `admin`)
3. Navigate to the DAGs page
4. Enable and trigger the `ml_training_pipeline` DAG

#### Using Kubeflow Pipelines

1. Access the Kubeflow Pipelines UI at `http://localhost:8889`
2. Upload a new pipeline or use an existing template
3. Create a run from the pipeline
4. Monitor the execution in the UI

### Tracking Experiments with MLflow

MLflow is used for experiment tracking and model registry.

1. Access MLflow UI at `http://localhost:5000`
2. View experiments, runs, and model artifacts
3. Compare different training runs

#### Example: Logging to MLflow from Python

```python
import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Start a new run
with mlflow.start_run(run_name="my_experiment"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 64)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Using Distributed Training

The PyTorch distributed training cluster is configured for multi-node training.

#### Example: Running a distributed training job

```python
import torch.distributed as dist
import torch.multiprocessing as mp

def train(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Training code here
    # ...
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 3  # Number of nodes
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## Monitoring and Observability

### Grafana Dashboards

Grafana dashboards are available for monitoring the ML infrastructure:

1. Access Grafana at `http://localhost:3000`
2. Log in with default credentials (username: `admin`, password: `admin`)
3. Navigate to the dashboards page
4. Select the ML dashboards from the "ML Models" folder

Key dashboards include:
- Model Inference Dashboard
- Model Performance Dashboard
- Training Cluster Dashboard
- GPU Utilization Dashboard

### Prometheus Metrics

Prometheus collects metrics from all ML components:

1. Access Prometheus at `http://localhost:9090`
2. Use PromQL to query metrics
3. View targets and their status

Important metrics to monitor:
- `nv_inference_request_success`: Successful inference requests
- `nv_inference_request_failure`: Failed inference requests
- `nv_inference_request_duration_ms`: Inference latency
- `nv_gpu_utilization`: GPU utilization

### Logs

Logs are collected and centralized with Loki:

1. Access logs through Grafana
2. Navigate to the Explore page
3. Select Loki as the data source
4. Query logs by container or service

Example query: `{container="ml-triton-inference"}`

## Maintenance

### Upgrading Components

To upgrade a component:

1. Update the image tag in the corresponding podman-compose.yaml file
2. Stop the component:
   ```powershell
   podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml stop triton-inference-server
   ```
3. Start the component with the new version:
   ```powershell
   podman-compose -f ml-infrastructure/model-serving/podman-compose.yaml up -d triton-inference-server
   ```

### Backup and Restore

Backup important data regularly:

```powershell
# Backup model artifacts
podman run --rm -v ml_model_data:/source -v C:\Backups\ml-models:/destination alpine sh -c "cp -r /source/* /destination/"

# Backup MLflow data
podman run --rm -v mlflow_data:/source -v C:\Backups\mlflow:/destination alpine sh -c "cp -r /source/* /destination/"
```

To restore from backup:

```powershell
# Restore model artifacts
podman run --rm -v ml_model_data:/destination -v C:\Backups\ml-models:/source alpine sh -c "cp -r /source/* /destination/"
```

### Health Checks

Run the validation script to check the health of the infrastructure:

```powershell
.\ml-infrastructure\validate-ml-infrastructure.ps1
```

## Troubleshooting

### Common Issues

#### Containers Not Starting

**Issue**: Containers fail to start or exit immediately after starting.

**Solution**:
1. Check container logs: `podman logs <container_name>`
2. Verify resource availability (memory, disk, ports)
3. Ensure required volumes exist
4. Check for network conflicts

#### Model Serving Errors

**Issue**: Model serving endpoints return errors.

**Solution**:
1. Verify model files exist in the correct location
2. Check model configuration files
3. Review the Triton/TorchServe logs for specific errors
4. Make sure GPU is available if required by the model

#### Training Job Failures

**Issue**: Training jobs fail or get stuck.

**Solution**:
1. Check Airflow logs for error messages
2. Verify GPU availability
3. Ensure vector databases are accessible
4. Check for resource constraints (memory, disk)

### Diagnostic Tools

Use these tools to diagnose issues:

```powershell
# Check container status
podman ps -a

# Inspect container details
podman inspect <container_name>

# View container logs
podman logs <container_name>

# Run validation script
.\ml-infrastructure\validate-ml-infrastructure.ps1
```

### Support

For additional support, contact the ML Infrastructure team at ml-infra-support@example.com or open an issue in the internal JIRA project.

---

## Contributing

To contribute to the ML infrastructure, please follow the standard git workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Ensure your changes pass the validation script before submitting.

