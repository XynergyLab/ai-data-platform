# Minikube Kubernetes Configuration for AI and Data Processing Platform

This document provides instructions for deploying our AI and data processing platform on a local Kubernetes cluster using Minikube, complementing our existing Podman-based infrastructure.

## Table of Contents

- [Overview](#overview)
- [Minikube Setup](#minikube-setup)
- [Cluster Configuration](#cluster-configuration)
- [Deploying AI Services](#deploying-ai-services)
- [Deploying Data Processing Services](#deploying-data-processing-services)
- [Deploying Vector Stores](#deploying-vector-stores)
- [Deploying Monitoring Stack](#deploying-monitoring-stack)
- [Integration with Podman Environment](#integration-with-podman-environment)
- [Troubleshooting](#troubleshooting)

## Overview

Minikube provides a lightweight Kubernetes implementation that runs on your local machine, enabling the development and testing of Kubernetes applications without requiring a full cluster. This setup extends our Podman Compose environment by adding orchestration capabilities, scaling, and advanced deployment features available in Kubernetes.

### Benefits of Kubernetes for AI Workflows

- **Scaling**: Dynamically scale AI model serving based on demand
- **Resource Management**: Precise control over CPU, memory, and GPU allocations
- **Service Discovery**: Automatic service registration and DNS resolution
- **High Availability**: Built-in support for redundancy and failover
- **Declarative Configuration**: Infrastructure as code for reproducible deployments

## Minikube Setup

### Prerequisites

- 16+ GB RAM
- 4+ CPU cores
- 50+ GB free disk space
- Virtualization support enabled in BIOS
- Docker, Podman, or Hypervisor (depends on driver choice)

### Installation

1. Install Minikube:

   ```bash
   # Windows
   choco install minikube

   # macOS
   brew install minikube

   # Linux
   curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
   sudo install minikube-linux-amd64 /usr/local/bin/minikube
   ```

2. Install kubectl:

   ```bash
   # Windows
   choco install kubernetes-cli

   # macOS
   brew install kubectl

   # Linux
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   ```

### Starting Minikube

Start a Minikube cluster with sufficient resources for AI workloads:

```bash
minikube start --cpus=4 --memory=8g --disk-size=50g --driver=docker --kubernetes-version=stable
```

#### Advanced Configuration Options

For AI workloads with GPU support:

```bash
minikube start --cpus=4 --memory=16g --disk-size=100g --driver=docker --kubernetes-version=stable --addons=nvidia-gpu-device-plugin --gpus=all
```

For a multi-node setup:

```bash
minikube start --nodes=3 --cpus=4 --memory=8g --disk-size=50g --driver=docker --kubernetes-version=stable
```

### Enabling Required Addons

```bash
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable dashboard
minikube addons enable storage-provisioner
```

## Cluster Configuration

### Creating Namespaces

Create namespaces to organize our services:

```bash
kubectl create namespace ai-services
kubectl create namespace data-processing
kubectl create namespace databases
kubectl create namespace monitoring
```

### Setting Resource Quotas

Set resource limits for each namespace:

```yaml
# resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ai-services-quota
  namespace: ai-services
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: data-processing-quota
  namespace: data-processing
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
```

Apply the quotas:

```bash
kubectl apply -f resource-quotas.yaml
```

### Setting Up Storage Classes

Create a storage class for our persistent volumes:

```yaml
# storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ai-platform-storage
provisioner: k8s.io/minikube-hostpath
reclaimPolicy: Retain
volumeBindingMode: Immediate
```

Apply the storage class:

```bash
kubectl apply -f storage-class.yaml
```

## Deploying AI Services

### LLM Inference Services

Create Kubernetes manifests for LLM inference services:

```yaml
# llm-inference.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: ai-services
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        env:
        - name: OLLAMA_HOST
          value: "0.0.0.0"
        - name: OLLAMA_MODELS_PATH
          value: "/root/.ollama"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: ollama-models
          mountPath: /root/.ollama
        - name: ollama-config
          mountPath: /etc/ollama
      volumes:
      - name: ollama-models
        persistentVolumeClaim:
          claimName: ollama-models-pvc
      - name: ollama-config
        configMap:
          name: ollama-config
---
apiVersion: v1
kind: Service
metadata:
  name: ollama
  namespace: ai-services
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-models-pvc
  namespace: ai-services
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ollama-config
  namespace: ai-services
data:
  inference.json: |
    {
      "default_model": "llama2",
      "gpu_layers": 35,
      "context_size": 4096,
      "num_ctx": 4096,
      "num_batch": 512,
      "num_thread": 8,
      "rope_scaling": 1.0
    }
```

Deploy the LLM inference services:

```bash
kubectl apply -f llm-inference.yaml
```

### Embedding Services

```yaml
# embedding-services.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-embedding
  namespace: ai-services
spec:
  replicas: 1
  selector:
    matchLabels:
      app: text-embedding
  template:
    metadata:
      labels:
        app: text-embedding
    spec:
      containers:
      - name: text-embedding
        image: ghcr.io/huggingface/text-embeddings-inference:latest
        ports:
        - containerPort: 80
        env:
        - name: MODEL_ID
          value: "BAAI/bge-large-en-v1.5"
        - name: MAX_BATCH_SIZE
          value: "32"
        - name: MAX_CONCURRENT_REQUESTS
          value: "256"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: embedding-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: text-embedding
  namespace: ai-services
spec:
  selector:
    app: text-embedding
  ports:
  - port: 8081
    targetPort: 80
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: embedding-models-pvc
  namespace: ai-services
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 5Gi
```

Deploy the embedding services:

```bash
kubectl apply -f embedding-services.yaml
```

### Multimodal Services

```yaml
# multimodal-services.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-processor
  namespace: ai-services
spec:
  replicas: 1
  selector:
    matchLabels:
      app: image-processor
  template:
    metadata:
      labels:
        app: image-processor
    spec:
      containers:
      - name: image-processor
        image: pytorch/pytorch:latest
        ports:
        - containerPort: 8090
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: MAX_BATCH_SIZE
          value: "16"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
        volumeMounts:
        - name: models
          mountPath: /models
        - name: data
          mountPath: /data
        - name: config
          mountPath: /app/config
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: multimodal-models-pvc
      - name: data
        persistentVolumeClaim:
          claimName: multimodal-data-pvc
      - name: config
        configMap:
          name: multimodal-config
---
apiVersion: v1
kind: Service
metadata:
  name: image-processor
  namespace: ai-services
spec:
  selector:
    app: image-processor
  ports:
  - port: 8090
    targetPort: 8090
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: multimodal-models-pvc
  namespace: ai-services
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: multimodal-data-pvc
  namespace: ai-services
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: multimodal-config
  namespace: ai-services
data:
  processing.json: |
    {
      "image_processing": {
        "supported_models": {
          "detection": ["yolov5", "faster_rcnn"],
          "classification": ["resnet50", "efficientnet"]
        },
        "preprocessing": {
          "resize_mode": "preserve_aspect_ratio",
          "target_size": [640, 640],
          "normalize": true
        }
      }
    }
```

Deploy the multimodal services:

```bash
kubectl apply -f multimodal-services.yaml
```

## Deploying Vector Stores

### Milvus Vector Database

```yaml
# vector-stores.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: milvus
  namespace: databases
spec:
  serviceName: "milvus"
  replicas: 1
  selector:
    matchLabels:
      app: milvus
  template:
    metadata:
      labels:
        app: milvus
    spec:
      containers:
      - name: milvus
        image: milvusdb/milvus:latest
        ports:
        - containerPort: 19530
        - containerPort: 9091
        env:
        - name: ETCD_HOST
          value: "etcd"
        - name: ETCD_PORT
          value: "2379"
        - name: MINIO_ADDRESS
          value: "minio:9000"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
        volumeMounts:
        - name: milvus-data
          mountPath: /var/lib/milvus
        - name: milvus-config
          mountPath: /milvus/configs
      volumes:
      - name: milvus-config
        configMap:
          name: milvus-config
  volumeClaimTemplates:
  - metadata:
      name: milvus-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ai-platform-storage
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: milvus
  namespace: databases
spec:
  selector:
    app: milvus
  ports:
  - port: 19530
    targetPort: 19530
    name: milvus
  - port: 9091
    targetPort: 9091
    name: metrics
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: etcd
  namespace: databases
spec:
  replicas: 1
  selector:
    matchLabels:
      app: etcd
  template:
    metadata:
      labels:
        app: etcd
    spec:
      containers:
      - name: etcd
        image: quay.io/coreos/etcd:v3.5.5
        ports:
        - containerPort: 2379
        env:
        - name: ETCD_AUTO_COMPACTION_MODE
          value: "revision"
        - name: ETCD_AUTO_COMPACTION_RETENTION
          value: "1000"
        - name: ETCD_QUOTA_BACKEND_BYTES
          value: "4294967296"
        command:
        - etcd
        - --advertise-client-urls=http://etcd:2379
        - --listen-client-urls=http://0.0.0.0:2379
        - --data-dir=/etcd
        volumeMounts:
        - name: etcd-data
          mountPath: /etcd
      volumes:
      - name: etcd-data
        persistentVolumeClaim:
          claimName: etcd-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: etcd
  namespace: databases
spec:
  selector:
    app: etcd
  ports:
  - port: 2379
    targetPort: 2379
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: etcd-data-pvc
  namespace: databases
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 5Gi
```

Deploy the vector stores:

```bash
kubectl apply -f vector-stores.yaml
```

## Deploying Data Processing Services

### Apache Airflow for ETL

```yaml
# data-processing.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airflow
  namespace: data-processing
spec:
  replicas: 1
  selector:
    matchLabels:
      app: airflow
  template:
    metadata:
      labels:
        app: airflow
    spec:
      containers:
      - name: airflow
        image: apache/airflow:latest
        ports:
        - containerPort: 8080
        env:
        - name: AIRFLOW__CORE__EXECUTOR
          value: "LocalExecutor"
        - name: AIRFLOW__CORE__SQL_ALCHEMY_CONN
          value: "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
        - name: AIRFLOW__CORE__LOAD_EXAMPLES
          value: "false"
        resources:
          requests:
            memory: "2Gi"
            cpu: "0.5"
          limits:
            memory: "4Gi"
            cpu: "1"
        volumeMounts:
        - name: dags
          mountPath: /opt/airflow/dags
        - name: plugins
          mountPath: /opt/airflow/plugins
        - name: logs
          mountPath: /opt/airflow/logs
      volumes:
      - name: dags
        persistentVolumeClaim:
          claimName: airflow-dags-pvc
      - name: plugins
        persistentVolumeClaim:
          claimName: airflow-plugins-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: airflow-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: airflow
  namespace: data-processing
spec:
  selector:
    app: airflow
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: data-processing
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          value: "airflow"
        - name: POSTGRES_PASSWORD
          value: "airflow"
        - name: POSTGRES_DB
          value: "airflow"
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.2"
          limits:
            memory: "1Gi"
            cpu: "0.5"
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-data
        persistentVolumeClaim:
          claimName: postgres-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: data-processing
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-dags-pvc
  namespace: data-processing
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-plugins-pvc
  namespace: data-processing
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-logs-pvc
  namespace: data-processing
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 2Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data-pvc
  namespace: data-processing
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 5Gi
```

### Apache Spark for Data Processing

```yaml
# spark-cluster.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-master
  namespace: data-processing
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spark-master
  template:
    metadata:
      labels:
        app: spark-master
    spec:
      containers:
      - name: spark-master
        image: bitnami/spark:latest
        ports:
        - containerPort: 8080
        - containerPort: 7077
        env:
        - name: SPARK_MODE
          value: "master"
        - name: SPARK_RPC_AUTHENTICATION_ENABLED
          value: "no"
        resources:
          requests:
            memory: "2Gi"
            cpu: "0.5"
          limits:
            memory: "4Gi"
            cpu: "1"
        volumeMounts:
        - name: spark-data
          mountPath: /opt/spark/work
      volumes:
      - name: spark-data
        persistentVolumeClaim:
          claimName: spark-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: spark-master
  namespace: data-processing
spec:
  selector:
    app: spark-master
  ports:
  - port: 8181
    targetPort: 8080
    name: ui
  - port: 7077
    targetPort: 7077
    name: spark
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-worker
  namespace: data-processing
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spark-worker
  template:
    metadata:
      labels:
        app: spark-worker
    spec:
      containers:
      - name: spark-worker
        image: bitnami/spark:latest
        env:
        - name: SPARK_MODE
          value: "worker"
        - name: SPARK_MASTER_URL
          value: "spark://spark-master:7077"
        - name: SPARK_WORKER_MEMORY
          value: "4G"
        - name: SPARK_WORKER_CORES
          value: "2"
        resources:
          requests:
            memory: "2Gi"
            cpu: "0.5"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: spark-worker-data
          mountPath: /opt/spark/work
      volumes:
      - name: spark-worker-data
        persistentVolumeClaim:
          claimName: spark-worker-data-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: spark-data-pvc
  namespace: data-processing
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: spark-worker-data-pvc
  namespace: data-processing
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 10Gi
```

Deploy the data processing services:

```bash
kubectl apply -f data-processing.yaml
kubectl apply -f spark-cluster.yaml
```

## Deploying Monitoring Stack

### Prometheus and Grafana

```yaml
# monitoring.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-data
          mountPath: /prometheus
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-data
        persistentVolumeClaim:
          claimName: prometheus-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
      
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name

      - job_name: 'node-exporter'
        static_configs:
          - targets: ['node-exporter:9100']

      - job_name: 'ai-services'
        static_configs:
          - targets:
            - 'ollama.ai-services.svc.cluster.local:11434'
            - 'text-embedding.ai-services.svc.cluster.local:8081'
            - 'image-processor.ai-services.svc.cluster.local:8090'

      - job_name: 'data-processing'
        static_configs:
          - targets:
            - 'airflow.data-processing.svc.cluster.local:8080'
            - 'spark-master.data-processing.svc.cluster.local:8181'
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-data-pvc
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 10Gi
```

### Grafana Dashboard

```yaml
# grafana.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin"
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        - name: GF_AUTH_ANONYMOUS_ENABLED
          value: "true"
        - name: GF_AUTH_ANONYMOUS_ORG_ROLE
          value: "Viewer"
        - name: GF_INSTALL_PLUGINS
          value: "grafana-piechart-panel,grafana-worldmap-panel"
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.2"
          limits:
            memory: "1Gi"
            cpu: "0.5"
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-datasources
          mountPath: /etc/grafana/provisioning/datasources
        - name: grafana-dashboards
          mountPath: /etc/grafana/provisioning/dashboards
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-storage-pvc
      - name: grafana-datasources
        configMap:
          name: grafana-datasources
      - name: grafana-dashboards
        configMap:
          name: grafana-dashboards
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-storage-pvc
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: monitoring
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus:9090
      access: proxy
      isDefault: true
    - name: Loki
      type: loki
      url: http://loki:3100
      access: proxy
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
    - name: 'default'
      orgId: 1
      folder: ''
      type: file
      disableDeletion: false
      editable: true
      options:
        path: /var/lib/grafana/dashboards
```

### Loki and Promtail for Log Aggregation

```yaml
# logging.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      containers:
      - name: loki
        image: grafana/loki:latest
        ports:
        - containerPort: 3100
        command:
        - /usr/bin/loki
        - -config.file=/etc/loki/loki-config.yml
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.2"
          limits:
            memory: "1Gi"
            cpu: "0.5"
        volumeMounts:
        - name: loki-config
          mountPath: /etc/loki
        - name: loki-data
          mountPath: /loki
      volumes:
      - name: loki-config
        configMap:
          name: loki-config
      - name: loki-data
        persistentVolumeClaim:
          claimName: loki-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: monitoring
spec:
  selector:
    app: loki
  ports:
  - port: 3100
    targetPort: 3100
  type: ClusterIP
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: monitoring
data:
  loki-config.yml: |
    auth_enabled: false

    server:
      http_listen_port: 3100

    ingester:
      lifecycler:
        address: 127.0.0.1
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1
        final_sleep: 0s
      chunk_idle_period: 5m
      chunk_retain_period: 30s

    schema_config:
      configs:
        - from: 2024-01-01
          store: boltdb
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h

    storage_config:
      boltdb:
        directory: /loki/index

      filesystem:
        directory: /loki/chunks

    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: loki-data-pvc
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ai-platform-storage
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    metadata:
      labels:
        app: promtail
    spec:
      containers:
      - name: promtail
        image: grafana/promtail:latest
        args:
        - -config.file=/etc/promtail/promtail-config.yml
        resources:
          requests:
            memory: "128Mi"
            cpu: "0.1"
          limits:
            memory: "256Mi"
            cpu: "0.2"
        volumeMounts:
        - name: promtail-config
          mountPath: /etc/promtail
        - name: docker-logs
          mountPath: /var/log/containers
          readOnly: true
        - name: pod-logs
          mountPath: /var/log/pods
          readOnly: true
      volumes:
      - name: promtail-config
        configMap:
          name: promtail-config
      - name: docker-logs
        hostPath:
          path: /var/log/containers
      - name: pod-logs
        hostPath:
          path: /var/log/pods
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: monitoring
data:
  promtail-config.yml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    positions:
      filename: /tmp/positions.yaml

    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        pipeline_stages:
          - docker: {}
        relabel_configs:
          - source_labels:
              - __meta_kubernetes_pod_controller_name
            regex: ([0-9a-z-.]+?)(-[0-9a-f]{8,10})?
            action: replace
            target_label: __tmp_controller_name
          - source_labels:
              - __meta_kubernetes_pod_label_app
            regex: (.+)
            action: replace
            target_label: app
          - source_labels:
              - __meta_kubernetes_pod_name
            action: replace
            target_label: pod
          - source_labels:
              - __meta_kubernetes_namespace
            action: replace
            target_label: namespace
          - source_labels:
              - __meta_kubernetes_pod_container_name
            action: replace
            target_label: container
```

### Node Exporter for System Metrics

```yaml
# node-exporter.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:latest
        ports:
        - containerPort: 9100
        args:
        - --path.procfs=/host/proc
        - --path.sysfs=/host/sys
        - --path.rootfs=/host/root
        - --collector.filesystem.ignored-mount-points=^/(dev|proc|sys|var/lib/docker/.+)($|/)
        - --collector.filesystem.ignored-fs-types=^(autofs|binfmt_misc|cgroup|configfs|debugfs|devpts|devtmpfs|fusectl|hugetlbfs|mqueue|overlay|proc|procfs|pstore|rpc_pipefs|securityfs|sysfs|tracefs)$
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: root
          mountPath: /host/root
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      - name: root
        hostPath:
          path: /
---
apiVersion: v1
kind: Service
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    app: node-exporter
  ports:
  - port: 9100
    targetPort: 9100
    name: metrics
  type: ClusterIP
```

Deploy the monitoring stack:

```bash
kubectl apply -f monitoring.yaml
kubectl apply -f grafana.yaml
kubectl apply -f logging.yaml
kubectl apply -f node-exporter.yaml
```

## Integration with Podman Environment

Our Kubernetes infrastructure complements the existing Podman environment, creating a hybrid deployment model where we can leverage the best of both worlds. Here's how to integrate them:

### Sharing Data Between Environments

To share data between Podman containers and Kubernetes:

1. Use shared volumes mounted to the host filesystem:

```yaml
# In Kubernetes manifests
volumes:
- name: shared-data
  hostPath:
    path: /mnt/shared-data
    type: DirectoryOrCreate
```

Then in your Podman configuration:

```yaml
# In Podman compose
volumes:
  - /mnt/shared-data:/data
```

### Network Communication

To enable communication between Podman containers and Kubernetes services:

1. Expose Kubernetes services via NodePort or LoadBalancer:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: exposed-service
  namespace: ai-services
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080
  type: NodePort
```

2. Access the service from Podman containers using the Minikube IP:

```bash
minikube ip  # Get the IP address
```

Then in your Podman container, access the service at `http://<minikube-ip>:30080`

### Shared CI/CD Pipeline

Create a unified deployment pipeline that handles both environments:

```bash
#!/bin/bash

# Deploy core infrastructure to Kubernetes
kubectl apply -f kubernetes/core/

# Deploy supporting services to Podman
cd podman_compose
podman-compose -f support-services.yaml up -d
```

### Resource Allocation Strategy

To prevent resource contention between Minikube and Podman:

1. Set hard limits on Minikube VM:

```bash
minikube config set memory 16384
minikube config set cpus 4
```

2. Configure resource limits in Podman:

```yaml
services:
  heavy-service:
    image: myapp
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Troubleshooting

### Common Issues and Solutions

#### Insufficient Resources

**Symptom**: Pods fail to schedule with "Insufficient memory" or "Insufficient CPU" errors.

**Solution**:
```bash
# Increase minikube resources
minikube stop
minikube config set memory 16384
minikube config set cpus 8
minikube start
```

#### Image Pull Failures

**Symptom**: "ImagePullBackOff" errors in pod status.

**Solution**:
```bash
# Pre-load images into minikube
minikube cache add ollama/ollama:latest
minikube cache add ghcr.io/huggingface/text-embeddings-inference:latest

# Or use a local registry
minikube addons enable registry
```

#### Persistent Volume Claims Not Binding

**Symptom**: PVCs stay in "Pending" state.

**Solution**:
```bash
# Check storage provisioner
kubectl get pods -n kube-system

# Enable storage provisioner if not running
minikube addons enable storage-provisioner
```

#### Networking Issues

**Symptom**: Services can't communicate with each other.

**Solution**:
```bash
# Check DNS resolution
kubectl run -it --rm debug --image=alpine -- sh
# Inside the container
nslookup ollama.ai-services.svc.cluster.local

# Check service endpoints
kubectl get endpoints -n ai-services
```

#### GPU Issues

**Symptom**: Pods requesting GPU resources can't schedule.

**Solution**:
```bash
# Verify GPU addon is enabled
minikube addons list | grep nvidia

# Enable GPU support
minikube stop
minikube start --driver=docker --gpus=all --addons=nvidia-gpu-device-plugin

# Verify GPU devices are available
kubectl describe node minikube | grep nvidia.com/gpu
```

### Debugging Tools

#### Pod and Container Logs

```bash
# Get logs for a specific pod
kubectl logs -n ai-services ollama-6d87d6f74b-2xvr9

# Follow logs
kubectl logs -n ai-services -f ollama-6d87d6f74b-2xvr9

# Get logs for a crashed container
kubectl logs -n ai-services ollama-6d87d6f74b-2xvr9 --previous
```

#### Shell Access

```bash
# Get a shell in a running pod
kubectl exec -it -n ai-services ollama-6d87d6f74b-2xvr9 -- bash

# For troubleshooting network issues
kubectl run -it --rm debug --image=nicolaka/netshoot -- bash
```

#### Resource Usage

```bash
# Get resource usage of nodes
kubectl top nodes

# Get resource usage of pods
kubectl top pods -n ai-services
```

#### Dashboard Access

```bash
# Start the dashboard
minikube dashboard

# Get dashboard URL
minikube dashboard --url
```

### Maintenance Tasks

#### Upgrading Minikube

```bash
# Update Minikube to latest version
minikube update-check

# On Windows
choco upgrade minikube

# On macOS
brew upgrade minikube
```

#### Cleaning Up Resources

```bash
# Delete all resources in a namespace
kubectl delete namespace ai-services

# Delete specific resources
kubectl delete -f multimodal-services.yaml

# Clean up minikube cache
minikube cache delete --all
```

#### Backup and Restore

```bash
# Backup Minikube VM
minikube stop
cp -r ~/.minikube ~/.minikube.backup

# Backup Kubernetes resources
kubectl get all --all-namespaces -o yaml > k8s-backup.yaml
```

## Conclusion

This Minikube Kubernetes configuration extends our Podman-based AI and data processing platform, providing advanced orchestration capabilities, scaling, and monitoring. By setting up these services in a local Kubernetes environment, we can develop and test deployments locally before moving to production clusters.

For production deployments, consider migrating from Minikube to a fully managed Kubernetes service like AKS, EKS, or GKE, or setting up a production-grade Kubernetes cluster using tools like kubeadm or Rancher.
