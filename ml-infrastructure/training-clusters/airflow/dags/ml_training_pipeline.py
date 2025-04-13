"""
ML Training Pipeline DAG
------------------------
A complete machine learning workflow that demonstrates integration with the ML infrastructure:
- Data extraction from vector databases
- Data preprocessing
- Model training on distributed PyTorch
- Model evaluation
- Model deployment to inference servers
- Logging results to MLflow
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

import requests
import numpy as np
import pandas as pd
import mlflow


# Default arguments for the DAG
default_args = {
    'owner': 'ml-engineer',
    'depends_on_past': False,
    'email': ['ml-alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['ml', 'training', 'production'],
)

# Define paths and variables
BASE_PATH = "/workspace"
MODEL_NAME = "text_classification_model"
MODEL_VERSION = "1.0.0"
VECTOR_DB_TYPE = "milvus"  # or "qdrant"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
OUTPUT_MODEL_PATH = f"{BASE_PATH}/models/{MODEL_NAME}/{MODEL_VERSION}"


# Task 1: Extract data from vector database
def extract_data_from_vector_db(**kwargs):
    """Extract training data from vector database"""
    import pymilvus
    from pymilvus import connections, utility
    
    # Connect to Milvus
    connections.connect(
        alias="default",
        host="ml-vector-milvus" if VECTOR_DB_TYPE == "milvus" else "ml-vector-qdrant",
        port="19530" if VECTOR_DB_TYPE == "milvus" else "6333"
    )
    
    collection_name = "training_embeddings"
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        # Get collection and load it
        collection = pymilvus.Collection(collection_name)
        collection.load()
        
        # Search for training data
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.query(
            expr="dataset == 'training'",
            output_fields=["id", "embedding", "label", "text"]
        )
        
        # Save to disk for preprocessing
        data_path = f"{BASE_PATH}/data"
        os.makedirs(data_path, exist_ok=True)
        
        # Convert to pandas DataFrame and save
        df = pd.DataFrame(results)
        df.to_parquet(f"{data_path}/training_data.parquet")
        
        print(f"Extracted {len(df)} records from vector database")
        
        # Close connection
        connections.disconnect("default")
        
        return f"{data_path}/training_data.parquet"
    else:
        raise ValueError(f"Collection {collection_name} does not exist")


extract_data_task = PythonOperator(
    task_id='extract_data_from_vector_db',
    python_callable=extract_data_from_vector_db,
    dag=dag,
)


# Task 2: Preprocess the data
def preprocess_data(**kwargs):
    """Preprocess the data for model training"""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Get the data path from the previous task
    ti = kwargs['ti']
    data_path = ti.xcom_pull(task_ids='extract_data_from_vector_db')
    
    # Load the data
    df = pd.read_parquet(data_path)
    
    # Convert labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])
    
    # Get embeddings
    embeddings = np.vstack(df['embedding'].apply(lambda x: np.array(x)).values)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Create datasets directory
    datasets_dir = f"{BASE_PATH}/data/processed"
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Save tensors
    torch.save(X_train_tensor, f"{datasets_dir}/X_train.pt")
    torch.save(y_train_tensor, f"{datasets_dir}/y_train.pt")
    torch.save(X_val_tensor, f"{datasets_dir}/X_val.pt")
    torch.save(y_val_tensor, f"{datasets_dir}/y_val.pt")
    
    # Save label encoder
    with open(f"{datasets_dir}/label_encoder.json", 'w') as f:
        json.dump(
            {
                'classes': label_encoder.classes_.tolist(),
                'embedding_dim': embeddings.shape[1],
                'num_classes': len(label_encoder.classes_)
            },
            f
        )
    
    print(f"Preprocessed data saved to {datasets_dir}")
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    return datasets_dir


preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)


# Task 3: Train the model on PyTorch distributed cluster
def generate_training_script(**kwargs):
    """Generate the PyTorch distributed training script"""
    ti = kwargs['ti']
    datasets_dir = ti.xcom_pull(task_ids='preprocess_data')
    
    # Script path
    script_path = f"{BASE_PATH}/scripts"
    os.makedirs(script_path, exist_ok=True)
    
    # Create the training script
    training_script = f"""#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
import mlflow
import mlflow.pytorch

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://ml-mlflow:5000")

# Define the model
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(rank, world_size):
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'ml-pytorch-master'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Load metadata
    with open("{datasets_dir}/label_encoder.json", 'r') as f:
        metadata = json.load(f)
    
    input_dim = metadata['embedding_dim']
    hidden_dim = 256
    output_dim = metadata['num_classes']
    
    # Load data
    X_train = torch.load("{datasets_dir}/X_train.pt")
    y_train = torch.load("{datasets_dir}/y_train.pt")
    X_val = torch.load("{datasets_dir}/X_val.pt")
    y_val = torch.load("{datasets_dir}/y_val.pt")
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size={BATCH_SIZE}, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size={BATCH_SIZE}, sampler=val_sampler
    )
    
    # Initialize model, loss function, and optimizer
    model = TextClassifier(input_dim, hidden_dim, output_dim).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr={LEARNING_RATE})
    
    # Start MLflow run if master process
    if rank == 0:
        mlflow.start_run(run_name="{MODEL_NAME}_{MODEL_VERSION}")
        mlflow.log_params({{
            "model_type": "TextClassifier",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "batch_size": {BATCH_SIZE},
            "epochs": {EPOCHS},
            "learning_rate": {LEARNING_RATE}
        }})
    
    # Training loop
    for epoch in range({EPOCHS}):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(rank), targets.to(rank)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Log metrics if master process
        if rank == 0:
            print(f"Epoch {{epoch+1}}/{{{EPOCHS}}}, "
                  f"Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%, "
                  f"Val Loss: {{val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%")
            
            mlflow.log_metrics({{
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }}, step=epoch)
    
    # Save the model if master process
    if rank == 0:
        # Save PyTorch model
        os.makedirs("{OUTPUT_MODEL_PATH}", exist_ok=True)
        torch.save(model.module.state_dict(), "{OUTPUT_MODEL_PATH}/model.pt")
        
        # Save model metadata
        with open("{OUTPUT_MODEL_PATH}/metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model.module, "model")
        mlflow.end_run()
        
        print(f"Model saved to {OUTPUT_MODEL_PATH}")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    # Get world size from env
    world_size = int(os.environ.get("WORLD_SIZE", 3))
    
    # Launch processes
    mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)
"""
    
    # Write the script to a file
    script_file = f"{script_path}/train_model.py"
    with open(script_file, 'w') as f:
        f.write(training_script)
    
    # Make the script executable
    os.chmod(script_file, 0o755)
    
    return script_file


generate_training_script_task = PythonOperator(
    task_id='generate_training_script',
    python_callable=generate_training_script,
    dag=dag,
)


# Task 4: Launch training job on PyTorch cluster
launch_training_job_task = BashOperator(
    task_id='launch_training_job',
    bash_command="""
        ssh ml-pytorch-master "cd {{ ti.xcom_pull(task_ids='generate_training_script').rsplit('/', 1)[0] }} && \
        CUDA_VISIBLE_DEVICES=0 python {{ ti.xcom_pull(task_ids='generate_training_script') }}"
    """,
    dag=dag,
)


# Task 5: Evaluate the trained model
def evaluate_model(**kwargs):
    """Evaluate the trained model and log metrics"""
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import json
    import mlflow
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ml-mlflow:5000")
    
    # Define model class (same as in training script)
    class TextClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(TextClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Get data directory from previous task
    ti = kwargs['ti']
    datasets_dir = ti.xcom_pull(task_ids='preprocess_data')
    
    # Load validation data
    X_val = torch.load(f"{datasets_dir}/X_val.pt")
    y_val = torch.load(f"{datasets_dir}/y_val.pt")
    
    # Load model metadata
    with open(f"{OUTPUT_MODEL_PATH}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    input_dim = metadata['embedding_dim']
    hidden_dim = 256
    output_dim = metadata['num_classes']
    
    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(input_dim, hidden_dim, output_dim).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(f"{OUTPUT_MODEL_PATH}/model.pt"))
    model.eval()
    
    # Evaluate model
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        
        outputs = model(X_val)
        _, predicted = torch.max(outputs, 1)
        
        # Convert to numpy for sklearn metrics
        y_true = y_val.cpu().numpy()
        y_pred = predicted.cpu().numpy()
    
    # Calculate metrics
    class_names = metadata.get('classes', [f"Class_{i}" for i in range(output_dim)])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Log metrics to MLflow
    with mlflow.start_run(run_name=f"{MODEL_NAME}_{MODEL_VERSION}_evaluation"):
        # Log overall metrics
        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "macro_precision": report['macro avg']['precision'],
            "macro_recall": report['macro avg']['recall'],
            "macro_f1": report['macro avg']['f1-score'],
            "weighted_f1": report['weighted avg']['f1-score']
        })
        
        # Log per-class metrics
        for class_name in class_names:
            if class_name in report:
                mlflow.log_metrics({
                    f"{class_name}_precision": report[class_name]['precision'],
                    f"{class_name}_recall": report[class_name]['recall'],
                    f"{class_name}_f1": report[class_name]['f1-score']
                })
        
        # Log confusion matrix as artifact
        np.savetxt(f"{OUTPUT_MODEL_PATH}/confusion_matrix.csv", conf_matrix, delimiter=",")
        mlflow.log_artifact(f"{OUTPUT_MODEL_PATH}/confusion_matrix.csv")
        
        # Save detailed evaluation results
        with open(f"{OUTPUT_MODEL_PATH}/evaluation_results.json", 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(f"{OUTPUT_MODEL_PATH}/evaluation_results.json")
    
    print(f"Model evaluation complete. Accuracy: {report['accuracy']:.4f}")
    return f"{OUTPUT_MODEL_PATH}/evaluation_results.json"


evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)


# Task 6: Convert model to ONNX format for Triton
def convert_to_onnx(**kwargs):
    """Convert PyTorch model to ONNX format for deployment to Triton Inference Server"""
    import torch
    import torch.nn as nn
    import json
    import os
    
    # Define model class (same as before)
    class TextClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(TextClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Load model metadata
    with open(f"{OUTPUT_MODEL_PATH}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    input_dim = metadata['embedding_dim']
    hidden_dim = 256
    output_dim = metadata['num_classes']
    
    # Initialize model and load weights
    device = torch.device("cpu")  # ONNX export needs to be on CPU
    model = TextClassifier(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(f"{OUTPUT_MODEL_PATH}/model.pt", map_location=device))
    model.eval()
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_dim, device=device)
    
    # Export model to ONNX
    onnx_model_path = f"{OUTPUT_MODEL_PATH}/model.onnx"
    torch.onnx.export(
        model,                     # model being run
        dummy_input,               # model input
        onnx_model_path,           # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},   # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX format at {onnx_model_path}")
    return onnx_model_path


convert_to_onnx_task = PythonOperator(
    task_id='convert_to_onnx',
    python_callable=convert_to_onnx,
    dag=dag,
)


# Task 7: Deploy model to Triton Inference Server
def create_triton_model_config(**kwargs):
    """Create Triton model repository structure and configuration files"""
    import json
    import os
    import shutil
    
    # Get ONNX model path from previous task
    ti = kwargs['ti']
    onnx_model_path = ti.xcom_pull(task_ids='convert_to_onnx')
    
    # Load model metadata
    with open(f"{OUTPUT_MODEL_PATH}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Define paths for Triton model repository
    triton_model_repo = "/models"
    model_version = "1"
    
    # Create directory structure
    model_dir = f"{triton_model_repo}/{MODEL_NAME}"
    version_dir = f"{model_dir}/{model_version}"
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy ONNX model to version directory
    shutil.copy(onnx_model_path, f"{version_dir}/model.onnx")
    
    # Create config.pbtxt for the model
    config = f"""
name: "{MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {metadata['embedding_dim']} ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {metadata['num_classes']} ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 1, 4, 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 50000
}}
instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""
    
    # Write config file
    with open(f"{model_dir}/config.pbtxt", 'w') as f:
        f.write(config)
    
    # Create labels file for class names
    if 'classes' in metadata:
        with open(f"{model_dir}/labels.txt", 'w') as f:
            for class_name in metadata['classes']:
                f.write(f"{class_name}\n")
    
    # Return the model repository directory
    print(f"Triton model config created at {model_dir}")
    return model_dir


create_triton_config_task = PythonOperator(
    task_id='create_triton_config',
    python_callable=create_triton_model_config,
    dag=dag,
)


# Task 8: Reload Triton models
reload_triton_models_task = SimpleHttpOperator(
    task_id='reload_triton_models',
    http_conn_id='triton_inference_server',
    endpoint='/v2/repository/index',
    method='POST',
    data=json.dumps({"action": "RELOAD"}),
    headers={"Content-Type": "application/json"},
    response_check=lambda response: response.status_code == 200,
    dag=dag,
)


# Set up task dependencies
extract_data_task >> preprocess_data_task >> generate_training_script_task >> launch_training_job_task
launch_training_job_task >> evaluate_model_task >> convert_to_onnx_task
convert_to_onnx_task >> create_triton_config_task >> reload_triton_models_task
