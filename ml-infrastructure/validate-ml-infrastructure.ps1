#Requires -Version 5.1
<#
.SYNOPSIS
    Validates the ML infrastructure components in Podman Compose environment.
.DESCRIPTION
    This script performs comprehensive validation of the ML infrastructure components:
    - Container status checks
    - Network connectivity between components
    - Model serving endpoint validation
    - Monitoring integration validation
    - GPU availability verification
    - Vector database connectivity
.NOTES
    File Name      : validate-ml-infrastructure.ps1
    Author         : ML Infrastructure Team
    Prerequisite   : PowerShell 5.1 or later, Podman
    Version        : 1.0
#>

# Configuration
$modelServingContainers = @(
    "ml-triton-inference",
    "ml-tf-serving",
    "ml-torchserve",
    "ml-model-gateway",
    "ml-metrics-exporter"
)

$trainingClusterContainers = @(
    "ml-mlflow", 
    "ml-minio", 
    "ml-pytorch-master",
    "ml-pytorch-worker-1",
    "ml-pytorch-worker-2",
    "ml-ray-head",
    "ml-ray-worker-1",
    "ml-training-metrics",
    "ml-kubeflow-api",
    "ml-kubeflow-ui",
    "ml-airflow-webserver",
    "ml-airflow-scheduler"
)

$vectorDatabaseContainers = @(
    "vector-store-milvus",
    "vector-store-qdrant",
    "vector-store-minio",
    "vector-store-etcd"
)

$monitoringContainers = @(
    "monitoring-prometheus",
    "monitoring-grafana",
    "monitoring-loki",
    "monitoring-promtail"
)

$endpoints = @{
    "Triton Inference Server" = "http://localhost:8000/v2/health/ready"
    "TensorFlow Serving" = "http://localhost:8501/v1/models/default"
    "TorchServe" = "http://localhost:8080/ping"
    "Model Gateway" = "http://localhost:8888/"
    "MLflow" = "http://localhost:5000/"
    "Kubeflow UI" = "http://localhost:8889/"
    "Airflow" = "http://localhost:8080/health"
    "Minio" = "http://localhost:9010/minio/health/ready"
    "Prometheus" = "http://localhost:9090/-/healthy"
    "Grafana" = "http://localhost:3000/api/health"
}

# Initialize results tracking
$results = @{
    ContainersRunning = @{Status = "Not Checked"; Details = @()}
    NetworkConnectivity = @{Status = "Not Checked"; Details = @()}
    ModelEndpoints = @{Status = "Not Checked"; Details = @()}
    MonitoringIntegration = @{Status = "Not Checked"; Details = @()}
    GpuAvailability = @{Status = "Not Checked"; Details = @()}
    VectorDbConnectivity = @{Status = "Not Checked"; Details = @()}
}

# Helper Functions
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Write-Status {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Component,
        
        [Parameter(Mandatory = $true)]
        [string]$Status,
        
        [Parameter(Mandatory = $false)]
        [string]$Message
    )
    
    $color = switch ($Status) {
        "PASS" { "Green" }
        "FAIL" { "Red" }
        "WARN" { "Yellow" }
        "INFO" { "Cyan" }
        default { "White" }
    }
    
    Write-Host "[$Status]" -ForegroundColor $color -NoNewline
    Write-Host " $Component" -NoNewline
    if ($Message) {
        Write-Host ": $Message"
    } else {
        Write-Host ""
    }
}

# 1. Check if all containers are running
function Test-ContainerStatus {
    Write-Host "`n=== Checking Container Status ===" -ForegroundColor Blue
    
    $allContainers = $modelServingContainers + $trainingClusterContainers + $vectorDatabaseContainers + $monitoringContainers
    $runningContainers = @()
    $notRunningContainers = @()
    
    if (-not (Test-CommandExists podman)) {
        Write-Status -Component "Podman" -Status "FAIL" -Message "Podman command not found"
        $results.ContainersRunning.Status = "FAIL"
        $results.ContainersRunning.Details += "Podman command not found"
        return
    }
    
    try {
        $containerList = podman ps --format "{{.Names}}"
        
        foreach ($container in $allContainers) {
            if ($containerList -contains $container) {
                Write-Status -Component $container -Status "PASS" -Message "Container is running"
                $runningContainers += $container
            } else {
                Write-Status -Component $container -Status "FAIL" -Message "Container is not running"
                $notRunningContainers += $container
            }
        }
        
        if ($notRunningContainers.Count -eq 0) {
            $results.ContainersRunning.Status = "PASS"
            $results.ContainersRunning.Details = "All containers are running"
        } else {
            $results.ContainersRunning.Status = "FAIL"
            $results.ContainersRunning.Details = "The following containers are not running: $($notRunningContainers -join ', ')"
        }
    } catch {
        Write-Status -Component "Container Status Check" -Status "FAIL" -Message "Error: $_"
        $results.ContainersRunning.Status = "FAIL"
        $results.ContainersRunning.Details += "Error checking container status: $_"
    }
}

# 2. Verify network connectivity between components
function Test-NetworkConnectivity {
    Write-Host "`n=== Checking Network Connectivity ===" -ForegroundColor Blue
    
    $networkTests = @(
        @{Source = "ml-pytorch-master"; Destination = "ml-mlflow"; Port = 5000},
        @{Source = "ml-pytorch-master"; Destination = "vector-store-milvus"; Port = 19530},
        @{Source = "ml-pytorch-master"; Destination = "vector-store-qdrant"; Port = 6333},
        @{Source = "ml-triton-inference"; Destination = "ml-model-gateway"; Port = 80},
        @{Source = "ml-metrics-exporter"; Destination = "monitoring-prometheus"; Port = 9090}
    )
    
    $successfulTests = 0
    $failedTests = @()
    
    foreach ($test in $networkTests) {
        try {
            $result = podman exec $test.Source nc -zv $test.Destination $test.Port 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Status -Component "$($test.Source) → $($test.Destination):$($test.Port)" -Status "PASS" -Message "Connection successful"
                $successfulTests++
            } else {
                Write-Status -Component "$($test.Source) → $($test.Destination):$($test.Port)" -Status "FAIL" -Message "Connection failed"
                $failedTests += "$($test.Source) → $($test.Destination):$($test.Port)"
            }
        } catch {
            Write-Status -Component "$($test.Source) → $($test.Destination):$($test.Port)" -Status "FAIL" -Message "Error: $_"
            $failedTests += "$($test.Source) → $($test.Destination):$($test.Port)"
        }
    }
    
    if ($failedTests.Count -eq 0) {
        $results.NetworkConnectivity.Status = "PASS"
        $results.NetworkConnectivity.Details = "All network connectivity tests passed"
    } else {
        $results.NetworkConnectivity.Status = "FAIL"
        $results.NetworkConnectivity.Details = "The following connectivity tests failed: $($failedTests -join ', ')"
    }
}

# 3. Validate access to model serving endpoints
function Test-ModelEndpoints {
    Write-Host "`n=== Checking Model Serving Endpoints ===" -ForegroundColor Blue
    
    $accessibleEndpoints = @()
    $inaccessibleEndpoints = @()
    
    foreach ($endpoint in $endpoints.GetEnumerator()) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint.Value -Method GET -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                Write-Status -Component $endpoint.Key -Status "PASS" -Message "Endpoint is accessible (Status: $($response.StatusCode))"
                $accessibleEndpoints += $endpoint.Key
            } else {
                Write-Status -Component $endpoint.Key -Status "FAIL" -Message "Endpoint returned error status (Status: $($response.StatusCode))"
                $inaccessibleEndpoints += "$($endpoint.Key) (Status: $($response.StatusCode))"
            }
        } catch {
            Write-Status -Component $endpoint.Key -Status "FAIL" -Message "Error accessing endpoint: $_"
            $inaccessibleEndpoints += "$($endpoint.Key) (Error: $_)"
        }
    }
    
    if ($inaccessibleEndpoints.Count -eq 0) {
        $results.ModelEndpoints.Status = "PASS"
        $results.ModelEndpoints.Details = "All endpoints are accessible"
    } else {
        $results.ModelEndpoints.Status = "FAIL"
        $results.ModelEndpoints.Details = "The following endpoints are inaccessible: $($inaccessibleEndpoints -join ', ')"
    }
}

# 4. Check that monitoring integrations are working
function Test-MonitoringIntegration {
    Write-Host "`n=== Checking Monitoring Integration ===" -ForegroundColor Blue
    
    $monitoringChecks = @()
    $monitoringFailures = @()
    
    # Check Prometheus targets
    try {
        $prometheusTargets = Invoke-RestMethod -Uri "http://localhost:9090/api/v1/targets" -Method GET
        $upTargets = ($prometheusTargets.data.activeTargets | Where-Object { $_.health -eq "up" }).Count
        $totalTargets = $prometheusTargets.data.activeTargets.Count
        
        if ($upTargets -eq $totalTargets -and $totalTargets -gt 0) {
            Write-Status -Component "Prometheus Targets" -Status "PASS" -Message "All targets are up ($upTargets/$totalTargets)"
            $monitoringChecks += "Prometheus Targets"
        } else {
            Write-Status -Component "Prometheus Targets" -Status "WARN" -Message "Some targets are down ($upTargets/$totalTargets)"
            $monitoringFailures += "Prometheus Targets ($upTargets/$totalTargets up)"
        }
    } catch {
        Write-Status -Component "Prometheus Targets" -Status "FAIL" -Message "Error checking targets: $_"
        $monitoringFailures += "Prometheus Targets"
    }
    
    # Check if ML metrics are being collected
    try {
        $mlMetricsQuery = Invoke-RestMethod -Uri "http://localhost:9090/api/v1/query?query=nv_inference_request_success" -Method GET
        if ($mlMetricsQuery.data.result.Count -gt 0) {
            Write-Status -Component "ML Metrics Collection" -Status "PASS" -Message "ML metrics are being collected"
            $monitoringChecks += "ML Metrics Collection"
        } else {
            Write-Status -Component "ML Metrics Collection" -Status "WARN" -Message "No ML metrics found"
            $monitoringFailures += "ML Metrics Collection"
        }
    } catch {
        Write-Status -Component "ML Metrics Collection" -Status "FAIL" -Message "Error checking metrics: $_"
        $monitoringFailures += "ML Metrics Collection"
    }
    
    # Check Grafana dashboards
    try {
        $grafanaDashboards = Invoke-RestMethod -Uri "http://localhost:3000/api/search?type=dash-db" -Method GET
        $mlDashboards = $grafanaDashboards | Where-Object { $_.title -match "ML" -or $_.title -match "Model" }
        
        if ($mlDashboards.Count -gt 0) {
            Write-Status -Component "Grafana ML Dashboards" -Status "PASS" -Message "Found $($mlDashboards.Count) ML dashboards"
            $monitoringChecks += "Grafana ML Dashboards"
        } else {
            Write-Status -Component "Grafana ML Dashboards" -Status "WARN" -Message "No ML dashboards found"
            $monitoringFailures += "Grafana ML Dashboards"
        }
    } catch {
        Write-Status -Component "Grafana ML Dashboards" -Status "FAIL" -Message "Error checking dashboards: $_"
        $monitoringFailures += "Grafana ML Dashboards"
    }
    
    if ($monitoringFailures.Count -eq 0) {
        $results.MonitoringIntegration.Status = "PASS"
        $results.MonitoringIntegration.Details = "All monitoring integrations are working"
    } else {
        $status = if ($monitoringFailures.Count -eq $monitoringChecks.Count + $monitoringFailures.Count) { "FAIL" } else { "WARN" }
        $results.MonitoringIntegration.Status = $status
        $results.MonitoringIntegration.Details = "The following monitoring integrations have issues: $($monitoringFailures -join ', ')"
    }
}

# 5. Verify GPU availability for training containers
function Test-GpuAvailability {
    Write-Host "`n=== Checking GPU Availability ===" -ForegroundColor Blue
    
    $gpuContainers = @(
        "ml-pytorch-master",
        "ml-pytorch-worker-1",
        "ml-pytorch-worker-2",
        "ml-triton-inference"
    )
    
    $containersWithGpu = @()
    $containersWithoutGpu = @()
    
    foreach ($container in $gpuContainers) {
        try {
            $gpuInfo = podman exec $container nvidia-smi 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Status -Component $container -Status "PASS" -Message "GPU available"
                $containersWithGpu += $container
            } else {
                Write-Status -Component $container -Status "FAIL" -Message "GPU not available"
                $containersWithoutGpu += $container
            }
        } catch {
            Write-Status -Component $container -Status "FAIL" -Message "Error checking GPU: $_"
            $containersWithoutGpu += "$container (Error)"
        }
    }
    
    if ($containersWithoutGpu.Count -eq 0 -and $containersWithGpu.Count -gt 0) {
        $results.GpuAvailability.Status = "PASS"
        $results.GpuAvailability.Details = "GPU available in all required containers"
    } else {
        $results.GpuAvailability.Status = "FAIL"
        $results.GpuAvailability.Details = "GPU not available in the following containers: $($containersWithoutGpu -join ', ')"
    }
}

# 6. Test connectivity to vector databases from training environment
function Test-VectorDatabaseConnectivity {
    Write-Host "`n=== Checking Vector Database Connectivity ===" -ForegroundColor Blue
    
    $vectorDbTests = @(
        @{Container = "ml-pytorch-master"; Type = "Milvus"; Command = "python -c 'from pymilvus import connections; connections.connect(host=\"vector-store-milvus\", port=\"19530\"); print(\"Connection successful\")'" },
        @{Container = "ml-pytorch-master"; Type = "Qdrant"; Command = "python -c 'import qdrant_client; client = qdrant_client.QdrantClient(url=\"http://vector-store-qdrant:6333\"); print(\"Connection successful\")'" }
    )
    
    $successfulDbTests = @()
    $failedDbTests = @()
    
    foreach ($test in $vectorDbTests) {
        try {
            $result = podman exec $test.Container bash -c $test.Command 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Status -Component "$($test.Container) → $($test.Type)" -Status "PASS" -Message "Connection successful"
                $successfulDbTests += "$($test.Container) → $($test.Type)"
            } else {
                Write-Status -Component "$($test.Container) → $($test.Type)" -Status "FAIL" -Message "Connection failed: $result"
                $failedDbTests += "$($test.Container) → $($test.Type)"
            }
        } catch {
            Write-Status -Component "$($test.Container) → $($test.Type)" -Status "FAIL" -Message "Error: $_"
            $failedDbTests += "$($test.Container) → $($test.Type)"
        }
    }
    
    # Check if vector databases have data
    try {
        $milvusCollectionCheck = podman exec ml-pytorch-master python -c "from pymilvus import connections, utility; connections.connect('default', host='vector-store-milvus', port='19530'); print(utility.list_collections())" 2>&1
        if ($milvusCollectionCheck -match "training_embeddings") {
            Write-Status -Component "Milvus Collections" -Status "PASS" -Message "Found training_embeddings collection"
            $successfulDbTests += "Milvus Collections"
        } else {
            Write-Status -Component "Milvus Collections" -Status "WARN" -Message "training_embeddings collection not found"
            $failedDbTests += "Milvus Collections"
        }
    } catch {
        Write-Status -Component "Milvus Collections" -Status "FAIL" -Message "Error checking collections: $_"
        $failedDbTests += "Milvus Collections"
    }
    
    if ($failedDbTests.Count -eq 0) {
        $results.VectorDbConnectivity.Status = "PASS"
        $results.VectorDbConnectivity.Details = "All vector database connectivity tests passed"
    } else {
        $status = if ($failedDbTests.Count -eq $successfulDbTests.Count + $failedDbTests.Count) { "FAIL" } else { "WARN" }
        $results.VectorDbConnectivity.Status = $status
        $results.VectorDbConnectivity.Details = "The following vector database tests failed: $($failedDbTests -join ', ')"
    }
}

# Display a summary of all validation results
function Show-ValidationSummary {
    Write-Host "`n`n=== ML Infrastructure Validation Summary ===" -ForegroundColor Magenta
    Write-Host "==============================================" -ForegroundColor Magenta
    
    $overallStatus = $true
    
    foreach ($result in $results.GetEnumerator()) {
        $statusColor = switch ($result.Value.Status) {
            "PASS" { "Green" }
            "WARN" { "Yellow"; $overallStatus = $overallStatus -and $true }
            "FAIL" { "Red"; $overallStatus = $false }
            default { "White" }
        }
        
        Write-Host "$($result.Key): " -NoNewline
        Write-Host $result.Value.Status -ForegroundColor $statusColor
        Write-Host "  - $($result.Value.Details)"
        Write-Host ""
    }
    
    Write-Host "==============================================" -ForegroundColor Magenta
    Write-Host -NoNewline "Overall Status: "
    
    if ($overallStatus) {
        Write-Host "PASS" -ForegroundColor Green
        Write-Host "ML Infrastructure validation completed successfully. All required components are functioning properly."
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        Write-Host "ML Infrastructure validation failed. Please review the details above and address any issues."
    }
}

# Main execution block
try {
    Write-Host "ML Infrastructure Validation Script" -ForegroundColor Cyan
    Write-Host "Starting validation of ML infrastructure components..." -ForegroundColor White
    
    # Run all validation checks
    Test-ContainerStatus
    Test-NetworkConnectivity
    Test-ModelEndpoints
    Test-MonitoringIntegration
    Test-GpuAvailability
    Test-VectorDatabaseConnectivity
    
    # Display summary
    Show-ValidationSummary
} catch {
    Write-Host "An unexpected error occurred during validation: $_" -ForegroundColor Red
    exit 1
}
