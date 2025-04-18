<#
.SYNOPSIS
    Deployment and management script for AI and Data Processing Platform
.DESCRIPTION
    Manages the deployment, updates, and maintenance of the AI platform services
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        'start-all',
        'stop-all',
        'start-core',
        'start-ai',
        'start-monitoring',
        'backup-all',
        'update-services',
        'status',
        'logs'
    )]
    [string]$Command,
    
    [Parameter(Mandatory=$false)]
    [string]$Service
)

# Configuration
$BASE_DIR = $PSScriptRoot
$BACKUP_DIR = Join-Path $BASE_DIR "backups"
$LOG_DIR = Join-Path $BASE_DIR "logs"

# Ensure required directories exist
function Initialize-Directories {
    @($BACKUP_DIR, $LOG_DIR) | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force
        }
    }
}

# Start core services (databases, message queues)
function Start-CoreServices {
    Write-Host "Starting core services..."
    Set-Location (Join-Path $BASE_DIR "databases\vector-stores")
    podman-compose up -d
}

# Start AI services
function Start-AIServices {
    Write-Host "Starting AI services..."
    @(
        "ai-services\llm-inference",
        "ai-services\embedding-services",
        "ai-services\multimodal"
    ) | ForEach-Object {
        Set-Location (Join-Path $BASE_DIR $_)
        podman-compose up -d
    }
}

# Start monitoring stack
function Start-Monitoring {
    Write-Host "Starting monitoring services..."
    Set-Location (Join-Path $BASE_DIR "monitoring")
    podman-compose up -d
}

# Start all services
function Start-AllServices {
    Start-CoreServices
    Start-AIServices
    Start-Monitoring
}

# Stop all services
function Stop-AllServices {
    Write-Host "Stopping all services..."
    Get-ChildItem -Path $BASE_DIR -Recurse -Filter "podman-compose.yaml" |
    ForEach-Object {
        Set-Location $_.DirectoryName
        podman-compose down
    }
}

# Backup all volumes
function Backup-AllServices {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backup_path = Join-Path $BACKUP_DIR $timestamp
    
    Write-Host "Backing up services to $backup_path..."
    New-Item -ItemType Directory -Path $backup_path -Force
    
    # Backup volumes
    podman volume ls --format "{{.Name}}" | ForEach-Object {
        Write-Host "Backing up volume $_..."
        podman volume export $_ > (Join-Path $backup_path "$_.tar")
    }
}

# Update all services
function Update-Services {
    Write-Host "Updating services..."
    Get-ChildItem -Path $BASE_DIR -Recurse -Filter "podman-compose.yaml" |
    ForEach-Object {
        Set-Location $_.DirectoryName
        podman-compose pull
        podman-compose up -d
    }
}

# Show status of all services
function Show-Status {
    Write-Host "Service Status:"
    podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Show logs for specific service
function Show-Logs {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ServiceName
    )
    
    Write-Host "Showing logs for $ServiceName..."
    podman logs --tail 100 -f $ServiceName
}

# Main execution
Initialize-Directories

switch ($Command) {
    'start-all' { Start-AllServices }
    'stop-all' { Stop-AllServices }
    'start-core' { Start-CoreServices }
    'start-ai' { Start-AIServices }
    'start-monitoring' { Start-Monitoring }
    'backup-all' { Backup-AllServices }
    'update-services' { Update-Services }
    'status' { Show-Status }
    'logs' { Show-Logs -ServiceName $Service }
}
