# Setup script for monitoring system
param(
    [string]$ConfigPath = "monitoring_config.json",
    [switch]$Force
)

# Import required modules
Import-Module -Name Microsoft.PowerShell.Utility

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    Write-Host "[$Status] $Message"
}

function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check Python installation
    try {
        $pythonVersion = python --version
        Write-Status "Found Python: $pythonVersion"
    }
    catch {
        Write-Status "Python not found. Please install Python 3.7 or later" "ERROR"
        return $false
    }
    
    # Check pip installation
    try {
        $pipVersion = pip --version
        Write-Status "Found pip: $pipVersion"
    }
    catch {
        Write-Status "pip not found. Please install pip" "ERROR"
        return $false
    }
    
    return $true
}

function Install-Requirements {
    Write-Status "Installing Python requirements..."
    
    $requirements = @(
        "prometheus_client",
        "croniter",
        "jsonschema",
        "requests"
    )
    
    foreach ($req in $requirements) {
        Write-Status "Installing $req..."
        pip install $req
        if ($LASTEXITCODE -ne 0) {
            Write-Status "Failed to install $req" "ERROR"
            return $false
        }
    }
    
    return $true
}

function Initialize-Directories {
    Write-Status "Initializing directory structure..."
    
    $directories = @(
        "logs/alerts",
        "logs/monitoring",
        "data/backups/daily",
        "data/backups/weekly",
        "data/backups/monthly"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Status "Created directory: $dir"
        }
    }
}

function Initialize-Configuration {
    param([string]$ConfigPath, [bool]$Force)
    
    Write-Status "Initializing configuration..."
    
    if (Test-Path $ConfigPath) {
        if ($Force) {
            Write-Status "Overwriting existing configuration file" "WARNING"
        }
        else {
            Write-Status "Configuration file already exists. Use -Force to overwrite" "WARNING"
            return $false
        }
    }
    
    # Load monitoring configuration
    try {
        $config = Get-Content $ConfigPath | ConvertFrom-Json -AsHashtable
        
        # Validate paths
        $dbPath = $config.state_management.sqlite.db_path
        $backupPath = $config.state_management.sqlite.backup_path
        
        # Create directories if they don't exist
        New-Item -ItemType Directory -Path (Split-Path $dbPath) -Force | Out-Null
        New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
        
        Write-Status "Configuration initialized successfully"
        return $true
    }
    catch {
        Write-Status "Failed to initialize configuration: $_" "ERROR"
        return $false
    }
}

function Initialize-Database {
    Write-Status "Initializing monitoring database..."
    
    try {
        # Import configuration
        $config = Get-Content $ConfigPath | ConvertFrom-Json -AsHashtable
        $dbPath = $config.state_management.sqlite.db_path
        
        # Initialize SQLite database
        python -c @"
import sqlite3
conn = sqlite3.connect('$dbPath')
cursor = conn.cursor()

# Create active alerts table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS active_alerts (
        alert_id TEXT PRIMARY KEY,
        alert_data TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Create alert history table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT NOT NULL,
        alert_data TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL
    )
''')

conn.commit()
conn.close()
"@
        
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Database initialized successfully"
            return $true
        }
        else {
            Write-Status "Failed to initialize database" "ERROR"
            return $false
        }
    }
    catch {
        Write-Status "Failed to initialize database: $_" "ERROR"
        return $false
    }
}

# Main setup process
Write-Status "Starting monitoring system setup..."

if (-not (Test-Prerequisites)) {
    Write-Status "Prerequisites check failed" "ERROR"
    exit 1
}

if (-not (Install-Requirements)) {
    Write-Status "Failed to install requirements" "ERROR"
    exit 1
}

Initialize-Directories

if (-not (Initialize-Configuration -ConfigPath $ConfigPath -Force $Force)) {
    Write-Status "Failed to initialize configuration" "ERROR"
    exit 1
}

if (-not (Initialize-Database)) {
    Write-Status "Failed to initialize database" "ERROR"
    exit 1
}

Write-Status "Setup completed successfully" "SUCCESS"
