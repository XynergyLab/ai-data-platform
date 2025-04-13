# Script to start the monitoring system
param(
    [string]$ConfigPath = "config/monitoring_config.json",
    [switch]$ForceDaemon
)

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    Write-Host "[$Status] $Message"
}

function Test-MonitoringSystem {
    # Check if monitoring system is already running
    $processName = "python"
    $monitoringScript = "coordinator.py"
    
    $running = Get-Process $processName -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*$monitoringScript*" }
    
    return $running -ne $null
}

function Start-MonitoringDaemon {
    param([string]$ConfigPath)
    
    $pythonScript = @"
from monitoring.coordinator import MonitoringCoordinator
import sys
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring/daemon.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('MonitoringDaemon')

try:
    # Initialize coordinator
    coordinator = MonitoringCoordinator('$ConfigPath')
    
    # Start monitoring
    coordinator.start_monitoring()
    
    logger.info('Monitoring system started')
    
    # Keep running
    while True:
        time.sleep(1)
        
except Exception as e:
    logger.error(f'Error in monitoring daemon: {str(e)}')
    sys.exit(1)
"@
    
    # Save script to temporary file
    $tempScript = Join-Path $env:TEMP "monitoring_daemon.py"
    $pythonScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    # Start daemon process
    Start-Process python -ArgumentList $tempScript -NoNewWindow
}

# Main execution
Write-Status "Starting monitoring system..."

# Check if system is already running
if (Test-MonitoringSystem) {
    if ($ForceDaemon) {
        Write-Status "Forcing restart of monitoring system..." "WARNING"
        Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
    } else {
        Write-Status "Monitoring system is already running" "ERROR"
        exit 1
    }
}

# Verify configuration exists
if (-not (Test-Path $ConfigPath)) {
    Write-Status "Configuration file not found: $ConfigPath" "ERROR"
    exit 1
}

# Start monitoring daemon
try {
    Start-MonitoringDaemon -ConfigPath $ConfigPath
    Write-Status "Monitoring system started successfully" "SUCCESS"
} catch {
    Write-Status "Failed to start monitoring system: $_" "ERROR"
    exit 1
}
