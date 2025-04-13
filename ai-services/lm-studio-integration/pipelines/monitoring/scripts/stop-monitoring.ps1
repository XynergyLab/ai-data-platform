# Script to stop the monitoring system
param(
    [switch]$Force
)

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    Write-Host "[$Status] $Message"
}

function Stop-MonitoringSystem {
    param([bool]$Force)
    
    $processName = "python"
    $monitoringScript = "coordinator.py"
    
    $processes = Get-Process $processName -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*$monitoringScript*" }
    
    if ($processes) {
        foreach ($process in $processes) {
            if ($Force) {
                Stop-Process -Id $process.Id -Force
            } else {
                Stop-Process -Id $process.Id
            }
        }
        return $true
    }
    
    return $false
}

# Main execution
Write-Status "Stopping monitoring system..."

if (Stop-MonitoringSystem -Force $Force) {
    Write-Status "Monitoring system stopped successfully" "SUCCESS"
} else {
    Write-Status "No monitoring system processes found" "WARNING"
}
