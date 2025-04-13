# Script to check monitoring system status
param(
    [string]$ConfigPath = "config/monitoring_config.json"
)

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    Write-Host "[$Status] $Message"
}

function Get-MonitoringStatus {
    $pythonScript = @"
from monitoring.coordinator import MonitoringCoordinator
import json
import sys

try:
    coordinator = MonitoringCoordinator('$ConfigPath')
    status = coordinator.get_monitoring_status()
    print(json.dumps(status, indent=2))
    sys.exit(0)
except Exception as e:
    print(json.dumps({'error': str(e)}))
    sys.exit(1)
"@
    
    # Save script to temporary file
    $tempScript = Join-Path $env:TEMP "check_status.py"
    $pythonScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    # Execute script and capture output
    $result = python $tempScript
    
    if ($LASTEXITCODE -eq 0) {
        return $result | ConvertFrom-Json
    }
    
    return $null
}

function Format-MetricsTable {
    param($Metrics)
    
    $table = @()
    foreach ($collector in $Metrics.PSObject.Properties) {
        $collectorName = $collector.Name
        $collectorData = $collector.Value
        
        foreach ($metric in $collectorData.metrics_count.PSObject.Properties) {
            $table += [PSCustomObject]@{
                Collector = $collectorName
                Metric = $metric.Name
                Count = $metric.Value
                LastUpdate = $collectorData.last_collection
            }
        }
    }
    
    return $table
}

# Main execution
Write-Status "Checking monitoring system status..."

$status = Get-MonitoringStatus
if ($status) {
    # Display system status
    Write-Status "System Status:"
    Write-Status "  Collectors Active: $($status.components.collectors_active)"
    Write-Status "  Alert Manager: $($status.components.alert_manager)"
    Write-Status "  State Manager: $($status.components.state_manager)"
    
    # Display metrics
    Write-Status "`nMetrics Summary:"
    Format-MetricsTable $status.metrics | Format-Table -AutoSize
    
    # Display active alerts
    Write-Status "`nActive Alerts:"
    if ($status.alerts.active) {
        $status.alerts.active | Format-Table -AutoSize
    } else {
        Write-Status "  No active alerts"
    }
    
    # Display backup status
    Write-Status "`nBackup Status:"
    Write-Status "  Last Backup: $($status.backups.last_backup)"
    Write-Status "  Next Backup: $($status.backups.next_backup)"
    Write-Status "  Total Backups: $($status.backups.total_backups)"
} else {
    Write-Status "Failed to get monitoring system status" "ERROR"
    exit 1
}
