# Script to manage monitoring system backups
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('create', 'list', 'restore', 'cleanup')]
    [string]$Action,
    
    [ValidateSet('daily', 'weekly', 'monthly')]
    [string]$BackupType = 'daily',
    
    [string]$BackupId,
    
    [switch]$Force
)

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    Write-Host "[$Status] $Message"
}

function Invoke-BackupAction {
    param(
        [string]$Action,
        [hashtable]$Params
    )
    
    $pythonScript = @"
from monitoring.alerts.backup import BackupManager, RecoveryManager
from monitoring.config import ConfigurationValidator
import json
import sys

try:
    # Load configuration
    validator = ConfigurationValidator()
    with open('config/monitoring_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize managers
    backup_manager = BackupManager(config)
    recovery_manager = RecoveryManager(config)
    
    # Execute requested action
    result = None
    if '$Action' == 'create':
        result = backup_manager.create_backup('$($Params.BackupType)')
    elif '$Action' == 'list':
        result = backup_manager.list_backups()
    elif '$Action' == 'restore':
        result = recovery_manager.recover_from_backup('$($Params.BackupId)')
    elif '$Action' == 'cleanup':
        result = backup_manager.rotate_backups('$($Params.BackupType)', 
                                            config['state_management']['sqlite']['max_$($Params.BackupType)_backups'])
    
    print(json.dumps(result))
    sys.exit(0)
except Exception as e:
    print(json.dumps({'error': str(e)}))
    sys.exit(1)
"@
    
    # Save script to temporary file
    $tempScript = Join-Path $env:TEMP "manage_backups.py"
    $pythonScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    # Execute script and capture output
    $result = python $tempScript | ConvertFrom-Json
    return $result
}

# Main execution
Write-Status "Managing backups: $Action"

switch ($Action) {
    'create' {
        Write-Status "Creating new $BackupType backup..."
        $result = Invoke-BackupAction -Action 'create' -Params @{
            BackupType = $BackupType
        }
        
        if ($result) {
            Write-Status "Backup created successfully" "SUCCESS"
            $result | Format-List
        } else {
            Write-Status "Failed to create backup" "ERROR"
            exit 1
        }
    }
    
    'list' {
        Write-Status "Listing available backups..."
        $backups = Invoke-BackupAction -Action 'list' -Params @{}
        
        if ($backups) {
            foreach ($type in $backups.PSObject.Properties) {
                Write-Status "`n$($type.Name) backups:"
                $type.Value | Format-Table -AutoSize
            }
        } else {
            Write-Status "No backups found" "WARNING"
        }
    }
    
    'restore' {
        if (-not $BackupId) {
            Write-Status "BackupId parameter is required for restore action" "ERROR"
            exit 1
        }
        
        if (-not $Force) {
            $confirm = Read-Host "Are you sure you want to restore from backup $BackupId? (y/N)"
            if ($confirm -ne 'y') {
                Write-Status "Restore cancelled" "WARNING"
                exit 0
            }
        }
        
        Write-Status "Restoring from backup $BackupId..."
        $result = Invoke-BackupAction -Action 'restore' -Params @{
            BackupId = $BackupId
        }
        
        if ($result) {
            Write-Status "Backup restored successfully" "SUCCESS"
        } else {
            Write-Status "Failed to restore backup" "ERROR"
            exit 1
        }
    }
    
    'cleanup' {
        Write-Status "Cleaning up old $BackupType backups..."
        $result = Invoke-BackupAction -Action 'cleanup' -Params @{
            BackupType = $BackupType
        }
        
        if ($result -ge 0) {
            Write-Status "Removed $result old backups" "SUCCESS"
        } else {
            Write-Status "Failed to cleanup backups" "ERROR"
            exit 1
        }
    }
}
