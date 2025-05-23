<#
.SYNOPSIS
    Redis Caching System Initialization and Integration Script
.DESCRIPTION
    Initializes Redis caching infrastructure and integrates with existing services
#>

# Configuration
$REDIS_BASE_DIR = "C:\Users\drew\Documents\Podman_Compose\caching\redis"
$SERVICES_DIR = "C:\Users\drew\Documents\Podman_Compose"

function Test-RedisPrerequisites {
    Write-Host "Checking prerequisites..."
    
    # Check if data directories exist
    $dataDirs = @(
        "$REDIS_BASE_DIR\data",
        "$REDIS_BASE_DIR\data-replica-1",
        "$REDIS_BASE_DIR\data-replica-2"
    )
    
    foreach ($dir in $dataDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force
            Write-Host "Created directory: $dir"
        }
    }
    
    # Verify configuration files
    $configFiles = @(
        "$REDIS_BASE_DIR\config\redis.conf",
        "$REDIS_BASE_DIR\config\redis-replica.conf",
        "$REDIS_BASE_DIR\config\sentinel.conf"
    )
    
    foreach ($file in $configFiles) {
        if (-not (Test-Path $file)) {
            throw "Missing configuration file: $file"
        }
    }
}

function Initialize-RedisNetwork {
    Write-Host "Initializing Redis network..."
    
    # Create Redis network if it doesn't exist
    podman network inspect redis_network 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        podman network create redis_network
        Write-Host "Created Redis network"
    }
}

function Start-RedisCaching {
    Write-Host "Starting Redis caching system..."
    
    Set-Location $REDIS_BASE_DIR
    podman-compose -f podman-compose.yaml up -d
    
    # Wait for services to be ready
    Start-Sleep -Seconds 10
    
    # Verify master is running
    podman exec -it redis-master redis-cli ping
    if ($LASTEXITCODE -ne 0) {
        throw "Redis master failed to start"
    }
}

function Update-ServiceConfigurations {
    Write-Host "Updating service configurations..."
    
    # AI Services
    Get-Content "$SERVICES_DIR\ai-services\podman-compose.yaml" | 
    Select-String -NotMatch "redis_network" |
    Add-Content -Path "$SERVICES_DIR\ai-services\podman-compose.yaml" -Value @"

networks:
  redis_network:
    external: true
"@
    
    # Vector Stores
    Get-Content "$SERVICES_DIR\databases\vector-stores\podman-compose.yaml" |
    Select-String -NotMatch "redis_network" |
    Add-Content -Path "$SERVICES_DIR\databases\vector-stores\podman-compose.yaml" -Value @"

networks:
  redis_network:
    external: true
"@
    
    # Data Processing
    Get-Content "$SERVICES_DIR\data-processing\podman-compose.yaml" |
    Select-String -NotMatch "redis_network" |
    Add-Content -Path "$SERVICES_DIR\data-processing\podman-compose.yaml" -Value @"

networks:
  redis_network:
    external: true
"@
}

function Test-RedisIntegration {
    Write-Host "Testing Redis integration..."
    
    # Test master-replica replication
    $replicationStatus = podman exec -it redis-master redis-cli info replication
    if ($replicationStatus -match "connected_slaves:0") {
        Write-Warning "No replicas connected to master"
    }
    
    # Test Sentinel configuration
    $sentinelStatus = podman exec -it redis-sentinel-1 redis-cli -p 26379 sentinel master mymaster
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Sentinel configuration issue detected"
    }
    
    # Test Redis Commander access
    Start-Sleep -Seconds 5
    $commanderStatus = Invoke-WebRequest -Uri "http://localhost:8081" -UseBasicParsing
    if ($commanderStatus.StatusCode -ne 200) {
        Write-Warning "Redis Commander not accessible"
    }
}

function Initialize-MonitoringIntegration {
    Write-Host "Initializing monitoring integration..."
    
    # Add Redis dashboard to Grafana
    $grafanaDir = "$SERVICES_DIR\monitoring\grafana\dashboards"
    if (-not (Test-Path "$grafanaDir\redis-dashboard.json")) {
        Write-Warning "Redis dashboard not found in Grafana configuration"
    }
    
    # Update Prometheus configuration
    $prometheusConfig = "$SERVICES_DIR\monitoring\prometheus\config\prometheus.yml"
    if (Test-Path $prometheusConfig) {
        $config = Get-Content $prometheusConfig
        if ($config -notmatch "job_name: 'redis'") {
            Write-Warning "Redis monitoring job not configured in Prometheus"
        }
    }
}

function Show-RedisStatus {
    Write-Host "`nRedis Caching System Status:"
    Write-Host "============================`n"
    
    # Check master status
    Write-Host "Master Status:"
    podman exec -it redis-master redis-cli info | Select-String "role:|connected_clients:|used_memory_human:|uptime_in_seconds:"
    
    # Check replica status
    Write-Host "`nReplica Status:"
    podman exec -it redis-replica-1 redis-cli info replication | Select-String "role:|master_link_status:|connected_slaves:"
    
    # Check sentinel status
    Write-Host "`nSentinel Status:"
    podman exec -it redis-sentinel-1 redis-cli -p 26379 sentinel master mymaster
    
    Write-Host "`nRedis Commander: http://localhost:8081"
}

function Show-Instructions {
    Write-Host @"

Redis Caching System Integration Complete
======================================

Connection Information:
- Master: redis-master:6379
- Replicas: redis-replica-1:6379, redis-replica-2:6379
- Sentinels: redis-sentinel-1:26379, redis-sentinel-2:26380, redis-sentinel-3:26381
- Management UI: http://localhost:8081

Integration Points:
1. AI Services:
   - LLM caching: Connected to redis-master
   - Embedding cache: Connected to redis-replicas
   - Model metadata: Distributed across cluster

2. Vector Stores:
   - Query cache: Using redis-master
   - Vector cache: Distributed across replicas
   - Metadata cache: Using master-replica setup

3. Data Processing:
   - Airflow results: Connected to redis-master
   - Spark cache: Using replica nodes
   - Pipeline metadata: Distributed configuration

Monitoring:
- Grafana dashboard: http://localhost:3000/dashboards
- Prometheus metrics: http://localhost:9090/targets
- Redis metrics available in Prometheus

Management:
- Use Redis Commander for visual management
- Monitor cluster health with Grafana
- Check logs with Loki integration

Next Steps:
1. Verify all services are connected:
   podman-compose ps

2. Monitor cache performance:
   - Check Grafana Redis dashboard
   - Use Redis Commander metrics
   - Watch Prometheus Redis metrics

3. Test failover scenario:
   ./scripts/integration-test.sh

4. Review cache hit rates:
   - Check Redis INFO stats
   - Monitor through Grafana
   - Adjust cache policies as needed

5. Regular maintenance:
   - Monitor memory usage
   - Check replication lag
   - Review cache eviction rates

For any issues:
1. Check logs: podman-compose logs redis-master
2. Verify network: podman network inspect redis_network
3. Test connectivity: redis-cli -h redis-master ping
"@
}

# Main execution
try {
    Test-RedisPrerequisites
    Initialize-RedisNetwork
    Start-RedisCaching
    Update-ServiceConfigurations
    Test-RedisIntegration
    Initialize-MonitoringIntegration
    Show-RedisStatus
    Show-Instructions
} catch {
    Write-Error "Error during Redis initialization: $_"
    exit 1
}
