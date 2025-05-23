# Redis Caching Quick Reference

## Connection Commands

```bash
# Connect to master
redis-cli -h redis-master -p 6379 -a your_strong_password_here

# Connect to replica
redis-cli -h redis-replica-1 -p 6379 -a your_strong_password_here

# Connect to sentinel
redis-cli -h redis-sentinel-1 -p 26379
```

## Common Operations

```bash
# Check replication status
redis-cli -h redis-master info replication

# Monitor cache hit rate
redis-cli -h redis-master info stats | grep hit_rate

# Check memory usage
redis-cli -h redis-master info memory

# List all keys
redis-cli -h redis-master keys *

# Monitor real-time commands
redis-cli -h redis-master monitor
```

## Cache Management

```bash
# Set key with expiration
redis-cli -h redis-master setex mykey 3600 "value"

# Get key
redis-cli -h redis-master get mykey

# Delete key
redis-cli -h redis-master del mykey

# Clear all keys (careful!)
redis-cli -h redis-master flushdb
```

## Health Checks

```bash
# Ping test
redis-cli -h redis-master ping

# Check role
redis-cli -h redis-master role

# Check clients
redis-cli -h redis-master client list
```

## Monitoring

```bash
# Basic stats
redis-cli -h redis-master info

# Memory stats
redis-cli -h redis-master info memory

# Replication stats
redis-cli -h redis-master info replication
```

## Sentinel Operations

```bash
# Get master info
redis-cli -h redis-sentinel-1 -p 26379 sentinel master mymaster

# Get replicas
redis-cli -h redis-sentinel-1 -p 26379 sentinel replicas mymaster

# Manual failover
redis-cli -h redis-sentinel-1 -p 26379 sentinel failover mymaster
```

## Common Issues

1. Connection refused:
   - Check if Redis is running
   - Verify network connectivity
   - Check firewall settings

2. Authentication failed:
   - Verify password in configuration
   - Check AUTH command usage

3. Replication issues:
   - Check network connectivity
   - Verify replica configuration
   - Monitor replication lag

4. Memory issues:
   - Monitor memory usage
   - Check maxmemory setting
   - Review eviction policy

5. Sentinel failover:
   - Check quorum settings
   - Verify sentinel connectivity
   - Monitor failover logs
