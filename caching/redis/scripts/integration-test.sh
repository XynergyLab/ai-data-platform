#!/bin/bash

# Test Redis connectivity
echo "Testing Redis master connectivity..."
redis-cli -h redis-master -p 6379 -a your_strong_password_here ping

# Test replication status
echo "Testing replication status..."
redis-cli -h redis-master -p 6379 -a your_strong_password_here info replication

# Test Sentinel connectivity
echo "Testing Sentinel connectivity..."
redis-cli -h redis-sentinel-1 -p 26379 sentinel master mymaster

# Test basic caching operations
echo "Testing basic caching operations..."
redis-cli -h redis-master -p 6379 -a your_strong_password_here set test_key "Hello Redis"
redis-cli -h redis-master -p 6379 -a your_strong_password_here get test_key
redis-cli -h redis-master -p 6379 -a your_strong_password_here del test_key

# Test cache persistence
echo "Testing cache persistence..."
redis-cli -h redis-master -p 6379 -a your_strong_password_here save

# Test replica read
echo "Testing replica read..."
redis-cli -h redis-replica-1 -p 6379 -a your_strong_password_here get test_key

# Test failover (manual test)
echo "Testing manual failover..."
redis-cli -h redis-sentinel-1 -p 26379 sentinel failover mymaster

# Test memory usage
echo "Testing memory usage..."
redis-cli -h redis-master -p 6379 -a your_strong_password_here info memory

# Cleanup
echo "Cleaning up test data..."
redis-cli -h redis-master -p 6379 -a your_strong_password_here flushdb
