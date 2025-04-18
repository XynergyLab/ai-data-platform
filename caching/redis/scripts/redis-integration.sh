#!/bin/bash

# Redis integration health check
check_redis_health() {
    redis-cli -h redis-master -p 6379 -a your_strong_password_here ping
}

# Cache model metadata
cache_model_metadata() {
    redis-cli -h redis-master -p 6379 -a your_strong_password_here HSET model:$1 name $1 version $2 loaded_time "$(date +%s)"
}

# Cache embedding results
cache_embedding() {
    redis-cli -h redis-master -p 6379 -a your_strong_password_here SETEX "embedding:$1" 3600 "$2"
}

# Cache LLM generation results
cache_llm_result() {
    redis-cli -h redis-master -p 6379 -a your_strong_password_here SETEX "llm:$1" 3600 "$2"
}

# Monitor Redis metrics
monitor_redis_metrics() {
    redis-cli -h redis-master -p 6379 -a your_strong_password_here INFO
}
