# Redis Caching Infrastructure

This configuration sets up a production-ready Redis caching system with high availability and monitoring capabilities.

## Architecture

- 1 Redis Master
- 2 Redis Replicas
- 3 Redis Sentinels for high availability
- Redis Commander for monitoring and management

## Integration Points

### AI Services
- LLM inference results caching
- Embedding vectors caching
- Model metadata caching
- Request rate limiting

### Data Processing
- Airflow task results caching
- Spark job metadata caching
- Intermediate data caching

### Vector Stores
- Query results caching
- Frequently accessed vector caching
- Metadata caching

## Configuration Details

### Master Node
- Memory: 8GB limit, 4GB reserved
- Persistence: RDB + AOF
- Max Memory Policy: allkeys-lru

### Replica Nodes
- Memory: 4GB limit, 2GB reserved
- Async replication
- Read-only operations

### Sentinel Setup
- Quorum: 2
- Failover timeout: 60000ms
- Down-after-milliseconds: 5000

## Monitoring

Access Redis Commander at http://localhost:8081

## Security

- Protected mode enabled
- Password authentication required
- Restricted commands in replicas
- Network isolation via Docker networks

## Usage

1. Start the Redis cluster:
   ```bash
   podman-compose up -d
   ```

2. Verify cluster health:
   ```bash
   ./scripts/redis-integration.sh
   ```

3. Monitor metrics:
   ```bash
   podman exec -it redis-master redis-cli -a your_strong_password_here INFO
   ```

## Integration Examples

### Python Integration
```python
from redis import Redis
from redis.sentinel import Sentinel

# Direct connection
redis_client = Redis(
    host='redis-master',
    port=6379,
    password='your_strong_password_here',
    decode_responses=True
)

# Sentinel connection
sentinel = Sentinel([
    ('redis-sentinel-1', 26379),
    ('redis-sentinel-2', 26380),
    ('redis-sentinel-3', 26381)
], password='your_strong_password_here')

master = sentinel.master_for('mymaster', password='your_strong_password_here')
replica = sentinel.slave_for('mymaster', password='your_strong_password_here')
```

### Cache Operations
```python
# Cache LLM results
def cache_llm_response(prompt_hash, response, expire_time=3600):
    redis_client.setex(f"llm:{prompt_hash}", expire_time, response)

# Cache embeddings
def cache_embedding(text_hash, embedding_vector, expire_time=3600):
    redis_client.setex(f"emb:{text_hash}", expire_time, embedding_vector)

# Rate limiting
def check_rate_limit(user_id, limit=100, window=3600):
    current = redis_client.incr(f"ratelimit:{user_id}")
    if current == 1:
        redis_client.expire(f"ratelimit:{user_id}", window)
    return current <= limit
```

## Maintenance

### Backup
```bash
# Create backup
podman exec redis-master redis-cli -a your_strong_password_here SAVE

# Copy backup file
podman cp redis-master:/data/dump.rdb /backup/redis/
```

### Scaling
```bash
# Add new replica
podman-compose up -d --scale redis-replica=3
```

### Failover Testing
```bash
# Trigger manual failover
redis-cli -h redis-sentinel-1 -p 26379 SENTINEL failover mymaster
```

Would you like to:
1. Start the Redis cluster and test the integration?
2. Configure specific caching policies for certain services?
3. Set up additional monitoring metrics?
4. Review the security configuration?