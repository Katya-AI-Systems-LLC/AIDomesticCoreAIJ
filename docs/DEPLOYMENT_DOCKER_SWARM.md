# Docker Swarm Deployment Guide
# AI Platform SDK - Lightweight Container Orchestration

## Overview

Docker Swarm is a simpler alternative to Kubernetes, built into Docker. Perfect for teams wanting orchestration without Kubernetes complexity.

## Quick Start (10 minutes)

### 1. Initialize Swarm
```bash
# On manager node
docker swarm init --advertise-addr <manager-ip>

# Get worker token
docker swarm join-token worker

# On worker nodes
docker swarm join --token <token> <manager-ip>:2377
```

### 2. Deploy Stack
```bash
cd swarm/

# Deploy
docker stack deploy -c docker-compose.swarm.yml aiplatform

# Verify
docker stack services aiplatform
docker stack ps aiplatform
```

### 3. Monitor
```bash
# Follow services
docker service ls
docker service logs aiplatform_aiplatform-api -f

# Scale
docker service scale aiplatform_aiplatform-api=5
```

## Complete Setup

### Prerequisites
- **Docker** 20.10+
- **Docker Compose** 1.29+
- **3+ Nodes** (1 manager, 2+ workers) - or single node for testing
- **Linux kernel** 4.4+ with VXLAN support

### Multi-Node Setup

**Manager Node:**
```bash
# Initialize swarm
docker swarm init --advertise-addr 192.168.1.100

# Get join token
docker swarm join-token worker
```

**Worker Nodes:**
```bash
# Join swarm
docker swarm join --token <token> 192.168.1.100:2377

# Verify on manager
docker node ls
```

### Deploy Monitoring Stack

**Prometheus:**
```bash
# Included in docker-compose.swarm.yml
# Access at http://manager:9090
```

**Grafana:**
```bash
# Included in docker-compose.swarm.yml
# Access at http://manager:3000
# Default: admin/admin
```

**Loki (Log Aggregation):**
```bash
# Included in docker-compose.swarm.yml
# Access at http://manager:3100
```

## Configuration

### Network Configuration
```bash
# Create overlay network
docker network create -d overlay aiplatform_network

# List networks
docker network ls

# Inspect network
docker network inspect aiplatform_network
```

### Volume Management
```bash
# Create named volumes
docker volume create postgres_data
docker volume create redis_data

# List volumes
docker volume ls

# Backup volume
docker run --rm -v postgres_data:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/db-backup.tar.gz /data
```

### Secrets Management
```bash
# Create secrets
echo "my-secret-value" | docker secret create jwt_secret -
echo "db-password" | docker secret create db_password -
echo "api-key" | docker secret create api_key -

# List secrets
docker secret ls

# Use in compose (reference in deploy.secrets)
```

## Services

### API Service
```bash
# View API service
docker service inspect aiplatform_aiplatform-api

# View logs
docker service logs aiplatform_aiplatform-api -f

# Scale up
docker service scale aiplatform_aiplatform-api=10

# Update image
docker service update \
  --image aiplatform/sdk:1.1.0 \
  aiplatform_aiplatform-api
```

### Mesh Workers
```bash
# Scale to 20 workers
docker service scale aiplatform_aiplatform-mesh-worker=20

# View worker distribution
docker service ps aiplatform_aiplatform-mesh-worker
```

### Database
```bash
# Backup PostgreSQL
docker exec $(docker ps -f "service=aiplatform_postgres" -q) \
  pg_dump -U postgres aiplatform > backup.sql

# Restore
docker exec -i $(docker ps -f "service=aiplatform_postgres" -q) \
  psql -U postgres aiplatform < backup.sql
```

### Cache
```bash
# Redis CLI
docker exec -it $(docker ps -f "service=aiplatform_redis" -q) \
  redis-cli

# Check memory
MEMORY_USAGE
```

## Monitoring

### Service Metrics
```bash
# Real-time stats
docker stats

# Service status
docker service ls
docker service ps aiplatform_aiplatform-api

# Service health
docker service inspect --format='{{json .ServiceStatus}}' \
  aiplatform_aiplatform-api
```

### Logging
```bash
# Service logs
docker service logs aiplatform_aiplatform-api

# Follow logs
docker service logs aiplatform_aiplatform-api -f

# Tail specific lines
docker service logs aiplatform_aiplatform-api --tail 100

# Filter logs
docker service logs aiplatform_aiplatform-api 2>&1 | grep ERROR
```

### Prometheus Metrics
```bash
# Access Prometheus
# http://manager-ip:9090

# Query CPU usage
rate(container_cpu_usage_seconds_total[1m])

# Query memory usage
container_memory_usage_bytes
```

### Grafana Dashboards
```bash
# Access Grafana
# http://manager-ip:3000
# Login: admin/admin

# Import dashboards for Docker Swarm:
# - Docker Engine (3143)
# - Kubernetes Cluster
# - Node Exporter
```

## Scaling Strategies

### Horizontal Scaling
```bash
# Scale API
docker service scale aiplatform_aiplatform-api=20

# Scale workers
docker service scale aiplatform_aiplatform-mesh-worker=50

# Monitor scaling
watch -n 1 'docker service ps aiplatform_aiplatform-api'
```

### Update Service
```bash
# Rolling update with zero downtime
docker service update \
  --image aiplatform/sdk:1.1.0 \
  --update-delay 10s \
  --update-parallelism 1 \
  aiplatform_aiplatform-api

# Watch update progress
docker service ps aiplatform_aiplatform-api
```

### Resource Constraints
```bash
# Update resource limits
docker service update \
  --limit-cpu 2 \
  --limit-memory 2G \
  --reserve-cpu 1 \
  --reserve-memory 1G \
  aiplatform_aiplatform-api
```

## Networking

### Service Discovery
```bash
# Services discoverable internally:
# aiplatform_aiplatform-api:8000
# aiplatform_postgres:5432
# aiplatform_redis:6379

# Test from service
docker exec -it <container-id> \
  curl http://aiplatform_aiplatform-api:8000/health
```

### Load Balancing
```bash
# Services load-balanced automatically
# Multiple replicas = round-robin by default

# Sticky routing via session affinity
docker service update \
  --endpoint-mode dnsrr \
  aiplatform_aiplatform-api
```

## Data Persistence

### Backup Strategy
```bash
# Backup all volumes
for volume in postgres_data redis_data; do
  docker run --rm -v $volume:/data \
    -v $(pwd):/backup alpine \
    tar czf /backup/$volume-$(date +%Y%m%d).tar.gz /data
done

# Backup docker config
docker config ls
docker secret ls
```

### Restore
```bash
# Restore volume
docker run --rm -v postgres_data:/data \
  -v $(pwd):/backup alpine \
  tar xzf /backup/postgres_data-20240101.tar.gz -C /

# Restart service
docker service update --force aiplatform_postgres
```

## High Availability

### Multi-Node Deployment
```bash
# Check node status
docker node ls

# Set manager mode
docker node update --role manager <node-id>

# Set worker mode
docker node update --role worker <node-id>

# Drain node (maintenance)
docker node update --availability drain <node-id>

# Restore node
docker node update --availability active <node-id>
```

### Service Replication
```bash
# Ensure replicas spread across nodes
docker service create \
  --mode replicated \
  --replicas 5 \
  --constraint node.role==worker \
  aiplatform/sdk

# View replica distribution
docker service ps aiplatform_aiplatform-api
```

## Troubleshooting

### Service Not Running
```bash
# Check service status
docker service ls
docker service inspect aiplatform_aiplatform-api

# Check tasks
docker service ps aiplatform_aiplatform-api

# Check logs
docker service logs aiplatform_aiplatform-api
```

### Network Issues
```bash
# Test connectivity
docker run --rm --network aiplatform_aiplatform \
  nicolaka/netshoot ping postgres

# Resolve DNS
docker run --rm --network aiplatform_aiplatform \
  nicolaka/netshoot nslookup aiplatform_postgres

# Test service port
docker run --rm --network aiplatform_aiplatform \
  curlimages/curl curl http://aiplatform_aiplatform-api:8000/health
```

### Performance Issues
```bash
# Check resource usage
docker stats

# Check node resources
docker node inspect <node-id> -f '{{json .Status}}'

# Check service constraints
docker service inspect --format='{{json .Spec.TaskTemplate.Placement}}' \
  aiplatform_aiplatform-api
```

## Cleanup

### Remove Stack
```bash
# Remove services (keeps volumes)
docker stack rm aiplatform

# Verify removal
docker service ls
```

### Remove Volumes (Caution!)
```bash
# Remove specific volume
docker volume rm postgres_data

# Remove all unused volumes
docker volume prune
```

### Leave Swarm
```bash
# On worker
docker swarm leave

# On manager (force)
docker swarm leave --force

# Verify
docker node ls  # Error: not in swarm
```

## Comparison: Swarm vs Kubernetes

| Feature | Swarm | K8s |
|---------|-------|-----|
| Complexity | Low | High |
| Learning Curve | Shallow | Steep |
| Setup Time | 5 min | 30 min |
| Features | Basic | Advanced |
| Scalability | Good | Excellent |
| Auto-scaling | Manual | Automatic |
| Monitoring | Basic | Extensive |
| Community | Small | Huge |
| Production Ready | Yes | Yes |
| Multi-cloud | Yes | Yes |

**Choose Swarm if:**
- Small to medium scale (< 100 nodes)
- Docker-focused team
- Simplicity preferred
- Quick deployment needed
- Limited DevOps resources

**Choose Kubernetes if:**
- Large-scale deployment (> 100 nodes)
- Complex networking needed
- Advanced features required
- Established DevOps team
- Multi-cloud strategy

## Maintenance

### Updates
```bash
# Update Docker
docker --version

# Update service image
docker service update \
  --image aiplatform/sdk:latest \
  aiplatform_aiplatform-api

# Rolling restart
docker service update --force aiplatform_aiplatform-api
```

### Health Checks
```bash
# Verify service health
docker service ls

# Check task status
docker service ps aiplatform_aiplatform-api

# View unhealthy tasks
docker service ps --filter "desired-state=shutdown" aiplatform_aiplatform-api
```

## Production Checklist

- [ ] **Nodes**: 3+ nodes for HA
- [ ] **Manager**: Multiple managers (3 or 5)
- [ ] **Networking**: Overlay networks configured
- [ ] **Secrets**: Secrets configured for sensitive data
- [ ] **Volumes**: Named volumes for persistence
- [ ] **Monitoring**: Prometheus + Grafana operational
- [ ] **Logging**: Loki collecting all logs
- [ ] **Backup**: Automated backup strategy
- [ ] **Health Checks**: Liveness probes configured
- [ ] **Scaling**: Auto-scaling policies defined
- [ ] **Updates**: Rolling update strategy tested
- [ ] **Security**: Firewall rules enforced
- [ ] **Documentation**: Runbooks created
- [ ] **Alerts**: Alert thresholds configured
- [ ] **Testing**: Failover tested

## Support & Resources

- [Docker Swarm Docs](https://docs.docker.com/engine/swarm/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Swarm Mode Networking](https://docs.docker.com/network/network-tutorial-overlay/)

## License
Same as AI Platform SDK
