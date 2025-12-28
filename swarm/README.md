# Docker Swarm Configuration Guide
# AI Platform SDK - Complete Swarm Setup

## Overview
Docker Swarm is a lighter alternative to Kubernetes for orchestrating containerized applications. This guide covers deploying AI Platform SDK to Docker Swarm.

## Prerequisites

### Install Docker
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify installation
docker --version
```

### Initialize Swarm
```bash
# On manager node
docker swarm init --advertise-addr <manager-ip>

# Get join token (for worker nodes)
docker swarm join-token worker

# Get manager token (if adding more managers)
docker swarm join-token manager

# On worker nodes
docker swarm join --token <token> <manager-ip>:2377
```

## Deployment Steps

### 1. Pre-deployment Setup
```bash
# Create necessary directories
mkdir -p config logs data certs

# Create secrets (non-production method)
docker secret create jwt_secret ./secrets/jwt_secret.txt
docker secret create api_key ./secrets/api_key.txt
docker secret create db_password ./secrets/db_password.txt
docker secret create redis_password ./secrets/redis_password.txt
```

### 2. Deploy Stack
```bash
# Deploy using docker-compose.swarm.yml
docker stack deploy -c docker-compose.swarm.yml aiplatform

# Verify deployment
docker stack services aiplatform
docker stack ps aiplatform

# View logs
docker service logs aiplatform_aiplatform-api
```

### 3. Scale Services
```bash
# Scale API service to 5 replicas
docker service scale aiplatform_aiplatform-api=5

# Scale mesh workers to 10 replicas
docker service scale aiplatform_aiplatform-mesh-worker=10

# Check scaled services
docker service ls
```

## Management

### Monitor Services
```bash
# List services
docker service ls

# Service details
docker service inspect aiplatform_aiplatform-api

# Service logs
docker service logs aiplatform_aiplatform-api

# Service processes
docker service ps aiplatform_aiplatform-api
```

### Update Services
```bash
# Update image
docker service update --image aiplatform/sdk:1.1.0 aiplatform_aiplatform-api

# Update environment variable
docker service update --env-add NEW_VAR=value aiplatform_aiplatform-api

# Update resource limits
docker service update \
  --limit-memory 2g \
  --limit-cpu 2 \
  aiplatform_aiplatform-api

# Update replicas
docker service update --mode replicated --replicas 5 aiplatform_aiplatform-api
```

### Rollback Services
```bash
# Rollback to previous version
docker service rollback aiplatform_aiplatform-api

# View update status
docker service update -f aiplatform_aiplatform-api
```

## Networking

### Overlay Networks
```bash
# Create network
docker network create -d overlay aiplatform_network

# List networks
docker network ls

# Inspect network
docker network inspect aiplatform_network
```

### Service Discovery
Services are automatically discoverable by name within the swarm:
- `aiplatform-api:8000` (service discovery)
- `aiplatform-mesh-worker:8000`
- `postgres:5432`
- `redis:6379`

## Storage

### Volumes
Persistent data directories:
- `postgres_data` - Database
- `redis_data` - Cache
- `prometheus_data` - Metrics
- `grafana_data` - Dashboards

### Manage Volumes
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect aiplatform_postgres_data

# Backup volume
docker run --rm -v aiplatform_postgres_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/postgres-backup.tar.gz /data

# Restore volume
docker run --rm -v aiplatform_postgres_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/postgres-backup.tar.gz -C /
```

## Monitoring & Logging

### Health Checks
Services include healthchecks. Monitor with:
```bash
# Check service health
docker service inspect --format='{{json .Endpoint.VirtualIPs}}' aiplatform_aiplatform-api

# View task status
docker service ps aiplatform_aiplatform-api
```

### Logging
```bash
# View service logs
docker service logs aiplatform_aiplatform-api -f

# View specific number of lines
docker service logs aiplatform_aiplatform-api --tail 100

# View since timestamp
docker service logs aiplatform_aiplatform-api --since 2024-01-01T00:00:00
```

### Prometheus Metrics
Access Prometheus at `http://swarm-host:9090`
- Scrapes metrics from all services
- Retention: 30 days
- Accessible from Grafana dashboards

### Grafana Dashboards
Access Grafana at `http://swarm-host:3000`
- Default credentials: admin/admin
- Pre-configured datasources
- Import dashboards from grafana.com

## Security

### Network Security
- Services communicate over encrypted overlay network
- External access via nginx load balancer only
- Network policies restrict traffic

### Secrets Management
```bash
# Create secret from file
docker secret create db_password ./secrets/db_password.txt

# Create secret from stdin
echo "my-secret" | docker secret create api_key -

# List secrets
docker secret ls

# Remove secret
docker secret rm api_key
```

### TLS/SSL
- Mount certificates in `./certs/` directory
- Nginx configured for HTTPS
- Certificate auto-renewal via certbot

## Scaling Strategies

### Horizontal Scaling
```bash
# Add worker nodes
docker swarm join-token worker

# Scale out
docker service scale aiplatform_aiplatform-api=10

# Monitor scaling
watch docker service ps aiplatform_aiplatform-api
```

### Resource Constraints
Each service has:
- CPU limits
- Memory limits
- Resource reservations

Modify in `docker-compose.swarm.yml` `deploy` section.

## Troubleshooting

### Service Not Scaling
```bash
# Check node availability
docker node ls

# Check resource availability
docker node inspect <node-id>

# Check service logs
docker service logs aiplatform_aiplatform-api

# Check task placement
docker service ps --filter "desired-state=running" aiplatform_aiplatform-api
```

### Connectivity Issues
```bash
# Test network connectivity
docker run --rm --network aiplatform_aiplatform \
  nicolaka/netshoot ping postgres

# Check DNS resolution
docker run --rm --network aiplatform_aiplatform \
  nicolaka/netshoot nslookup redis

# Test service communication
docker run --rm --network aiplatform_aiplatform \
  curlimages/curl curl http://aiplatform-api:8000/health
```

### Performance Issues
```bash
# Check container resource usage
docker stats

# Check node resources
docker node inspect <node-id> --pretty

# Check service metrics
docker service inspect --format='{{.Spec.Resources}}' aiplatform_aiplatform-api
```

## Updates & Maintenance

### Update Stack
```bash
# Update compose file, then:
docker stack deploy -c docker-compose.swarm.yml aiplatform

# This will update/recreate services as needed
```

### Rolling Updates
Services automatically perform rolling updates:
- Old replicas stop
- New replicas start
- No downtime

Configure update strategy in compose file:
```yaml
update_config:
  parallelism: 1
  delay: 10s
  failure_action: rollback
```

### Backup Database
```bash
# Backup PostgreSQL
docker exec $(docker ps -f "service=aiplatform_postgres" -q) \
  pg_dump -U postgres aiplatform > backup.sql

# Backup Redis
docker exec $(docker ps -f "service=aiplatform_redis" -q) \
  redis-cli BGSAVE

# Backup volumes
docker run --rm -v aiplatform_postgres_data:/data \
  -v $(pwd):/backup alpine \
  tar czf /backup/swarm-backup.tar.gz /data
```

## Cleanup

### Remove Stack
```bash
# Remove services (keeps volumes)
docker stack rm aiplatform

# Verify removal
docker service ls
```

### Remove Volumes
```bash
# WARNING: This deletes data!
docker volume rm aiplatform_postgres_data aiplatform_redis_data

# Or remove all unused volumes
docker volume prune
```

### Leave Swarm
```bash
# On worker node
docker swarm leave

# On manager node (force)
docker swarm leave --force
```

## Advanced: Multi-Node Setup

### Manager Nodes (3+ recommended)
```bash
# Initialize first manager
docker swarm init

# Add second manager
docker swarm join-token manager
docker swarm join --token <token> <manager-ip>:2377

# Verify managers
docker node ls
```

### Worker Nodes
```bash
# Join as worker
docker swarm join --token <token> <manager-ip>:2377

# Worker tasks run containers
# Manager nodes distribute workload
```

### Labels for Constraints
```bash
# Label node
docker node update --label-add hardware=gpu <node-id>

# Constrain service to labeled nodes
# In compose file:
# deploy:
#   placement:
#     constraints:
#       - node.labels.hardware == gpu
```

## Comparison: Swarm vs Kubernetes

| Feature | Swarm | Kubernetes |
|---------|-------|-----------|
| Setup | Simple | Complex |
| Learning Curve | Easy | Steep |
| Scalability | Good | Excellent |
| Features | Basic | Advanced |
| Community | Smaller | Large |
| Production-Ready | Yes | Yes |
| Self-healing | Yes | Yes |
| Auto-scaling | Manual | Automatic |
| Networking | Overlay | CNI plugins |

**Use Swarm for:**
- Small to medium deployments
- Simple scaling needs
- Quick setup
- Docker-focused teams

**Use Kubernetes for:**
- Large-scale deployments
- Complex networking
- Advanced features
- Multi-cloud

## Support & Resources

- [Docker Swarm Docs](https://docs.docker.com/engine/swarm/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Docker Networking](https://docs.docker.com/network/)

## License
Same as AI Platform SDK
