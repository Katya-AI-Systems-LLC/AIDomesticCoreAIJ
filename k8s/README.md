# Kubernetes Configuration Reference Guide
# AI Platform SDK - Complete K8s Setup

## Overview
This directory contains complete Kubernetes manifests for deploying the AI Platform SDK to production.

## Files

### Core Manifests
- **namespace.yaml** - Kubernetes namespace, network policies, resource quotas
- **deployment.yaml** - Main deployments (API, Mesh Workers, 3D Renderer)
- **service.yaml** - Service definitions (ClusterIP, NodePort, LoadBalancer)
- **configmap.yaml** - Configuration files (app.yaml, logging.yaml, nginx.conf)
- **secrets.yaml.template** - Secret templates (copy and fill with real values)
- **ingress.yaml** - Ingress rules, TLS certificates, cert-manager setup
- **hpa.yaml** - Horizontal & Vertical Pod Autoscaling
- **pvc.yaml** - Persistent volumes and storage classes

## Quick Start

### 1. Prerequisites
```bash
# Install kubectl
# Install helm (optional)
# Configure kubeconfig for your cluster

kubectl config current-context
```

### 2. Create Namespace
```bash
kubectl apply -f namespace.yaml
```

### 3. Setup Secrets
```bash
# Copy template and edit with real values
cp secrets.yaml.template secrets.yaml
# Edit secrets.yaml with your actual credentials
# Then apply:
kubectl apply -f secrets.yaml
```

### 4. Deploy Application
```bash
# Deploy in order
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
```

### 5. Verify Deployment
```bash
# Check namespace
kubectl get ns aiplatform

# Check pods
kubectl get pods -n aiplatform

# Check services
kubectl get svc -n aiplatform

# Check ingress
kubectl get ingress -n aiplatform

# Check PVCs
kubectl get pvc -n aiplatform

# Check HPA
kubectl get hpa -n aiplatform
```

## Deployment Components

### Deployments
1. **aiplatform-api** (3 replicas)
   - Main API server
   - Quantum worker sidecar
   - Port: 8000 (HTTP), 8001 (Metrics)

2. **aiplatform-mesh-worker** (5 replicas)
   - Distributed mesh nodes
   - High availability network
   - Autoscales to 50 pods

3. **aiplatform-3d-renderer** (2 replicas)
   - WebGL rendering server
   - Visualization layer
   - Port: 9000

### Services
- `aiplatform-api` - ClusterIP, internal access
- `aiplatform-mesh-worker` - Headless service for StatefulSet
- `aiplatform-3d-renderer` - ClusterIP for 3D rendering
- `redis` - Redis cache cluster
- `postgres` - PostgreSQL database

### Storage
- **postgres-pvc** - 100Gi (Database)
- **redis-pvc** - 50Gi (Cache)
- **app-data-pvc** - 200Gi (Application data)

### Autoscaling
- **API HPA**: 3-10 replicas (CPU 70%, Memory 80%)
- **Mesh HPA**: 5-50 replicas (CPU 75%)
- **3D HPA**: 2-8 replicas (CPU 70%)
- **VPA**: Automatic resource right-sizing

## Configuration

### Environment Variables
Set in deployment.yaml and configmap.yaml:
- `ENVIRONMENT=production`
- `LOG_LEVEL=INFO`
- `WORKERS=4` (for API)
- `QUANTUM_BACKEND=qiskit`
- `MESH_MODE=worker` (for mesh pods)

### Resource Limits
API Pod:
- Request: 500m CPU, 512Mi memory
- Limit: 2 CPU, 2Gi memory

Quantum Worker:
- Request: 1 CPU, 2Gi memory
- Limit: 4 CPU, 4Gi memory

### Network Policies
- Restrict ingress to aiplatform namespace pods
- Restrict egress to DNS, internal pods, kube-system
- Allow ingress from ingress-nginx controller

## Monitoring & Observability

### Metrics
Prometheus scrapes metrics at:
- `aiplatform-api:8001/metrics`

Metrics include:
- HTTP request metrics
- Quantum execution time
- Mesh network latency
- Cache hit rates

### Logging
Logs written to:
- Console (DEBUG level)
- File (INFO level, JSON format)
- Rotate every 10MB, keep 5 backups

### Health Checks
- **Liveness Probe**: /health endpoint, 30s initial delay, 10s interval
- **Readiness Probe**: /ready endpoint, 10s initial delay, 5s interval

## Ingress Setup

### DNS Records
```
api.aiplatform.example.com          -> ingress IP
mesh.aiplatform.example.com         -> ingress IP
3d.aiplatform.example.com           -> ingress IP
*.aiplatform.example.com            -> ingress IP (wildcard)
```

### SSL/TLS
- Automatic certificate generation via cert-manager
- Let's Encrypt issuer configured
- Auto-renewal before expiration

### Rate Limiting
- Default: 100 requests per second
- Configurable in ingress annotations

## Data Persistence

### Database
- PostgreSQL 15 Alpine
- 100Gi persistent storage
- Automatic backups recommended
- Consider managed RDS in production

### Cache
- Redis 7 Alpine
- 50Gi persistent storage
- LRU eviction policy
- 16 databases

### Application Data
- NFS mount for shared data
- 200Gi capacity
- ReadWriteMany access

## Security

### RBAC
- ServiceAccount: aiplatform-sa
- ClusterRole with minimal required permissions
- Pod Security Policy enforced

### Network Security
- Network policies restrict traffic
- Internal services (ClusterIP)
- Ingress only from nginx controller
- mTLS ready (TLS secrets available)

### Pod Security
- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Drop all capabilities

### Secrets Management
- Use sealed-secrets for production
- Never commit secrets.yaml to git
- Rotate credentials regularly
- Use AWS Secrets Manager / Azure Key Vault

## Scaling

### Horizontal Scaling
```bash
# Scale deployment manually
kubectl scale deployment aiplatform-api -n aiplatform --replicas=5

# HPA will automatically adjust based on metrics
```

### Vertical Scaling
```bash
# VPA recommends resource changes
kubectl describe vpa aiplatform-api-vpa -n aiplatform
```

### Database Scaling
```bash
# Expand PVC size
kubectl patch pvc postgres-pvc -n aiplatform -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

## Troubleshooting

### Pod Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n aiplatform

# Check events
kubectl get events -n aiplatform

# Check logs
kubectl logs <pod-name> -n aiplatform
```

### Performance Issues
```bash
# Check resource usage
kubectl top pod -n aiplatform
kubectl top node

# Check HPA status
kubectl get hpa -n aiplatform -w

# Check metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/aiplatform/pods/*/cpu_usage
```

### Network Issues
```bash
# Check services
kubectl get svc -n aiplatform

# Test connectivity
kubectl run -it --rm debug --image=busybox:1.28 -- sh
# nslookup aiplatform-api.aiplatform.svc.cluster.local
# curl http://aiplatform-api:8000/health
```

## Production Checklist

- [ ] Secrets properly configured and encrypted
- [ ] Storage classes appropriate for cloud provider
- [ ] Ingress domain configured with DNS
- [ ] SSL certificates generated and valid
- [ ] Pod disruption budgets configured
- [ ] Network policies enforced
- [ ] RBAC properly configured
- [ ] Resource requests/limits set appropriately
- [ ] HPA thresholds tuned for workload
- [ ] Monitoring and alerting configured
- [ ] Database backups automated
- [ ] Log aggregation configured
- [ ] Disaster recovery plan created
- [ ] Load testing completed
- [ ] Security scanning enabled

## Cloud Provider Specifics

### AWS EKS
- Update storage class to use gp3 volumes
- Use IAM roles for pod service accounts
- CloudWatch for logging
- Application Load Balancer for ingress

### Azure AKS
- Update storage class to use Azure Disk
- Use Azure Key Vault for secrets
- Azure Monitor for metrics
- Azure Application Gateway for ingress

### GCP GKE
- Update storage class to use pd-ssd
- Use Workload Identity for authentication
- Stackdriver for logging/monitoring
- Google Cloud Load Balancer for ingress

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Charts](https://helm.sh/) - Templating for K8s
- [Kustomize](https://kustomize.io/) - K8s customization
- [Prometheus Operator](https://prometheus-operator.dev/) - Monitoring
- [Cert-manager](https://cert-manager.io/) - Certificate automation
- [ArgoCD](https://argo-cd.readthedocs.io/) - GitOps deployment

## Support

For issues or questions:
1. Check Kubernetes events: `kubectl get events -n aiplatform`
2. Review pod logs: `kubectl logs <pod-name> -n aiplatform`
3. Check deployment status: `kubectl rollout status deployment/<name> -n aiplatform`
4. Review this guide's troubleshooting section
5. Open an issue on GitHub
