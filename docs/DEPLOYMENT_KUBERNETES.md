# Kubernetes Deployment Guide
# AI Platform SDK - Complete K8s Deployment

## Overview

This guide covers deploying AI Platform SDK to Kubernetes (EKS, AKS, GKE, or self-managed).

## Prerequisites

1. **Kubernetes Cluster** (1.24+)
2. **kubectl** configured to access cluster
3. **Helm** (optional, for package management)
4. **StorageClass** configured for persistent volumes
5. **Ingress Controller** (nginx recommended)

## Quick Deployment (5 minutes)

```bash
# Navigate to k8s directory
cd k8s/

# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Setup secrets
cp secrets.yaml.template secrets.yaml
# Edit secrets.yaml with your actual values
kubectl apply -f secrets.yaml

# 3. Deploy application stack
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml

# 4. Setup ingress & autoscaling
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# 5. Verify
kubectl get all -n aiplatform
```

## Detailed Setup

### 1. Cluster Verification
```bash
# Check cluster status
kubectl cluster-info
kubectl get nodes
kubectl describe nodes

# Check available storage classes
kubectl get storageclasses
```

### 2. Namespace & RBAC
```bash
# Namespace created with:
kubectl apply -f namespace.yaml

# Includes:
# - aiplatform namespace
# - Network policies
# - Resource quotas
# - Limit ranges
```

### 3. Storage Setup
```bash
# Create persistent volumes
kubectl apply -f pvc.yaml

# Verify PVCs
kubectl get pvc -n aiplatform
kubectl describe pvc postgres-pvc -n aiplatform
```

### 4. Configuration
```bash
# ConfigMap with application settings
kubectl apply -f configmap.yaml

# View configuration
kubectl get configmap -n aiplatform
kubectl describe configmap aiplatform-config -n aiplatform
```

### 5. Secrets Management
```bash
# Copy and edit template
cp secrets.yaml.template secrets.yaml

# Edit with real credentials
nano secrets.yaml

# Apply secrets
kubectl apply -f secrets.yaml

# Verify (without seeing values)
kubectl get secrets -n aiplatform
kubectl describe secret aiplatform-secrets -n aiplatform
```

### 6. Deploy Services
```bash
# Internal services for pod communication
kubectl apply -f service.yaml

# Verify
kubectl get svc -n aiplatform
kubectl describe svc aiplatform-api -n aiplatform
```

### 7. Deploy Deployments
```bash
# Three deployments:
# - aiplatform-api (3 replicas)
# - aiplatform-mesh-worker (5 replicas)
# - aiplatform-3d-renderer (2 replicas)
kubectl apply -f deployment.yaml

# Monitor rollout
kubectl rollout status deployment/aiplatform-api -n aiplatform
kubectl rollout status deployment/aiplatform-mesh-worker -n aiplatform

# View pods
kubectl get pods -n aiplatform
kubectl describe pod <pod-name> -n aiplatform
```

### 8. External Access (Ingress)
```bash
# Prerequisites:
# - Ingress controller installed (nginx, traefik, etc.)
# - cert-manager for SSL
# - DNS configured

kubectl apply -f ingress.yaml

# Verify ingress
kubectl get ingress -n aiplatform
kubectl describe ingress aiplatform-ingress -n aiplatform

# Check certificate status
kubectl get certificate -n aiplatform
kubectl describe certificate aiplatform-cert -n aiplatform
```

### 9. Auto-scaling
```bash
# Deploy HPA (Horizontal Pod Autoscaler)
kubectl apply -f hpa.yaml

# Monitor autoscaling
kubectl get hpa -n aiplatform -w

# Check metrics
kubectl top pods -n aiplatform
kubectl top nodes
```

## Verification Checklist

```bash
# Namespace
kubectl get namespace | grep aiplatform

# Network Policy
kubectl get networkpolicies -n aiplatform

# Service Accounts & RBAC
kubectl get serviceaccount -n aiplatform
kubectl get rolebindings -n aiplatform

# Storage
kubectl get pvc -n aiplatform
kubectl get pv

# ConfigMaps & Secrets
kubectl get configmaps -n aiplatform
kubectl get secrets -n aiplatform

# Services
kubectl get services -n aiplatform

# Deployments & Pods
kubectl get deployments -n aiplatform
kubectl get pods -n aiplatform
kubectl get statefulsets -n aiplatform

# Ingress
kubectl get ingress -n aiplatform
kubectl get certificate -n aiplatform

# Auto-scaling
kubectl get hpa -n aiplatform
kubectl get vpa -n aiplatform
```

## Accessing the Application

### Via Ingress (Recommended)
```bash
# Get ingress IP
kubectl get ingress -n aiplatform

# Update /etc/hosts or DNS
api.aiplatform.example.com -> INGRESS_IP
mesh.aiplatform.example.com -> INGRESS_IP
3d.aiplatform.example.com -> INGRESS_IP

# Access application
curl https://api.aiplatform.example.com/health
```

### Via Port Forward (Development)
```bash
# Forward local port to pod
kubectl port-forward -n aiplatform svc/aiplatform-api 8000:8000

# Access
curl http://localhost:8000/health
```

### Via NodePort
```bash
# Change service type (for testing)
kubectl patch svc aiplatform-api -n aiplatform -p '{"spec":{"type":"NodePort"}}'

# Get NodePort
kubectl get svc -n aiplatform

# Access via node-ip:nodeport
curl http://<node-ip>:<node-port>/health
```

## Monitoring & Logging

### Pod Logs
```bash
# View pod logs
kubectl logs -n aiplatform <pod-name>

# Follow logs in real-time
kubectl logs -n aiplatform <pod-name> -f

# View logs from all pods in deployment
kubectl logs -n aiplatform -l app=aiplatform -f

# View previous pod logs (if crashed)
kubectl logs -n aiplatform <pod-name> --previous
```

### Metrics
```bash
# CPU and memory usage
kubectl top pods -n aiplatform
kubectl top nodes

# Detailed metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/aiplatform/pods
```

### Events
```bash
# View cluster events
kubectl get events -n aiplatform

# Watch events in real-time
kubectl get events -n aiplatform -w

# Describe resource for events
kubectl describe pod <pod-name> -n aiplatform
```

### Port-Forward to Monitoring
```bash
# Prometheus
kubectl port-forward -n aiplatform svc/prometheus 9090:9090
# Access: http://localhost:9090

# Grafana (if deployed)
kubectl port-forward -n aiplatform svc/grafana 3000:3000
# Access: http://localhost:3000
```

## Scaling

### Manual Scaling
```bash
# Scale deployment
kubectl scale deployment aiplatform-api -n aiplatform --replicas 5

# Scale multiple replicas
kubectl scale deployment aiplatform-mesh-worker -n aiplatform --replicas 20

# Check status
kubectl get replicas -n aiplatform
```

### Auto-scaling
```bash
# Check HPA status
kubectl get hpa -n aiplatform

# Watch HPA scaling
kubectl get hpa -n aiplatform -w

# Check scaling events
kubectl describe hpa aiplatform-api-hpa -n aiplatform
```

### Rolling Updates
```bash
# Update image
kubectl set image deployment/aiplatform-api \
  api=aiplatform/sdk:1.1.0 -n aiplatform

# Check rollout status
kubectl rollout status deployment/aiplatform-api -n aiplatform

# View rollout history
kubectl rollout history deployment/aiplatform-api -n aiplatform

# Rollback if needed
kubectl rollout undo deployment/aiplatform-api -n aiplatform
```

## Updates & Maintenance

### Update Application
```bash
# Method 1: kubectl set image
kubectl set image deployment/aiplatform-api \
  api=aiplatform/sdk:1.1.0 -n aiplatform

# Method 2: kubectl patch
kubectl patch deployment aiplatform-api -n aiplatform \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","image":"aiplatform/sdk:1.1.0"}]}}}}'

# Method 3: Edit manifest and apply
kubectl edit deployment aiplatform-api -n aiplatform
kubectl apply -f deployment.yaml
```

### Update Configuration
```bash
# Edit ConfigMap
kubectl edit configmap aiplatform-config -n aiplatform

# Restart pods to pick up new config
kubectl rollout restart deployment/aiplatform-api -n aiplatform
```

### Update Secrets
```bash
# Recreate secret
kubectl delete secret aiplatform-secrets -n aiplatform
kubectl apply -f secrets.yaml

# Restart pods
kubectl rollout restart deployment/aiplatform-api -n aiplatform
```

### Scale Database
```bash
# Check PVC size
kubectl get pvc postgres-pvc -n aiplatform

# Expand PVC (if storage class supports expansion)
kubectl patch pvc postgres-pvc -n aiplatform \
  -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'

# Verify expansion
kubectl get pvc postgres-pvc -n aiplatform
```

## Troubleshooting

### Pod Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n aiplatform

# Check logs
kubectl logs <pod-name> -n aiplatform

# Check for stuck init container
kubectl get events -n aiplatform

# Check resource constraints
kubectl describe resourcequota -n aiplatform
```

### Connectivity Issues
```bash
# Test DNS
kubectl run -it --rm debug --image=busybox:1.28 -- sh
nslookup aiplatform-api.aiplatform.svc.cluster.local

# Test service connectivity
curl http://aiplatform-api:8000/health

# Check network policy
kubectl get networkpolicies -n aiplatform
kubectl describe networkpolicy aiplatform-network-policy -n aiplatform
```

### Storage Issues
```bash
# Check PVC status
kubectl get pvc -n aiplatform

# Describe PVC for events
kubectl describe pvc postgres-pvc -n aiplatform

# Check node storage
kubectl get nodes
kubectl describe node <node-name>
```

### Performance Issues
```bash
# Check resource usage
kubectl top pods -n aiplatform
kubectl top nodes

# Check HPA status
kubectl get hpa -n aiplatform
kubectl describe hpa aiplatform-api-hpa -n aiplatform

# Check node resource allocation
kubectl describe node <node-name>
```

## Backup & Recovery

### Backup Database
```bash
# Backup PostgreSQL
kubectl exec -it -n aiplatform \
  $(kubectl get pod -n aiplatform -l app=postgres -o jsonpath='{.items[0].metadata.name}') \
  -- pg_dump -U postgres aiplatform > backup.sql

# Backup PVC
kubectl get pvc postgres-pvc -n aiplatform -o json > postgres-pvc-backup.json
```

### Backup Etcd (Cluster-level)
```bash
# On master node
sudo etcdctl snapshot save /tmp/etcd-backup.db

# Verify backup
sudo etcdctl snapshot status /tmp/etcd-backup.db
```

### Restore from Backup
```bash
# Restore database
kubectl exec -i -n aiplatform \
  $(kubectl get pod -n aiplatform -l app=postgres -o jsonpath='{.items[0].metadata.name}') \
  -- psql -U postgres aiplatform < backup.sql
```

## Production Checklist

- [ ] **Namespace**: Resources isolated
- [ ] **RBAC**: Least privilege access
- [ ] **Secrets**: Properly encrypted and rotated
- [ ] **Network Policy**: Traffic restricted
- [ ] **Storage**: Persistent volumes configured
- [ ] **Resource Limits**: Set and appropriate
- [ ] **Health Checks**: Liveness/readiness configured
- [ ] **Auto-scaling**: HPA/VPA configured
- [ ] **Monitoring**: Metrics collected
- [ ] **Logging**: Centralized logging setup
- [ ] **Backup**: Automated backup strategy
- [ ] **Disaster Recovery**: Plan tested
- [ ] **Security**: Pod security policies enforced
- [ ] **High Availability**: Multi-replica deployments
- [ ] **Ingress**: External access configured
- [ ] **TLS**: Certificates valid and renewed
- [ ] **Rate Limiting**: Configured in ingress
- [ ] **Auth**: JWT/API key validation
- [ ] **Cost Optimization**: Right-sizing verified
- [ ] **Documentation**: Runbooks created

## Common Commands Quick Reference

```bash
# View resources
kubectl get all -n aiplatform
kubectl get pods -n aiplatform -o wide
kubectl get svc -n aiplatform

# Describe for troubleshooting
kubectl describe pod <pod> -n aiplatform
kubectl describe deployment <deployment> -n aiplatform
kubectl describe node <node>

# Logs and debugging
kubectl logs <pod> -n aiplatform
kubectl logs <pod> -n aiplatform --previous
kubectl exec -it <pod> -n aiplatform -- bash

# Scaling and updates
kubectl scale deployment <deployment> --replicas=5 -n aiplatform
kubectl set image deployment/<deployment> <container>=<image> -n aiplatform
kubectl rollout restart deployment/<deployment> -n aiplatform

# Port forwarding
kubectl port-forward svc/<service> 8000:8000 -n aiplatform

# Edit resources
kubectl edit deployment <deployment> -n aiplatform
kubectl patch <resource> <name> -p '{"spec":{"image":"new"}}' -n aiplatform
kubectl apply -f <file> -n aiplatform
```

## Cloud Provider Specifics

### AWS EKS
- Update kubeconfig: `aws eks update-kubeconfig --name <cluster> --region <region>`
- LoadBalancer: Uses AWS NLB/ALB
- Storage: Use aws-ebs StorageClass
- IAM: Use IRSA (IAM Roles for Service Accounts)

### Azure AKS
- Update kubeconfig: `az aks get-credentials --name <cluster> --resource-group <group>`
- LoadBalancer: Uses Azure Load Balancer
- Storage: Use azure-disk or azure-file StorageClass
- Identity: Use AAD Pod Identity

### GCP GKE
- Update kubeconfig: `gcloud container clusters get-credentials <cluster>`
- LoadBalancer: Uses Google Cloud Load Balancer
- Storage: Use pd-standard or pd-ssd StorageClass
- Identity: Use Workload Identity

## Support & Resources

- [Kubernetes Official Docs](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- Project README.md
- GitHub Issues

## Next Steps

1. Verify all pods are running: `kubectl get pods -n aiplatform`
2. Test application: `kubectl port-forward svc/aiplatform-api 8000:8000 -n aiplatform`
3. Setup monitoring: Deploy Prometheus/Grafana
4. Configure logging: Setup ELK or Loki
5. Enable backups: Configure automated database backups
6. Test disaster recovery: Run recovery drill
7. Document runbooks: Operational procedures
8. Setup alerts: CloudWatch/Prometheus alerts

## License
Same as AI Platform SDK
