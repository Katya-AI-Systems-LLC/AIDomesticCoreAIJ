# Azure Deployment Guide
# AI Platform SDK - Microsoft Azure

## Overview

Complete guide for deploying AI Platform SDK to Microsoft Azure using Terraform and Azure services.

## Architecture

```
┌─────────────────────────────────────────┐
│      Azure Subscription (East US)       │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Resource Group: aiplatform-rg    │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Azure Kubernetes Service   │ │ │
│  │  │  (aiplatform-aks)           │ │ │
│  │  │  - 3 node pools             │ │ │
│  │  │  - Auto-scaling (3-10)      │ │ │
│  │  │  - Multi-AZ ready           │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Azure Database for         │ │ │
│  │  │  PostgreSQL (Flexible)      │ │ │
│  │  │  - Multi-zone redundancy    │ │ │
│  │  │  - Automated backups        │ │ │
│  │  │  - Encryption at rest       │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Azure Cache for Redis      │ │ │
│  │  │  - Premium tier             │ │ │
│  │  │  - Clustering enabled       │ │ │
│  │  │  - Geo-replication ready    │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Storage & Container        │ │ │
│  │  │  - Storage Account          │ │ │
│  │  │  - Container Registry       │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  └───────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

## Prerequisites

1. **Azure Account** with active subscription
2. **Azure CLI** installed and authenticated
3. **Terraform** >= 1.0
4. **kubectl** installed
5. **Appropriate IAM permissions** in Azure

## Quick Deployment

### 1. Install Azure CLI
```bash
# macOS
brew install azure-cli

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Windows
# Download from https://aka.ms/azurecli

# Verify
az --version
```

### 2. Authenticate
```bash
# Login to Azure
az login

# If multiple subscriptions
az account list
az account set --subscription <subscription-id>

# Verify access
az account show
```

### 3. Prepare Terraform
```bash
cd terraform/azure/

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
azure_region           = "East US"
resource_group_name    = "aiplatform-rg"
cluster_name           = "aiplatform-aks"
node_count             = 3
vm_size                = "Standard_D4s_v3"
kubernetes_version     = "1.27"
database_sku_name      = "Standard_B2s"
redis_sku_name         = "P1"
alert_email            = "your-email@example.com"
EOF
```

### 4. Deploy
```bash
# Initialize
terraform init

# Plan
terraform plan -out=tfplan

# Apply (10-15 minutes)
terraform apply tfplan

# Get outputs
terraform output
```

### 5. Configure kubectl
```bash
# Get credentials
az aks get-credentials \
  --resource-group aiplatform-rg \
  --name aiplatform-aks

# Verify
kubectl cluster-info
```

### 6. Deploy K8s Manifests
```bash
cd ../../k8s/
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f pvc.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
```

## Azure Services

### Azure Kubernetes Service (AKS)
- **Version**: 1.27 (configurable)
- **Node Pools**: 3 pools (system, compute, quantum)
- **Node Size**: Standard_D4s_v3 (4 vCPU, 16GB RAM)
- **Scaling**: 3-10 nodes per pool
- **Networking**: Azure CNI
- **Monitoring**: Azure Monitor enabled
- **Cost**: ~$70/month (control plane) + compute

### Azure Database for PostgreSQL - Flexible
- **Version**: 15
- **SKU**: Standard_B2s (2 vCPU, 4GB RAM)
- **Storage**: 32GB (auto-scaling to 1TB)
- **Backup**: 7-day retention
- **HA**: Zone-redundant
- **Encryption**: At rest + in transit
- **Cost**: ~$100-150/month

### Azure Cache for Redis
- **Version**: 7.0
- **SKU**: Premium P1 (1GB)
- **Clustering**: Enabled
- **Persistence**: RDB snapshots
- **Geo-replication**: Ready
- **Cost**: ~$150-200/month

### Container Registry (ACR)
- **SKU**: Standard
- **Storage**: Unlimited
- **Geo-replication**: Optional
- **Image scanning**: Enabled
- **Cost**: ~$5/month

### Storage Account
- **Type**: General Purpose v2
- **Redundancy**: Geo-redundant storage (GRS)
- **Access tier**: Hot
- **Cost**: ~$0.02/GB/month

## Monitoring & Logging

### Azure Monitor
```bash
# View cluster metrics
az monitor metrics list \
  --resource /subscriptions/<id>/resourceGroups/aiplatform-rg/providers/Microsoft.ContainerService/managedClusters/aiplatform-aks \
  --metric CpuUsagePercentage

# View container logs
az container logs \
  --resource-group aiplatform-rg \
  --name aiplatform-api
```

### Application Insights (Optional)
```bash
# Create Application Insights
az monitor app-insights component create \
  --resource-group aiplatform-rg \
  --app aiplatform-insights

# Get instrumentation key
az monitor app-insights component show \
  --resource-group aiplatform-rg \
  --app aiplatform-insights \
  --query instrumentationKey
```

## Scaling

### Scale Nodes
```bash
# Scale node pool
az aks scale \
  --resource-group aiplatform-rg \
  --name aiplatform-aks \
  --nodepool-name nodepool1 \
  --node-count 10
```

### Enable Auto-scaling
```bash
# Enable cluster autoscaler
az aks update \
  --resource-group aiplatform-rg \
  --name aiplatform-aks \
  --enable-cluster-autoscaling \
  --min-count 3 \
  --max-count 10
```

### Scale Database
```bash
# Upgrade SKU
az postgres flexible-server update \
  --resource-group aiplatform-rg \
  --name aiplatform-db \
  --sku-name Standard_D4s_v3
```

## Cost Optimization

### Estimates
- AKS Control Plane: ~$70/month
- Compute (3x Standard_D4s_v3): ~$400/month
- PostgreSQL: ~$150/month
- Redis: ~$200/month
- Storage: ~$10/month
- **Total**: ~$830/month

### Reduce Costs
1. **Use spot instances** (save 70%)
   ```bash
   az aks nodepool add \
     --resource-group aiplatform-rg \
     --cluster-name aiplatform-aks \
     --name spotnodepool \
     --priority Spot \
     --eviction-policy Delete
   ```

2. **Use smaller VMs**: Standard_B2s instead of Standard_D4s_v3
3. **Reserved instances**: 1-year or 3-year plans
4. **Use B-series VMs**: Burstable, cost-effective for variable workloads

## High Availability

### Multi-Zone
```bash
# Enable zone redundancy
az aks create \
  --resource-group aiplatform-rg \
  --name aiplatform-aks \
  --zones 1 2 3
```

### Database HA
```bash
# Enable high availability
az postgres flexible-server update \
  --resource-group aiplatform-rg \
  --name aiplatform-db \
  --high-availability Enabled
```

### Backup
```bash
# Create database backup
az postgres flexible-server backup create \
  --resource-group aiplatform-rg \
  --server-name aiplatform-db \
  --backup-name backup-$(date +%Y%m%d)
```

## Security

### Network Security
```bash
# Create network security group
az network nsg create \
  --resource-group aiplatform-rg \
  --name aiplatform-nsg

# Add rules
az network nsg rule create \
  --resource-group aiplatform-rg \
  --nsg-name aiplatform-nsg \
  --name allow-https \
  --priority 100 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes '*' \
  --source-port-ranges '*' \
  --destination-address-prefixes '*' \
  --destination-port-ranges 443
```

### Encryption
```bash
# All services encrypted by default:
# - AKS: encryption at rest
# - PostgreSQL: TDE (Transparent Data Encryption)
# - Redis: encryption at rest and in transit
# - Storage: AES-256
```

### Key Vault
```bash
# Create Azure Key Vault
az keyvault create \
  --resource-group aiplatform-rg \
  --name aiplatform-kv

# Store secret
az keyvault secret set \
  --vault-name aiplatform-kv \
  --name db-password \
  --value <password>

# Use in AKS
# Enable KeyVault CSI driver for pod secrets
```

## Troubleshooting

### Cluster Issues
```bash
# Check cluster status
az aks show \
  --resource-group aiplatform-rg \
  --name aiplatform-aks

# Get logs
az aks diagnostics-show \
  --resource-group aiplatform-rg \
  --name aiplatform-aks
```

### Node Pool Issues
```bash
# List node pools
az aks nodepool list \
  --resource-group aiplatform-rg \
  --cluster-name aiplatform-aks

# Check node health
kubectl get nodes
kubectl describe node <node-name>
```

### Database Issues
```bash
# Check DB status
az postgres flexible-server show \
  --resource-group aiplatform-rg \
  --name aiplatform-db

# View DB logs
az postgres flexible-server server-logs list \
  --resource-group aiplatform-rg \
  --server-name aiplatform-db
```

## Cleanup

```bash
# Delete resource group (deletes all resources)
az group delete \
  --name aiplatform-rg \
  --yes --no-wait
```

## Support & Resources

- [Azure Kubernetes Service Docs](https://docs.microsoft.com/azure/aks/)
- [Azure Database for PostgreSQL](https://docs.microsoft.com/azure/postgresql/)
- [Azure Cache for Redis](https://docs.microsoft.com/azure/azure-cache-for-redis/)
- [Azure CLI Reference](https://docs.microsoft.com/cli/azure/)

## License
Same as AI Platform SDK
