# Google Cloud Platform Deployment Guide
# AI Platform SDK - GCP

## Overview

Complete guide for deploying AI Platform SDK to Google Cloud Platform using Terraform and GCP services.

## Architecture

```
┌─────────────────────────────────────┐
│      GCP Project (us-central1)      │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Google Kubernetes Engine   │   │
│  │  (aiplatform-gke)           │   │
│  │  - 3-10 nodes               │   │
│  │  - Auto-scaling enabled     │   │
│  │  - Workload Identity        │   │
│  └─────────────────────────────┘   │
│         ↓                           │
│  ┌─────────────────────────────┐   │
│  │  Cloud SQL for PostgreSQL   │   │
│  │  - HA instance              │   │
│  │  - Automated backups        │   │
│  │  - Point-in-time recovery   │   │
│  └─────────────────────────────┘   │
│         ↓                           │
│  ┌─────────────────────────────┐   │
│  │  Memorystore for Redis      │   │
│  │  - Standard tier            │   │
│  │  - Replication              │   │
│  └─────────────────────────────┘   │
│         ↓                           │
│  ┌─────────────────────────────┐   │
│  │  Storage & Registry         │   │
│  │  - Cloud Storage            │   │
│  │  - Artifact Registry        │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

## Prerequisites

1. **GCP Account** with active billing
2. **gcloud CLI** installed and configured
3. **Terraform** >= 1.0
4. **kubectl** installed
5. **Service account** with Editor role

## Quick Start

### 1. Install gcloud CLI
```bash
# macOS
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# Verify
gcloud --version
```

### 2. Authenticate
```bash
# Login
gcloud auth login

# Set project
gcloud config set project aiplatform-project

# Create project if needed
gcloud projects create aiplatform-project \
  --name="AI Platform Project"

# Enable billing
gcloud billing projects link aiplatform-project \
  --billing-account <billing-account-id>
```

### 3. Enable APIs
```bash
# Required APIs
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  cloudsql.googleapis.com \
  redis.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com \
  cloudkms.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com
```

### 4. Create Service Account
```bash
# Create service account
gcloud iam service-accounts create terraform \
  --display-name="Terraform Service Account"

# Grant Editor role
gcloud projects add-iam-policy-binding aiplatform-project \
  --member="serviceAccount:terraform@aiplatform-project.iam.gserviceaccount.com" \
  --role="roles/editor"

# Create key
gcloud iam service-accounts keys create ~/terraform-key.json \
  --iam-account=terraform@aiplatform-project.iam.gserviceaccount.com
```

### 5. Prepare Terraform
```bash
cd terraform/gcp/

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=~/terraform-key.json

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id              = "aiplatform-project"
region                  = "us-central1"
cluster_name            = "aiplatform-gke"
node_count              = 3
machine_type            = "n1-standard-4"
database_version        = "POSTGRES_15"
database_instance_class = "db-custom-2-8192"
redis_memory_size_gb    = 5
alert_email             = "your-email@example.com"
EOF
```

### 6. Deploy
```bash
# Initialize
terraform init

# Plan
terraform plan -out=tfplan

# Apply (15-20 minutes)
terraform apply tfplan

# Get outputs
terraform output
```

### 7. Configure kubectl
```bash
# Get credentials
gcloud container clusters get-credentials aiplatform-gke \
  --zone us-central1-a \
  --project aiplatform-project

# Verify
kubectl cluster-info
```

### 8. Deploy K8s Manifests
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

## GCP Services

### Google Kubernetes Engine (GKE)
- **Version**: 1.27 (auto-upgraded)
- **Nodes**: 3-10 per pool
- **Machine Type**: n1-standard-4 (4 vCPU, 15GB RAM)
- **Network**: Custom VPC
- **Logging**: Cloud Logging integrated
- **Monitoring**: Cloud Monitoring integrated
- **Cost**: ~$70/month (control plane) + compute

### Cloud SQL for PostgreSQL
- **Version**: PostgreSQL 15
- **Instance Class**: db-custom-2-8192 (2 vCPU, 8GB RAM)
- **Storage**: 32GB (auto-expanding)
- **Backup**: Daily + on-demand
- **HA**: Regional HA configuration
- **Encryption**: CMek supported
- **Cost**: ~$150-200/month

### Memorystore for Redis
- **Version**: 7.0
- **Memory**: 5GB
- **Tier**: Standard
- **Replication**: Enabled
- **Persistence**: RDB snapshots
- **Cost**: ~$100-150/month

### Cloud Storage
- **Bucket**: aiplatform-artifacts
- **Storage Class**: Standard
- **Redundancy**: Multi-region
- **Cost**: ~$0.02/GB/month

### Artifact Registry
- **Location**: us-central1
- **Format**: Docker
- **Storage**: 0.5GB free per month
- **Cost**: ~$0.005/GB/month

## Monitoring

### Cloud Logging
```bash
# View cluster logs
gcloud logging read "resource.type=k8s_cluster" --limit 10

# View pod logs
gcloud logging read "resource.type=k8s_pod" --limit 10

# Stream logs
gcloud logging read "resource.type=k8s_pod" --limit 0 --follow
```

### Cloud Monitoring
```bash
# List metrics
gcloud monitoring metrics-descriptors list

# Get metric data
gcloud monitoring time-series list \
  --filter='resource.type="k8s_node"'
```

### Cloud Trace (APM)
```bash
# Enable Cloud Trace
gcloud services enable cloudtrace.googleapis.com

# View traces
gcloud trace list
```

## Scaling

### Scale GKE Cluster
```bash
# Scale node pool
gcloud container clusters resize aiplatform-gke \
  --num-nodes 10 \
  --zone us-central1-a

# Or enable autoscaling
gcloud container clusters update aiplatform-gke \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --zone us-central1-a
```

### Scale Database
```bash
# Upgrade instance class
gcloud sql instances patch aiplatform-db \
  --tier=db-custom-4-16384

# Increase storage
gcloud sql instances patch aiplatform-db \
  --database-flags=cloudsql_iam_authentication=on
```

### Scale Redis
```bash
# Upgrade memory
gcloud redis instances upgrade aiplatform-redis \
  --size-gb=10 \
  --region=us-central1
```

## Cost Optimization

### Estimates
- GKE Control Plane: ~$70/month
- Compute (3x n1-standard-4): ~$400/month
- Cloud SQL: ~$200/month
- Memorystore Redis: ~$150/month
- Storage: ~$10/month
- **Total**: ~$830/month

### Reduce Costs
1. **Use Preemptible VMs** (save 70%)
   ```bash
   gcloud container node-pools create preemptible \
     --cluster=aiplatform-gke \
     --preemptible \
     --num-nodes=5
   ```

2. **Use E2 machines**: Cheaper than N1
3. **Use Committed Use Discounts**: 25-50% discount
4. **Use Autopilot**: Managed alternative to GKE

## High Availability

### Regional Cluster
```bash
# Create regional cluster (HA by default)
gcloud container clusters create aiplatform-gke \
  --region=us-central1 \
  --enable-stackdriver-kubernetes \
  --enable-ip-alias
```

### Database HA
```bash
# Update to HA config
gcloud sql instances patch aiplatform-db \
  --availability-type=REGIONAL
```

### Backup Strategy
```bash
# Automated backups enabled
# View backups
gcloud sql backups list --instance=aiplatform-db

# Create on-demand backup
gcloud sql backups create \
  --instance=aiplatform-db \
  --description="manual backup"

# Restore from backup
gcloud sql backups restore <backup-id> \
  --backup-instance=aiplatform-db
```

## Security

### VPC & Firewall
```bash
# Create firewall rule
gcloud compute firewall-rules create allow-api \
  --direction=INGRESS \
  --priority=1000 \
  --action=ALLOW \
  --rules=tcp:8000

# View rules
gcloud compute firewall-rules list
```

### Service Account & Workload Identity
```bash
# Create Kubernetes service account
kubectl create serviceaccount aiplatform -n aiplatform

# Link to Google service account
gcloud iam service-accounts add-iam-policy-binding \
  terraform@aiplatform-project.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:aiplatform-project.svc.id.goog[aiplatform/aiplatform]"
```

### Encryption
```bash
# Enable CSEK (Customer-Supplied Encryption Keys)
gcloud compute disks create encrypted-disk \
  --csek-key-file=~/key.json

# All services encrypted by default:
# - GKE: encryption at rest
# - Cloud SQL: TDE
# - Memorystore: at rest + in transit
# - Cloud Storage: AES-256
```

### Cloud KMS
```bash
# Create KMS key ring
gcloud kms keyrings create aiplatform-ring \
  --location=us-central1

# Create key
gcloud kms keys create aiplatform-key \
  --location=us-central1 \
  --keyring=aiplatform-ring \
  --purpose=encryption
```

## Troubleshooting

### Cluster Issues
```bash
# Check cluster status
gcloud container clusters describe aiplatform-gke \
  --zone=us-central1-a

# Check node pools
gcloud container node-pools list --cluster=aiplatform-gke

# View cluster logs
gcloud logging read "resource.type=k8s_cluster" --limit 10
```

### Database Issues
```bash
# Check database status
gcloud sql instances describe aiplatform-db

# View database logs
gcloud sql operations list --instance=aiplatform-db

# View slow query logs
gcloud logging read "resource.type=cloudsql_database" --limit 10
```

### Connectivity Issues
```bash
# Test from pod
kubectl run -it --rm debug --image=busybox -- sh

# Test DNS
nslookup kubernetes.default

# Test service
curl http://aiplatform-api:8000/health
```

## Cleanup

```bash
# Delete cluster
gcloud container clusters delete aiplatform-gke \
  --zone=us-central1-a

# Delete Cloud SQL instance
gcloud sql instances delete aiplatform-db

# Delete Memorystore instance
gcloud redis instances delete aiplatform-redis \
  --region=us-central1

# Delete Cloud Storage bucket
gsutil rm -r gs://aiplatform-artifacts-123456789

# Delete project (optional)
gcloud projects delete aiplatform-project
```

## Support & Resources

- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [Memorystore Documentation](https://cloud.google.com/memorystore/docs)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)

## License
Same as AI Platform SDK
