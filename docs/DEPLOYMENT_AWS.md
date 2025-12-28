# AWS Deployment Guide
# AI Platform SDK - Amazon Web Services

## Overview

Complete guide for deploying AI Platform SDK to AWS using Terraform and AWS services.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      AWS Region (us-east-1)             │
│                                                          │
│  ┌─────────────────────────────────────────────────┐  │
│  │              VPC (10.0.0.0/16)                  │  │
│  │                                                  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │    EKS Cluster (aiplatform-eks)          │  │  │
│  │  │                                           │  │  │
│  │  │  • General Node Group (3-10 nodes)      │  │  │
│  │  │  • Quantum Node Group (2-20 nodes)      │  │  │
│  │  │  • Auto-scaling enabled                 │  │  │
│  │  │  • Multi-AZ deployment                  │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │                    ↓                           │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │    Data Tier                             │  │  │
│  │  │  • RDS PostgreSQL (Multi-AZ)             │  │  │
│  │  │  • ElastiCache Redis (3 nodes)           │  │  │
│  │  │  • Encrypted volumes (gp3, 100Gi)       │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │                                                  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │    Storage & Registry                    │  │  │
│  │  │  • S3 (Artifacts)                        │  │  │
│  │  │  • ECR (Docker images)                   │  │  │
│  │  │  • CloudWatch Logs                       │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │                                                  │  │
│  └─────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** >= 1.0
3. **AWS CLI** configured
4. **kubectl** installed
5. **IAM Permissions** for creating EKS, RDS, ElastiCache, VPC
6. **KMS Key** for encryption (created by Terraform)

## Deployment Steps

### 1. Setup AWS Credentials
```bash
# Configure AWS CLI
aws configure

# Verify access
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "AIDAI...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/..."
# }
```

### 2. Prepare Terraform
```bash
cd terraform/

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars

# Important variables:
# - aws_region = "us-east-1"
# - cluster_name = "aiplatform-eks"
# - alert_email = "your-email@example.com"
# - db_username = "postgres"
```

### 3. Initialize Terraform
```bash
# Download providers and setup backend
terraform init

# Output:
# Initializing the backend...
# Initializing provider plugins...
# Terraform has been successfully configured!
```

### 4. Plan Deployment
```bash
# Review all changes
terraform plan -out=tfplan

# Expected resources to create: ~50+ items
# - VPC with subnets, NAT gateways, route tables
# - EKS cluster with node groups
# - RDS PostgreSQL instance
# - ElastiCache Redis cluster
# - ECR repository
# - S3 bucket
# - CloudWatch logs
# - SNS topic for alerts
# - Security groups
# - IAM roles and policies
```

### 5. Deploy Infrastructure
```bash
# Apply configuration (15-20 minutes)
terraform apply tfplan

# Monitor progress
# Watch for "aws_eks_cluster.main" - this takes longest

# Expected output:
# Apply complete! Resources added: 50
# 
# Outputs:
# eks_cluster_name = "aiplatform-eks"
# eks_cluster_endpoint = "https://..."
# rds_endpoint = "aiplatform-db.c..."
# redis_endpoint = "aiplatform-redis...."
# ecr_repository_url = "123456789012.dkr.ecr.us-east-1.amazonaws.com/aiplatform"
```

### 6. Configure kubectl
```bash
# Get kubeconfig
aws eks update-kubeconfig \
  --region us-east-1 \
  --name aiplatform-eks

# Verify connection
kubectl cluster-info
kubectl get nodes
```

### 7. Deploy Kubernetes Manifests
```bash
# Navigate to K8s directory
cd ../k8s/

# Apply manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml      # Edit first!
kubectl apply -f pvc.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Verify
kubectl get pods -n aiplatform -w
```

### 8. Configure DNS & SSL
```bash
# Get Load Balancer IP/DNS
kubectl get ingress -n aiplatform

# Create Route53 records
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch file://route53-changes.json

# Or manually add DNS records:
# api.aiplatform.example.com     -> <INGRESS_IP>
# mesh.aiplatform.example.com    -> <INGRESS_IP>
# 3d.aiplatform.example.com      -> <INGRESS_IP>

# SSL certificates auto-issued via cert-manager
# Check status:
kubectl get certificate -n aiplatform
```

## AWS Services Usage

### VPC & Networking
- **VPC**: 10.0.0.0/16
- **Public Subnets**: 3 (10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24)
- **Private Subnets**: 3 (10.0.11.0/24, 10.0.12.0/24, 10.0.13.0/24)
- **NAT Gateways**: 3 (one per AZ for high availability)
- **Internet Gateway**: 1

### EKS Cluster
- **Version**: 1.27 (configurable)
- **Name**: aiplatform-eks
- **Endpoint**: Publicly accessible with restrictions
- **Logging**: CloudWatch Logs for cluster
- **RBAC**: Enabled with IAM integration

### Node Groups

**General Purpose** (for API and mesh)
- Instance types: t3.large, t3.xlarge
- Min: 3, Max: 10, Desired: 5
- Capacity: on-demand (reliable)
- Storage: 100Gi gp3 (3000 IOPS)

**Quantum Computing**
- Instance types: t3.xlarge, t3.2xlarge
- Min: 2, Max: 20, Desired: 5
- Capacity: spot (cost-effective)
- Storage: 100Gi gp3

### RDS PostgreSQL
- **Engine**: PostgreSQL 15.3
- **Instance Class**: db.t3.large
- **Storage**: 100Gi (auto-scales to 1TB)
- **Multi-AZ**: Yes (automatic failover)
- **Backup**: 30-day retention
- **Encryption**: KMS encryption at rest
- **Monitoring**: CloudWatch metrics, enhanced monitoring
- **Cost**: ~$150-300/month

### ElastiCache Redis
- **Engine**: Redis 7.0
- **Node Type**: cache.r7g.large (memory-optimized)
- **Nodes**: 3 (multi-AZ)
- **Replication**: Yes (automatic failover)
- **Encryption**: At rest + in transit
- **Backup**: Daily snapshots
- **Cost**: ~$100-200/month

### ECR Repository
- **Name**: aiplatform
- **Encryption**: KMS
- **Scan**: Image scanning on push
- **Retention**: Configurable
- **Cost**: ~$0.50/GB/month

### S3 Bucket
- **Name**: aiplatform-artifacts-{account-id}
- **Encryption**: AES-256
- **Versioning**: Enabled
- **Public Access**: Blocked
- **Lifecycle**: Configurable
- **Cost**: ~$0.023/GB/month

### CloudWatch
- **Logs**: Cluster, RDS, Redis
- **Metrics**: Custom metrics, alarms
- **Retention**: 30 days
- **Cost**: ~$10-50/month

## Accessing the Application

### Via Load Balancer
```bash
# Get Load Balancer DNS
aws elbv2 describe-load-balancers \
  --query 'LoadBalancers[0].DNSName' \
  --output text

# Access via DNS
curl https://api.aiplatform.example.com/health
```

### Via kubectl Port Forward
```bash
# Local access (development)
kubectl port-forward svc/aiplatform-api 8000:8000 -n aiplatform
curl http://localhost:8000/health
```

### Via AWS Systems Manager Session Manager
```bash
# Access pod directly (debug)
aws ssm start-session --target <node-instance-id>
# Then access from within node
curl http://aiplatform-api:8000/health
```

## Monitoring & Logging

### CloudWatch Dashboards
```bash
# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name AIplatform \
  --dashboard-body file://dashboard.json
```

### CloudWatch Alarms
```bash
# List alarms
aws cloudwatch describe-alarms --alarm-name-prefix aiplatform

# Check alarm state
aws cloudwatch describe-alarms \
  --alarm-names aiplatform-eks-node-cpu
```

### View Logs
```bash
# EKS cluster logs
aws logs tail /aws/eks/aiplatform-eks/cluster --follow

# RDS logs
aws logs tail /aws/rds/instance/aiplatform-db --follow

# Filter by timestamp
aws logs tail /aws/eks/aiplatform-eks/cluster \
  --since 1h --follow
```

### Metrics
```bash
# CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 300 \
  --statistics Average
```

## Scaling

### Scale Kubernetes Nodes
```bash
# Via Terraform
terraform apply -var="desired_node_count=10"

# Or via AWS Console
# EKS > Clusters > aiplatform-eks > Compute > Edit node group
```

### Scale Database
```bash
# Increase storage (auto-scaling enabled)
# No action needed - scales automatically to 1TB max

# Upgrade instance class
terraform apply -var="db_instance_class=db.t3.xlarge"
```

### Scale Cache
```bash
# Increase replica count
terraform apply -var="redis_node_type=cache.r7g.xlarge"
```

## Cost Optimization

### Current Estimates
- EKS Cluster: ~$70/month (1x control plane)
- EC2 Nodes (5): ~$400/month (t3.large)
- RDS: ~$200/month (db.t3.large)
- Redis: ~$150/month (3x cache.r7g.large)
- S3: ~$10/month (minimal usage)
- NAT Gateway: ~$45/month (3x @ $15 ea)
- Data Transfer: ~$0-50/month
- **Total**: ~$875/month

### Cost Reduction Strategies
1. **Use Spot Instances**: Save 70% on compute
   ```bash
   terraform apply -var="capacity_type=spot"
   ```

2. **Use Smaller Instances**: t3.medium instead of t3.large
   ```bash
   terraform apply -var='node_instance_types=["t3.medium"]'
   ```

3. **Use Reserved Instances**: 1-year or 3-year commitments
   - Save 40% with 1-year commitment
   - Save 60% with 3-year commitment

4. **Right-size Database**: db.t3.micro for dev
   ```bash
   terraform apply -var="db_instance_class=db.t3.micro"
   ```

5. **Use Single NAT Gateway**: Save $30/month
   ```bash
   # Edit main.tf: single_nat_gateway = true
   ```

6. **Enable Auto-scaling**: Only pay for what you use

## High Availability

### Multi-AZ Deployment
- ✅ EKS cluster spans 3 AZs
- ✅ Node groups distributed across AZs
- ✅ RDS Multi-AZ (automatic failover)
- ✅ ElastiCache Multi-AZ
- ✅ NAT Gateway in each AZ

### Backup Strategy
```bash
# RDS automatic backups (30 days)
aws rds describe-db-instances --db-instance-identifier aiplatform-db

# Manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier aiplatform-db \
  --db-snapshot-identifier aiplatform-db-backup-$(date +%Y%m%d)

# Copy snapshot to another region
aws rds copy-db-snapshot \
  --source-db-snapshot-identifier arn:aws:rds:us-east-1:123456789012:snapshot:aiplatform-db-backup \
  --target-db-snapshot-identifier aiplatform-db-backup-us-west-2 \
  --region us-west-2
```

## Security

### Network Security
- Security groups restrict traffic
- Network ACLs allow specific ports
- Private subnets for data layer
- No public access to database/cache

### Encryption
- All storage encrypted (KMS)
- TLS for data in transit
- Secrets in AWS Secrets Manager

### IAM Security
- Service roles with least privilege
- No root account usage
- MFA enabled for humans
- Regular access reviews

### Compliance
- CloudTrail for audit logs
- VPC Flow Logs for network monitoring
- Config for compliance tracking
- GuardDuty for threat detection

## Disaster Recovery

### RTO & RPO
- **RTO** (Recovery Time Objective): < 5 minutes
- **RPO** (Recovery Point Objective): < 1 hour

### Backup Verification
```bash
# List backups
aws rds describe-db-snapshots --query 'DBSnapshots[].{Id:DBSnapshotIdentifier,Created:SnapshotCreateTime}'

# Restore from snapshot (test)
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier aiplatform-db-restore \
  --db-snapshot-identifier aiplatform-db-backup-20240101
```

### Failover Testing
```bash
# Force RDS failover
aws rds reboot-db-instance \
  --db-instance-identifier aiplatform-db \
  --force-failover

# Monitor failover
aws rds describe-db-instances \
  --db-instance-identifier aiplatform-db \
  --query 'DBInstances[0].DBInstanceStatus'
```

## Troubleshooting

### Cluster Issues
```bash
# Check cluster status
aws eks describe-cluster --name aiplatform-eks

# Check node groups
aws eks describe-nodegroup \
  --cluster-name aiplatform-eks \
  --nodegroup-name general

# Check events
kubectl get events -n aiplatform -w
```

### Database Issues
```bash
# Check RDS status
aws rds describe-db-instances --db-instance-identifier aiplatform-db

# Check parameter groups
aws rds describe-db-parameters \
  --db-instance-identifier aiplatform-db

# View RDS logs
aws logs tail /aws/rds/instance/aiplatform-db --follow
```

### Networking Issues
```bash
# Check security groups
aws ec2 describe-security-groups \
  --filters Name=group-name,Values=aiplatform-*

# Check route tables
aws ec2 describe-route-tables \
  --filters Name=vpc-id,Values=<VPC_ID>

# Check NAT gateway
aws ec2 describe-nat-gateways \
  --filter Name=vpc-id,Values=<VPC_ID>
```

## Cleanup

### Destroy Infrastructure
```bash
# WARNING: Destroys everything!

# Backup database first
aws rds create-db-snapshot \
  --db-instance-identifier aiplatform-db \
  --db-snapshot-identifier aiplatform-db-final-backup

# Destroy
terraform destroy

# Confirm
# Type 'yes' and press Enter

# Verify cleanup
aws eks describe-clusters --region us-east-1
```

## Cost Tracking

### AWS Cost Explorer
```bash
# View costs by service
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE
```

### Set Budget Alerts
```bash
# Create budget
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget file://budget.json
```

## Support & Documentation

- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [AWS ElastiCache Documentation](https://docs.aws.amazon.com/elasticache/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

## Next Steps

1. [ ] Deploy infrastructure using Terraform
2. [ ] Configure kubectl and verify cluster access
3. [ ] Deploy Kubernetes manifests
4. [ ] Configure DNS and SSL
5. [ ] Setup CloudWatch dashboards
6. [ ] Configure SNS alerts
7. [ ] Test database backups
8. [ ] Test failover scenarios
9. [ ] Document runbooks
10. [ ] Setup cost monitoring

## License
Same as AI Platform SDK
