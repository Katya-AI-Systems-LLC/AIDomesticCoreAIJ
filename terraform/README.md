# AI Platform SDK - Terraform Infrastructure

Complete Infrastructure as Code for deploying AI Platform SDK to AWS using Terraform.

## Overview

This Terraform configuration creates a production-ready Kubernetes infrastructure on AWS with:
- **EKS Cluster** - Managed Kubernetes (1.27)
- **VPC** - Custom VPC with public/private subnets
- **RDS PostgreSQL** - Multi-AZ managed database
- **ElastiCache Redis** - Multi-AZ managed cache
- **ECR** - Docker image registry
- **S3** - Artifact storage
- **CloudWatch** - Logging and monitoring
- **SNS** - Alerting

## Directory Structure

```
terraform/
├── main.tf              # Main configuration (EKS, VPC, RDS, Redis, etc.)
├── variables.tf         # Input variables
├── outputs.tf          # Output values
├── providers.tf        # Provider configuration
├── terraform.tfvars    # Variable overrides (not committed)
├── aws/                # AWS-specific modules (optional)
├── azure/              # Azure-specific modules (optional)
├── gcp/                # GCP-specific modules (optional)
└── README.md           # This file
```

## Prerequisites

1. **Terraform** >= 1.0
   ```bash
   # Install Terraform
   # https://www.terraform.io/downloads
   ```

2. **AWS CLI** configured
   ```bash
   aws configure
   aws sts get-caller-identity  # Verify credentials
   ```

3. **kubectl** installed
   ```bash
   # Install kubectl
   # https://kubernetes.io/docs/tasks/tools/
   ```

4. **Helm** (optional, for additional deployments)
   ```bash
   # Install Helm
   # https://helm.sh/docs/intro/install/
   ```

## Quick Start

### 1. Initialize Terraform
```bash
cd terraform
terraform init
```

This will:
- Download required providers
- Initialize backend (S3 with DynamoDB lock)
- Setup .terraform directory

### 2. Create Variables File
```bash
# Copy and customize
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
vim terraform.tfvars
```

**Key variables:**
```hcl
aws_region  = "us-east-1"
cluster_name = "aiplatform-eks"
alert_email = "your-email@example.com"
```

### 3. Plan Deployment
```bash
terraform plan -out=tfplan
```

Review the changes before applying.

### 4. Apply Configuration
```bash
terraform apply tfplan
```

This will create:
- VPC with 6 subnets
- EKS cluster with 2 node groups
- RDS PostgreSQL instance
- ElastiCache Redis cluster
- ECR repository
- S3 bucket
- CloudWatch log groups
- SNS alert topic

**Estimated time: 15-20 minutes**

### 5. Configure kubectl
```bash
# Get kubeconfig
aws eks update-kubeconfig \
  --region us-east-1 \
  --name aiplatform-eks

# Verify
kubectl cluster-info
kubectl get nodes
```

### 6. Deploy Applications
```bash
# Deploy Kubernetes manifests
kubectl apply -f ../k8s/

# Check deployment status
kubectl get pods -n aiplatform
```

## Configuration

### Main Variables (terraform.tfvars)

```hcl
# AWS Configuration
aws_region    = "us-east-1"
environment   = "production"

# Cluster Configuration
cluster_name       = "aiplatform-eks"
kubernetes_version = "1.27"

# VPC Configuration
vpc_cidr             = "10.0.0.0/16"
public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
private_subnet_cidrs = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]

# Node Configuration
min_node_count     = 3
max_node_count     = 10
desired_node_count = 5
node_instance_types = ["t3.large", "t3.xlarge"]

# Database Configuration
database_name   = "aiplatform-db"
db_instance_class = "db.t3.large"
postgres_version = "15.3"

# Cache Configuration
redis_cluster_name = "aiplatform-redis"
redis_node_type    = "cache.r7g.large"

# Alerts
alert_email = "devops@example.com"
```

### Advanced Configuration

**Custom Node Groups:**
Edit `eks_managed_node_groups` in main.tf to add specialized node groups:
- GPU nodes for quantum computing
- Memory-optimized nodes for analytics
- Spot instances for cost savings

**Multi-Region Setup:**
Create `terraform/aws/us-west-2.tf` and duplicate main.tf with different region.

**Backend Configuration:**
By default uses S3 with DynamoDB lock. To use local backend:
1. Remove `backend "s3"` block from main.tf
2. Run `terraform init`

## Common Tasks

### Scale Cluster
```bash
# Increase node count
terraform apply -var="desired_node_count=10"

# Or update terraform.tfvars and apply
# desired_node_count = 10
# terraform apply
```

### Update Kubernetes Version
```bash
# Update variable
terraform apply -var="kubernetes_version=1.28"
```

### Increase RDS Storage
```bash
# Update variable
terraform apply -var="allocated_storage=200"
```

### Change Node Types
```bash
# Update variable
terraform apply -var='node_instance_types=["t3.2xlarge"]'
```

### Add Encryption Key
```bash
# KMS key is created automatically for ECR
# To add to RDS:
# terraform apply -var="storage_encrypted=true"
```

## Monitoring & Logging

### CloudWatch Logs
```bash
# View EKS cluster logs
aws logs tail /aws/eks/aiplatform-eks/cluster --follow

# View RDS logs
aws logs tail /aws/rds/instance/aiplatform-db --follow

# View Redis logs
aws logs tail /aws/elasticache/redis/aiplatform-redis --follow
```

### CloudWatch Alarms
```bash
# List alarms
aws cloudwatch describe-alarms

# Check specific alarm
aws cloudwatch describe-alarms --alarm-names aiplatform-eks-node-cpu
```

### EKS Metrics
```bash
# CPU utilization
kubectl top nodes
kubectl top pods -n aiplatform

# Pod events
kubectl get events -n aiplatform

# Describe cluster
kubectl describe cluster
```

## Security

### Network Security
- Private subnets for data layer
- Public subnets for load balancers
- Security groups restrict traffic
- Network ACLs provide additional layer

### Encryption
- S3 encryption at rest (AES-256)
- RDS encryption at rest (KMS)
- Redis encryption at rest (KMS)
- TLS for transport encryption
- Secrets Manager for password storage

### IAM & RBAC
- EKS service role with minimal permissions
- Node IAM roles with necessary permissions
- Kubernetes RBAC configured
- ConfigMap-based auth management

### Secrets Management
- Passwords stored in AWS Secrets Manager
- Database password rotated regularly
- Never commit secrets to git
- Use Sealed Secrets for Kubernetes

## Troubleshooting

### Provider Errors
```bash
# Validate terraform syntax
terraform validate

# Format terraform files
terraform fmt -recursive

# Check specific errors
terraform plan
```

### AWS Errors
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check IAM permissions
aws iam get-user

# Check service quotas
aws service-quotas list-service-quotas --service-code ec2
```

### Terraform State Issues
```bash
# Lock state
terraform force-unlock <LOCK_ID>

# Refresh state
terraform refresh

# Show current state
terraform show

# Import existing resource
terraform import aws_security_group.example sg-12345678
```

### EKS Issues
```bash
# Check cluster status
aws eks describe-cluster --name aiplatform-eks

# Check node groups
aws eks describe-nodegroup --cluster-name aiplatform-eks --nodegroup-name general

# View cluster logs
aws logs describe-log-groups --log-group-name-prefix /aws/eks
```

## Cost Optimization

### Strategies
1. **Use Spot Instances** for non-critical workloads (save 70%)
2. **Reserved Instances** for predictable workload (save 40%)
3. **Scaling** - Min size = 3, scale up/down based on demand
4. **Rightsize** - Use `terraform destroy` for unused environments

### Cost Monitoring
```bash
# Estimate cost
terraform plan -json | jq '.resource_changes[]'

# AWS Cost Explorer
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 --granularity DAILY --metrics "BlendedCost"

# Billing alerts
aws budgets create-budget --account-id $(aws sts get-caller-identity --query Account --output text)
```

## Destruction

### Destroy Infrastructure (Warning: Deletes Everything)
```bash
# Plan destruction
terraform plan -destroy

# Destroy
terraform destroy

# Or
terraform apply -destroy
```

**Before destroying:**
1. Backup data from RDS: `aws rds create-db-snapshot --db-instance-identifier aiplatform-db`
2. Export data from Redis
3. Download artifacts from S3

## Advanced: Multi-Cloud Setup

### AWS (Primary)
- Complete setup above

### Azure (Optional)
See `azure/` directory for Azure-specific Terraform

### GCP (Optional)
See `gcp/` directory for GCP-specific Terraform

## Maintenance

### Regular Tasks
- [ ] Review cloud costs monthly
- [ ] Update Kubernetes version quarterly
- [ ] Patch node AMIs monthly
- [ ] Rotate database credentials
- [ ] Audit IAM permissions
- [ ] Review security group rules
- [ ] Test disaster recovery

### Updates
```bash
# Check for provider updates
terraform init -upgrade

# Plan terraform version update
terraform version

# Update .terraform.lock.hcl
terraform lock validate
```

## Support & Documentation

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Kubernetes Provider](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## Best Practices

1. **Always use terraform.tfvars** - Don't commit secrets
2. **Use state locking** - Prevent concurrent modifications
3. **Enable versioning** - Track all changes
4. **Use workspaces** - Separate dev/staging/prod
5. **Monitor costs** - Use AWS Cost Explorer
6. **Test changes** - Use `terraform plan` first
7. **Document assumptions** - Add comments to code
8. **Use modules** - Keep code DRY
9. **Enforce policies** - Use Terraform Cloud/Enterprise
10. **Automate everything** - GitOps with Terraform

## License
Same as AI Platform SDK

## Contributing
See CONTRIBUTING.md in project root
