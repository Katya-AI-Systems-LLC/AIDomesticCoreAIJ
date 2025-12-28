terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }

  backend "s3" {
    bucket         = "aiplatform-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "aiplatform-sdk"
      ManagedBy   = "terraform"
      CreatedAt   = timestamp()
    }
  }
}

# Kubernetes Provider
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_auth.cluster.token

  experiments {
    manifest_resource = true
  }
}

# Helm Provider
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_auth.cluster.token
  }
}

# Get auth token for EKS
data "aws_eks_auth" "cluster" {
  name = module.eks.cluster_name
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "aiplatform-vpc"
  cidr = var.vpc_cidr

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  enable_vpn_gateway = false

  # For EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = "1"
  }

  tags = {
    Name = "aiplatform-vpc"
  }
}

# EKS Module
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = concat(module.vpc.private_subnets, module.vpc.public_subnets)

  # Cluster Security Group
  cluster_security_group_additional_rules = {
    egress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "egress"
      source_node_security_group = true
    }
  }

  # Node Groups
  eks_managed_node_groups = {
    general = {
      name           = "general-node-group"
      use_name_prefix = true
      
      capacity_type = "on-demand"
      
      min_size     = var.min_node_count
      max_size     = var.max_node_count
      desired_size = var.desired_node_count

      instance_types = var.node_instance_types

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            delete_on_termination = true
            encrypted             = true
          }
        }
      }

      labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      tags = {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/enabled"             = "true"
      }
    }

    quantum = {
      name           = "quantum-node-group"
      use_name_prefix = true
      
      capacity_type = "spot"
      
      min_size     = 2
      max_size     = 20
      desired_size = 5

      instance_types = ["t3.xlarge", "t3.2xlarge"]

      labels = {
        Environment = var.environment
        NodeGroup   = "quantum"
        Workload    = "compute-intensive"
      }

      taints = [{
        key    = "workload"
        value  = "quantum"
        effect = "NoSchedule"
      }]

      tags = {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
      }
    }
  }

  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/AdminRole"
      username = "admin"
      groups   = ["system:masters"]
    }
  ]

  aws_auth_users = [
    {
      userarn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/devops"
      username = "devops"
      groups   = ["system:masters"]
    }
  ]

  tags = {
    Cluster = var.cluster_name
  }
}

# RDS PostgreSQL
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = var.database_name

  engine               = "postgres"
  engine_version       = var.postgres_version
  family               = "postgres${split(".", var.postgres_version)[0]}"
  major_engine_version = split(".", var.postgres_version)[0]
  instance_class       = var.db_instance_class

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  storage_type          = "gp3"
  iops                  = 3000

  db_name  = "aiplatform"
  username = var.db_username
  password = random_password.db_password.result
  port     = 5432

  multi_az            = true
  publicly_accessible = false

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.rds.name

  backup_retention_period = 30
  backup_window           = "03:00-04:00"
  maintenance_window      = "mon:04:00-mon:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "${var.database_name}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  copy_tags_to_snapshot = true

  enable_http_endpoint            = false
  enable_cloudwatch_logs_exports  = ["postgresql"]
  cloudwatch_log_retention_in_days = 30

  deletion_protection = true

  tags = {
    Name = "aiplatform-database"
  }
}

# ElastiCache Redis
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.1"

  name              = var.redis_cluster_name
  engine            = "redis"
  engine_version    = "7.0"
  family            = "redis7"
  node_type         = var.redis_node_type
  num_cache_nodes   = 3
  parameter_group_name = "default.redis7"
  port              = 6379

  automatic_failover_enabled = true
  multi_az_enabled           = true

  security_group_ids = [aws_security_group.redis.id]
  subnet_group_name  = aws_elasticache_subnet_group.redis.name

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = true

  backup_retention_limit = 30
  snapshot_window        = "03:00-05:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  log_delivery_configuration = {
    slow_log = {
      cloudwatch_log_group      = aws_cloudwatch_log_group.redis_logs.name
      cloudwatch_log_group_arn  = "${aws_cloudwatch_log_group.redis_logs.arn}:*"
      enabled                   = true
      log_format                = "json"
    }
  }

  tags = {
    Name = "aiplatform-redis"
  }
}

# S3 Bucket for artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = "aiplatform-artifacts-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "aiplatform-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ECR Repository for Docker images
resource "aws_ecr_repository" "aiplatform" {
  name                 = "aiplatform"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.ecr.arn
  }

  tags = {
    Name = "aiplatform-ecr"
  }
}

resource "aws_ecr_repository_policy" "aiplatform" {
  repository = aws_ecr_repository.aiplatform.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
      }
    ]
  })
}

# KMS Key for encryption
resource "aws_kms_key" "ecr" {
  description             = "KMS key for ECR encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "aiplatform-ecr-key"
  }
}

resource "aws_kms_alias" "ecr" {
  name          = "alias/aiplatform-ecr"
  target_key_id = aws_kms_key.ecr.key_id
}

# CloudWatch Logs
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = 30

  tags = {
    Name = "eks-cluster-logs"
  }
}

resource "aws_cloudwatch_log_group" "redis_logs" {
  name              = "/aws/elasticache/redis/${var.redis_cluster_name}"
  retention_in_days = 30

  tags = {
    Name = "redis-logs"
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# Random password for RDS
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Store secrets in AWS Secrets Manager
resource "aws_secretsmanager_secret" "db_password" {
  name = "aiplatform/database/password"
  recovery_window_in_days = 7

  tags = {
    Name = "aiplatform-db-password"
  }
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}

# Security Groups
resource "aws_security_group" "rds" {
  name        = "aiplatform-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "aiplatform-rds-sg"
  }
}

resource "aws_security_group" "redis" {
  name        = "aiplatform-redis-sg"
  description = "Security group for Redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "aiplatform-redis-sg"
  }
}

# Subnet Group for ElastiCache
resource "aws_elasticache_subnet_group" "redis" {
  name       = "aiplatform-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "aiplatform-redis-subnet-group"
  }
}

# Subnet Group for RDS
resource "aws_db_subnet_group" "rds" {
  name       = "aiplatform-rds-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "aiplatform-rds-subnet-group"
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "eks_node_cpu" {
  alarm_name          = "aiplatform-eks-node-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Alert when EKS node CPU exceeds 80%"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  alarm_name          = "aiplatform-rds-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Alert when RDS CPU exceeds 80%"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  dimensions = {
    DBInstanceIdentifier = module.rds.this_db_instance_id
  }
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "aiplatform-alerts"

  tags = {
    Name = "aiplatform-alerts"
  }
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
