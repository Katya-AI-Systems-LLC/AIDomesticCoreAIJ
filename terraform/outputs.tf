output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "eks_cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "eks_node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.this_db_instance_endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.this_db_instance_port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.this_db_instance_name
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.redis.this_elasticache_cluster_address
}

output "redis_port" {
  description = "Redis cluster port"
  value       = module.redis.this_elasticache_cluster_port
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.aiplatform.repository_url
}

output "s3_artifacts_bucket" {
  description = "S3 artifacts bucket name"
  value       = aws_s3_bucket.artifacts.id
}

output "sns_alert_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.alerts.arn
}

output "cloudwatch_log_group_eks" {
  description = "CloudWatch log group for EKS"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "deploy_kubernetes_manifests" {
  description = "Command to deploy Kubernetes manifests"
  value       = "kubectl apply -f k8s/"
}

output "get_db_password" {
  description = "Retrieve RDS password from Secrets Manager"
  value       = "aws secretsmanager get-secret-value --secret-id aiplatform/database/password --region ${var.aws_region}"
}

output "jump_into_cluster" {
  description = "Verify cluster access"
  value       = "kubectl cluster-info"
}

output "deployment_summary" {
  description = "Deployment summary"
  value = {
    cluster_name     = module.eks.cluster_name
    cluster_endpoint = module.eks.cluster_endpoint
    region           = var.aws_region
    rds_database     = module.rds.this_db_instance_name
    redis_cluster    = var.redis_cluster_name
    ecr_repository   = aws_ecr_repository.aiplatform.repository_url
  }
}
