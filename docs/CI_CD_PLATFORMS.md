# CI/CD Platforms - Complete Guide

Comprehensive documentation for setting up and using Jenkins, CircleCI, Travis CI, and Azure DevOps pipelines for the AI Platform project.

## Table of Contents

1. [Overview](#overview)
2. [Platform Comparison](#platform-comparison)
3. [Jenkins Setup](#jenkins-setup)
4. [CircleCI Setup](#circleci-setup)
5. [Travis CI Setup](#travis-ci-setup)
6. [Azure DevOps Setup](#azure-devops-setup)
7. [Best Practices](#best-practices)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

## Overview

The AI Platform project includes complete CI/CD pipeline configurations for four major CI/CD platforms, enabling automated testing, building, security scanning, and deployment to production environments.

### Pipeline Stages

All platforms follow a consistent pipeline structure:

1. **Checkout** - Get latest code
2. **Lint & Format** - Code quality checks
3. **Security Scan** - Vulnerability and dependency checks
4. **Unit Tests** - Run test suite with coverage
5. **Integration Tests** - End-to-end testing
6. **Build Docker** - Create container image
7. **Container Scan** - Security scan of image
8. **Push Image** - Deploy to container registry
9. **Deploy Staging** - Deploy to staging environment
10. **Deploy Production** - Deploy to production (approval required)
11. **Performance Tests** - Load and stress testing

## Platform Comparison

### Jenkins

**Pros:**
- Self-hosted, complete control
- Highly customizable with plugins
- Large community and ecosystem
- Declarative and scripted pipelines
- Pipeline as Code (Jenkinsfile)

**Cons:**
- Requires infrastructure management
- Steep learning curve
- More setup and maintenance needed

**Best For:**
- Enterprise environments
- Complex, custom workflows
- Organizations wanting complete control

**Cost:**
- Free (open-source)
- Infrastructure costs only

**Setup Time:**
- 2-4 hours (fresh server)
- 30 minutes (existing Jenkins server)

---

### CircleCI

**Pros:**
- Cloud-hosted, no infrastructure
- Excellent Docker integration
- Fast, parallel builds
- Good free tier
- Simple YAML configuration
- Built-in orbs for reusable components

**Cons:**
- Less customization than Jenkins
- Limited free tier minutes
- Vendor lock-in

**Best For:**
- Small to medium teams
- Docker-heavy workflows
- Teams wanting minimal DevOps overhead

**Cost:**
- Free tier: 6,000 minutes/month
- Paid: $20-200/month depending on needs

**Setup Time:**
- 15 minutes (GitHub/GitLab connection)
- 30 minutes (full configuration)

---

### Travis CI

**Pros:**
- GitHub-native integration
- Simple .travis.yml format
- Supports multiple programming languages
- Good matrix builds
- Free for open-source

**Cons:**
- Recent pricing changes
- Less powerful than competitors
- Limited customization
- Smaller ecosystem

**Best For:**
- Open-source projects
- Simple build workflows
- GitHub-hosted projects

**Cost:**
- Free for open-source
- Paid plans: $69-129/month

**Setup Time:**
- 10 minutes (GitHub integration)
- 20 minutes (configuration)

---

### Azure DevOps

**Pros:**
- Integration with Microsoft ecosystem
- Powerful YAML pipelines
- Excellent for .NET projects
- Strong enterprise features
- Free tier for public repos

**Cons:**
- Steeper learning curve
- Complex UI
- Primarily enterprise-focused

**Best For:**
- Enterprise organizations
- Microsoft Stack projects
- Teams using Azure infrastructure

**Cost:**
- Free for public repos (unlimited minutes)
- Paid plans for private repos

**Setup Time:**
- 30 minutes (setup)
- 45 minutes (full configuration)

---

## Jenkins Setup

### Prerequisites

```bash
# Java 11+
java -version

# Docker (for agents)
docker --version

# Jenkins
# Download from https://jenkins.io/download/
```

### Installation

```bash
# Docker (recommended)
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts
```

### Initial Configuration

1. Access Jenkins at `http://localhost:8080`
2. Retrieve initial password:
   ```bash
   docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Install recommended plugins
4. Create admin user

### Required Plugins

```
Pipeline
Blue Ocean
Docker Pipeline
Kubernetes
Git
Slack Notification
Cobertura Plugin
Email Extension
Performance Plugin
```

Install via:
- Manage Jenkins > Manage Plugins > Available
- Search and install plugins

### Configure Credentials

1. Go to Manage Jenkins > Manage Credentials
2. Add credentials for:
   - Docker Registry (username/password)
   - Kubernetes (kubeconfig)
   - GitHub (personal access token)
   - Slack (webhook URL)

### Setup Pipeline

1. Create new Pipeline job
2. Configure pipeline from SCM
3. Repository URL: `https://github.com/yourorg/aiplatform`
4. Script Path: `Jenkinsfile`
5. Branch: `*/main`

### Run Pipeline

```bash
# Trigger manually
# Jenkins UI > Pipeline > Build Now

# Or trigger on git push
# Configure GitHub webhook:
# Settings > Webhooks > Add webhook
# Payload URL: http://jenkins.example.com/github-webhook/
# Content type: application/json
# Let me select individual events: Push events, Pull requests
```

### View Results

- Jenkins UI > Pipeline > Console Output
- Blue Ocean view for better visualization
- Test reports, coverage, artifacts

---

## CircleCI Setup

### Prerequisites

- GitHub or GitLab account
- CircleCI account (https://circleci.com)

### Initial Setup

1. Sign up at https://circleci.com
2. Choose GitHub/GitLab as VCS
3. Grant permissions to access repositories
4. Select repository

### Create Configuration

Configuration file: `.circleci/config.yml`

```yaml
version: 2.1
```

This is already provided in the repository.

### Environment Variables

1. Go to Project Settings > Environment Variables
2. Add variables:
   - `DOCKER_USERNAME` - Docker registry username
   - `DOCKER_PASSWORD` - Docker registry password (encrypted)
   - `KUBECONFIG_STAGING` - Base64-encoded kubeconfig
   - `KUBECONFIG_PRODUCTION` - Base64-encoded kubeconfig

### Secrets and Security

For sensitive values, use environment variables:

```bash
# Encrypt secrets
circleci config validate
circleci context create my-context
```

### Monitoring

1. Go to Pipelines tab
2. Click on workflow to see execution
3. View logs for each job
4. Check artifacts for test reports, coverage

### Webhooks

Automatically created when repository is connected.

---

## Travis CI Setup

### Prerequisites

- GitHub account
- Repository with `.travis.yml`

### Initial Setup

1. Go to https://travis-ci.com
2. Sign in with GitHub
3. Authorize Travis CI
4. Enable repository

### Configuration

File: `.travis.yml`

Already provided in repository.

### Environment Variables

1. Go to Repository Settings
2. Add environment variables:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `KUBECONFIG_STAGING` (encrypted)
   - `KUBECONFIG_PRODUCTION` (encrypted)

### Encrypt Secrets

```bash
# Install Travis CLI
gem install travis

# Encrypt value
travis encrypt DOCKER_PASSWORD=mypassword --add

# This adds encrypted value to .travis.yml
```

### Monitoring

1. Navigate to build history
2. Click on build number
3. View job logs
4. Check build status badge

### Status Badge

Add to README.md:

```markdown
[![Build Status](https://travis-ci.com/yourorg/aiplatform.svg?branch=main)](https://travis-ci.com/yourorg/aiplatform)
```

---

## Azure DevOps Setup

### Prerequisites

- Azure DevOps account (https://dev.azure.com)
- Azure account (optional, for infrastructure)

### Initial Setup

1. Create new project in Azure DevOps
2. Go to Pipelines > Create Pipeline
3. Select GitHub/GitLab/Azure Repos
4. Authorize and select repository
5. Select "Existing Azure Pipelines YAML"

### Configuration

File: `azure-pipelines.yml`

Already provided in repository.

### Service Connections

1. Project Settings > Service connections
2. Create new service connection:
   - Docker Registry (for pushing images)
   - Kubernetes (for deployments)
   - GitHub (for accessing repo)

### Environment Variables

1. Pipelines > Library > Secure files
2. Add secret files:
   - Kubeconfig for each environment
3. Pipelines > Library > Variable groups
4. Create variable group with:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD` (mark as secret)

### Pipeline Approvals

1. Pipelines > Environments
2. Create environments: dev, staging, production
3. Add approvers for production environment

### Monitoring

1. Go to Pipelines > Recent runs
2. Click on pipeline to view stages
3. Check logs for each job
4. View artifacts and test results

---

## Best Practices

### 1. Branching Strategy

Use Git Flow or trunk-based development:

```
main (production)
  ├── release/* (release branches)
  ├── hotfix/* (urgent fixes)
develop (staging)
  ├── feature/* (features)
  └── bugfix/* (bug fixes)
```

**Pipeline Behavior:**
- `main` branch → deploy to production (approval required)
- `develop` branch → deploy to staging
- `feature/*` → run tests only
- PR → run tests, security scans

### 2. Environment-Specific Configuration

Use environment variables for configuration:

```yaml
# .env.production
API_ENV=production
LOG_LEVEL=INFO
DATABASE_POOL_SIZE=20

# .env.staging
API_ENV=staging
LOG_LEVEL=DEBUG
DATABASE_POOL_SIZE=10
```

### 3. Secrets Management

**Never commit secrets!**

Options:
- Environment variables (CI/CD platform)
- Sealed Secrets (Kubernetes)
- HashiCorp Vault
- AWS Secrets Manager

### 4. Docker Best Practices

```dockerfile
# Use specific versions
FROM python:3.11-slim

# Don't run as root
RUN useradd -m appuser
USER appuser

# Multi-stage builds for smaller images
FROM python:3.11 as builder
RUN pip install -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib /usr/local/lib
```

### 5. Test Strategy

```python
# Unit tests
pytest tests/ -v --cov=aiplatform

# Integration tests
pytest tests/integration_tests.py -v

# Smoke tests (after deployment)
pytest tests/smoke_tests.py -v

# Health checks (production only)
pytest tests/health_check.py -v
```

### 6. Build Caching

**CircleCI:**
```yaml
save_cache:
  paths:
    - ~/.cache/pip
  key: v1-pip-{{ checksum "requirements.txt" }}
```

**Azure DevOps:**
```yaml
- task: CacheBeta@1
  inputs:
    key: 'pip | "$(Agent.OS)" | requirements.txt'
    path: '$(Pipeline.Workspace)/.cache/pip'
```

### 7. Artifact Management

```yaml
# Archive logs, reports, coverage
artifacts:
  - junit.xml
  - htmlcov/
  - bandit-report.json
  - trivy-report.json
```

### 8. Parallel Execution

Run independent jobs in parallel:
- Linting and security scans can run together
- Multiple test suites can run simultaneously
- Reduces total pipeline time

### 9. Notifications

Slack integration for alerts:

```yaml
# Jenkins
post {
  failure {
    slackSend(
      color: 'danger',
      message: "Build ${BUILD_NUMBER} failed"
    )
  }
}

# CircleCI
notify:
  webhooks:
    - url: https://hooks.slack.com/...

# Travis CI
notifications:
  slack: workspace:channel
```

### 10. Pipeline Monitoring

Monitor key metrics:
- Build success rate
- Average build time
- Test coverage trend
- Security scan results
- Deployment frequency
- Lead time for changes

---

## Security Considerations

### 1. Secrets Management

✅ DO:
- Use environment variables
- Encrypt secrets in transit
- Rotate credentials regularly
- Use short-lived tokens
- Audit secret access

❌ DON'T:
- Commit secrets to git
- Use simple passwords
- Share credentials via chat
- Log sensitive values
- Use same credentials everywhere

### 2. Code Security

```yaml
# Bandit for Python security
bandit -r aiplatform/ -f json -o bandit-report.json

# Dependency vulnerability check
safety check

# Container image scanning
trivy image --severity HIGH,CRITICAL
```

### 3. Supply Chain Security

```yaml
# Verify Docker image signatures
docker trust inspect --pretty image:tag

# SBOM (Software Bill of Materials)
syft aiplatform:tag -o json > sbom.json

# Check for unsigned commits
git verify-commit HEAD
```

### 4. Access Control

- Restrict who can trigger deployments
- Require approval for production
- Use RBAC in Kubernetes
- Audit all deployments
- Separate staging and production credentials

### 5. Network Security

```yaml
# Private container registry
docker push private.registry.com/aiplatform:latest

# VPN for Kubernetes access
# Only expose APIs through API Gateway

# Network policies in Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

---

## Troubleshooting

### Common Issues

#### 1. Build Fails with "Docker daemon not found"

**Jenkins:**
```bash
# Make sure Jenkins has Docker access
docker exec jenkins id

# Add jenkins user to docker group
docker exec jenkins usermod -aG docker jenkins
```

**Solution:**
- Use `docker:dind` service
- Or use Kubernetes executor instead of Docker

#### 2. Tests Timeout

**Increase timeout:**

```yaml
# Jenkins
timeout(time: 2, unit: 'HOURS')

# CircleCI
timeout: 3600 # seconds

# Travis
timeout: 3600 # seconds

# Azure DevOps
timeoutInMinutes: 120
```

#### 3. Kubectl Not Found in Pipeline

**Install kubectl:**

```yaml
# Jenkins
sh 'curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"'
sh 'chmod +x kubectl && mv kubectl /usr/local/bin/'

# CircleCI
- kubernetes/install-kubectl:
    kubectl-version: v1.27.0

# Azure DevOps
- task: KubernetesManifest@0
  inputs:
    kubernetesServiceConnection: 'my-cluster'
```

#### 4. Kubeconfig Not Found

**Encode kubeconfig:**

```bash
# Create base64-encoded kubeconfig
cat ~/.kube/config | base64 -w 0 > kubeconfig.b64

# Add to CI/CD platform environment variable
# Then in pipeline:
echo $KUBECONFIG_STAGING | base64 -d > ~/.kube/config
```

#### 5. Docker Login Fails

**Check credentials:**

```yaml
# Verify format
echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin registry.com

# Check registry URL
docker login docker.io  # for Docker Hub
docker login registry.example.com  # for private registry
```

#### 6. Storage and Caching Issues

**Clean workspace:**

```yaml
# Jenkins
cleanWs deleteDirs: true, patterns: [[pattern: 'venv/**', type: 'INCLUDE']]

# CircleCI
- run: rm -rf ~/.cache/pip

# Azure DevOps
- script: rm -rf $(Build.ArtifactStagingDirectory)/*
```

### Debug Mode

Enable verbose logging:

**Jenkins:**
```groovy
wrap([$class: 'BuildNameUpdater']) {
  // Pipeline steps
}
```

**CircleCI:**
```yaml
- run: set -x  # Enable verbose output
```

**Azure DevOps:**
```yaml
system.debug: true
```

### Logs and Artifacts

Always preserve logs:

```yaml
# Keep test results
junit allowEmptyResults: true, testResults: '**/junit.xml'

# Archive artifacts
archiveArtifacts artifacts: '*.json,*.xml,logs/**'

# Publish coverage
publishHTML(reportDir: 'htmlcov', reportFiles: 'index.html')
```

---

## Performance Optimization

### Reduce Build Time

1. **Parallel Jobs**: Run independent tasks simultaneously
2. **Caching**: Cache dependencies, build artifacts
3. **Layer Caching**: Use Docker layer caching
4. **Spot Instances**: Use cheaper compute resources
5. **Matrix Builds**: Test multiple versions simultaneously

### Monitoring

Track these metrics:
- P50, P95, P99 build times
- Success/failure rates
- Test execution time trends
- Deployment frequency
- Time to production

---

## Next Steps

1. **Choose platform** based on team needs
2. **Set up repository** with chosen CI/CD config
3. **Configure credentials** securely
4. **Test pipeline** with merge request
5. **Monitor builds** and optimize
6. **Scale** to additional projects

---

## Resources

### Jenkins
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Jenkinsfile Reference](https://www.jenkins.io/doc/book/pipeline/)
- [Jenkins Plugins](https://plugins.jenkins.io/)

### CircleCI
- [CircleCI Documentation](https://circleci.com/docs/)
- [Orbs](https://circleci.com/developer/orbs)
- [Config Reference](https://circleci.com/docs/configuration-reference/)

### Travis CI
- [Travis CI Documentation](https://docs.travis-ci.com/)
- [Building Guides](https://docs.travis-ci.com/user/languages/python/)
- [Deployment](https://docs.travis-ci.com/user/deployment)

### Azure DevOps
- [Azure Pipelines Documentation](https://learn.microsoft.com/azure/devops/pipelines/)
- [YAML Schema](https://learn.microsoft.com/azure/devops/pipelines/yaml-schema/)
- [Task Reference](https://learn.microsoft.com/azure/devops/pipelines/tasks/reference/)

---

## Support

For issues or questions:
1. Check platform documentation
2. Review pipeline logs
3. Check GitHub issues
4. Contact platform support
