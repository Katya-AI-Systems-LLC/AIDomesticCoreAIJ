# Setup Instructions for Each CI/CD Platform

## Quick Start Summary

| Platform | Setup Time | Free Tier | Best For |
|----------|-----------|-----------|----------|
| **Jenkins** | 2-4 hours | Yes | Enterprise, Custom workflows |
| **CircleCI** | 15 mins | 6,000 min/mo | Small-medium teams, Docker |
| **Travis CI** | 10 mins | Open-source | GitHub projects |
| **Azure DevOps** | 30 mins | Unlimited (public) | Enterprise, Microsoft stack |

---

## Jenkins Setup (Self-Hosted)

### Step 1: Install Jenkins

```bash
# Option A: Docker (Recommended)
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts

# Option B: Linux/Mac
wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo apt-key add -
sudo sh -c 'echo deb https://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
sudo apt-get update
sudo apt-get install jenkins
sudo systemctl start jenkins

# Get initial password
docker logs jenkins 2>&1 | grep -A 5 "Please use the following password"
```

### Step 2: Initial Configuration

```
1. Go to http://localhost:8080
2. Enter initial password from logs
3. Select "Install suggested plugins"
4. Create admin user
5. Set Jenkins URL
```

### Step 3: Install Required Plugins

Manage Jenkins → Manage Plugins → Available:

```
- Pipeline
- Blue Ocean
- Docker Pipeline
- Kubernetes
- GitHub Integration
- Slack Notification
- Cobertura Plugin
- Performance Plugin
```

### Step 4: Add Credentials

Manage Jenkins → Manage Credentials → Global → Add Credentials:

```
1. Docker Registry
   Kind: Username with password
   Scope: Global
   Username: ${DOCKER_USERNAME}
   Password: ${DOCKER_PASSWORD}
   ID: docker-registry-credentials

2. Kubernetes Config
   Kind: Secret file
   Scope: Global
   File: kubeconfig content
   ID: kubeconfig

3. GitHub Token
   Kind: Secret text
   Scope: Global
   Secret: ${GITHUB_TOKEN}
   ID: github-credentials

4. Slack Webhook
   Kind: Secret text
   Scope: Global
   Secret: ${SLACK_WEBHOOK_URL}
   ID: slack-webhook
```

### Step 5: Create Pipeline Job

```
1. New Item
2. Enter job name: "aiplatform-pipeline"
3. Select: Pipeline
4. OK

Configuration:
  Pipeline section:
    Definition: Pipeline script from SCM
    SCM: Git
    Repository URL: https://github.com/yourorg/aiplatform.git
    Credentials: github-credentials
    Branch: */main
    Script Path: Jenkinsfile
    
5. Save → Build Now
```

### Step 6: GitHub Webhook

```
1. Go to GitHub repository
2. Settings → Webhooks → Add webhook
3. Payload URL: http://jenkins.example.com/github-webhook/
4. Content type: application/json
5. Let me select individual events:
   ✓ Push events
   ✓ Pull requests
6. Active: ✓
7. Add webhook
```

---

## CircleCI Setup (Cloud)

### Step 1: Create Account

1. Go to https://circleci.com/signup
2. Sign in with GitHub/GitLab
3. Grant permissions to access repositories

### Step 2: Select Repository

```
1. Dashboard → Projects
2. Find your repository
3. Click "Set Up Project"
4. Select "Existing config" (uses .circleci/config.yml)
5. Start building
```

### Step 3: Add Environment Variables

```
1. Project Settings → Environment Variables
2. Add the following:

Name: DOCKER_USERNAME
Value: your-docker-username
Add Variable

Name: DOCKER_PASSWORD
Value: your-docker-password (encrypted)
Add Variable

Name: KUBECONFIG_STAGING
Value: (base64-encoded kubeconfig)
Add Variable

Name: KUBECONFIG_PRODUCTION
Value: (base64-encoded kubeconfig)
Add Variable
```

### Step 4: Create Context (Optional)

```
Organization Settings → Contexts → Create Context
Name: docker-credentials

Add environment variables:
- DOCKER_USERNAME
- DOCKER_PASSWORD
```

### Step 5: Verify Configuration

```bash
# Install CircleCI CLI
curl https://raw.githubusercontent.com/CircleCI-Public/circleci-cli/master/install.sh | bash

# Validate config
circleci config validate .circleci/config.yml

# Process config
circleci config process .circleci/config.yml
```

### Step 6: Trigger Build

```
1. Push code to repository
2. CircleCI automatically starts build
3. View progress at https://app.circleci.com/
```

---

## Travis CI Setup (Cloud)

### Step 1: Create Account

1. Go to https://travis-ci.com/signin
2. Sign in with GitHub
3. Authorize Travis CI

### Step 2: Enable Repository

```
1. Go to Travis CI account
2. Click on account name → Settings
3. Find repository in the list
4. Toggle to ON
```

### Step 3: Add Environment Variables

```
1. Repository Settings
2. Add environment variables:

DOCKER_USERNAME = your-username
DOCKER_PASSWORD = your-password

KUBECONFIG_STAGING = (base64-encoded)
KUBECONFIG_PRODUCTION = (base64-encoded)
```

### Step 4: Encrypt Sensitive Values

```bash
# Install Travis gem
gem install travis

# Encrypt and add to .travis.yml
travis encrypt DOCKER_PASSWORD=mypassword --add

travis encrypt KUBECONFIG_STAGING=... --add

# This updates .travis.yml automatically
```

### Step 5: Verify Configuration

```bash
# Validate YAML
yamllint .travis.yml

# Check syntax
python -m yaml .travis.yml
```

### Step 6: Trigger Build

```
1. Push to GitHub repository
2. Build automatically starts
3. View at https://travis-ci.com/yourorg/aiplatform
```

---

## Azure DevOps Setup (Cloud)

### Step 1: Create Project

```
1. Go to https://dev.azure.com
2. Create new organization (if needed)
3. Create new project:
   Name: aiplatform
   Visibility: Private (or Public)
   Version control: Git
   Work item process: Agile
   Create
```

### Step 2: Create Pipeline

```
1. Pipelines → Create Pipeline
2. Select GitHub (or your VCS)
3. Authorize and select repository
4. Select "Existing Azure Pipelines YAML"
5. Select branch: main
6. Review pipeline configuration
7. Save and run
```

### Step 3: Create Service Connections

```
Project Settings → Service connections → Create service connection

1. Docker Registry
   Registry type: Docker Registry
   Registry URL: https://docker.io
   Username: ${DOCKER_USERNAME}
   Password: ${DOCKER_PASSWORD}
   Service connection name: docker-hub

2. Kubernetes
   Authentication method: Kubeconfig
   Kubeconfig: (paste content)
   Service connection name: staging-kubernetes
   (Repeat for production-kubernetes)

3. GitHub
   GitHub service connection: (for accessing repo)
```

### Step 4: Create Variable Groups

```
Pipelines → Library → Variable groups

1. Create "docker-vars"
   DOCKER_USERNAME = your-username
   DOCKER_PASSWORD = ✓ (mark as secret)
   
2. Create "kubernetes-vars"
   KUBECONFIG_STAGING = (base64-encoded)
   KUBECONFIG_PRODUCTION = (base64-encoded)
```

### Step 5: Create Environments

```
Pipelines → Environments

1. Create "staging"
   No approvals needed

2. Create "production"
   Approvals → Add approvers
   Add your DevOps team members
```

### Step 6: Verify Pipeline

```
1. Pipelines → Recent runs
2. Select your pipeline
3. Click "Run pipeline"
4. Select branch: main
5. Click "Run"
6. Monitor execution
```

---

## Environment Configuration

### Create Environment Files

```bash
# .env.local
DATABASE_URL=postgresql://user:pass@localhost:5432/aiplatform
REDIS_URL=redis://localhost:6379/0
API_PORT=8000

# .env.staging
DATABASE_URL=postgresql://user:pass@staging-db.example.com:5432/aiplatform
REDIS_URL=redis://staging-redis.example.com:6379/0
API_PORT=8000
API_ENV=staging

# .env.production
DATABASE_URL=postgresql://user:pass@prod-db.example.com:5432/aiplatform
REDIS_URL=redis://prod-redis.example.com:6379/0
API_PORT=8000
API_ENV=production
LOG_LEVEL=INFO
```

### Load Environment

```python
# config.py
import os
from dotenv import load_dotenv

env = os.getenv('API_ENV', 'local')
load_dotenv(f'.env.{env}')

DATABASE_URL = os.getenv('DATABASE_URL')
REDIS_URL = os.getenv('REDIS_URL')
API_PORT = os.getenv('API_PORT', 8000)
```

---

## Credentials Setup

### Docker Registry

```bash
# Create personal access token (Docker Hub)
# https://hub.docker.com/settings/security

# Or for private registry
docker login registry.example.com
cat ~/.docker/config.json | base64

# Add to CI/CD platform as:
DOCKER_USERNAME = your-username
DOCKER_PASSWORD = your-token-or-password
```

### Kubernetes Access

```bash
# Get kubeconfig
# From cluster provider dashboard or:
kubectl config view --raw > ~/.kube/config

# Encode for CI/CD
cat ~/.kube/config | base64 -w 0 > kubeconfig.b64

# Add to CI/CD platform:
KUBECONFIG_STAGING = (contents of kubeconfig.b64)
KUBECONFIG_PRODUCTION = (contents of kubeconfig.b64)
```

### GitHub Token

```bash
# Create personal access token
# https://github.com/settings/tokens

# Scopes needed:
# - repo (full control)
# - read:org
# - admin:org_hook

# Add to CI/CD:
GITHUB_TOKEN = ghp_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Slack Webhook

```bash
# Create incoming webhook
# https://api.slack.com/apps → Create New App → From scratch

# Or use existing workspace app
# https://api.slack.com/apps → select app → Incoming Webhooks

# Enable and create webhook URL

# Add to CI/CD:
SLACK_WEBHOOK_URL = https://hooks.slack.com/services/...
```

---

## Test Configuration Files

### conftest.py

```python
import pytest
import os
from dotenv import load_dotenv

# Load test environment
load_dotenv('.env.test')

@pytest.fixture
def test_db():
    """Provide test database connection"""
    # Setup
    yield db
    # Teardown

@pytest.fixture
def test_redis():
    """Provide test Redis connection"""
    # Setup
    yield redis
    # Teardown

@pytest.fixture
def test_client():
    """Provide test API client"""
    from app import create_app
    app = create_app('testing')
    return app.test_client()
```

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: unit tests
    integration: integration tests
    slow: slow tests
```

### tox.ini

```ini
[tox]
envlist = py39,py310,py311

[testenv]
deps = 
    pytest
    pytest-cov
    pytest-xdist
commands = 
    pytest tests/ --cov=aiplatform
```

---

## Docker Configuration

### .dockerignore

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv
.env
.git
.gitignore
.dockerignore
Dockerfile
docker-compose*.yml
.DS_Store
.idea
.vscode
tests/
.coverage
htmlcov/
```

### Dockerfile (Multi-stage)

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app
COPY . .

# Create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Security Best Practices

### 1. Secrets Rotation

```bash
# Rotate Docker credentials
# - Update token in Docker Hub
# - Update in all CI/CD platforms
# - Test deployments

# Rotate Kubernetes credentials
# - Create new service account
# - Update kubeconfig
# - Test access
```

### 2. Code Scanning

```bash
# Run locally before push
bandit -r aiplatform/
flake8 aiplatform/
safety check

# Or use pre-commit hook
echo '#!/bin/bash
bandit -r aiplatform/
' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 3. Container Scanning

```bash
# Scan before pushing
trivy image aiplatform:latest

# Scan in registry
# Most registries have built-in scanning
```

### 4. Audit Logging

Keep audit trail:
- Track all deployments
- Log who approved what
- Maintain deployment history
- Archive logs for 90+ days

---

## Monitoring and Alerts

### Slack Notifications

```yaml
# Get webhook from:
# https://api.slack.com/apps → Create App → Incoming Webhooks

# Add to CI/CD configuration for:
- Build started
- Build succeeded
- Build failed
- Deployment started
- Deployment completed
- Security scan failed
```

### Email Notifications

Configure for:
- Failed builds
- Failed deployments
- Security vulnerabilities

### Dashboard Monitoring

Use tools like:
- DataDog
- New Relic
- Prometheus + Grafana
- CloudWatch

---

## Troubleshooting

### Build Timeouts

**Increase timeout:**
- Jenkins: Set in job configuration or Jenkinsfile
- CircleCI: Add `timeout: 3600` in config.yml
- Travis CI: Add `timeout: 3600` in .travis.yml
- Azure DevOps: Add `timeoutInMinutes: 120` in pipeline

### Authentication Failures

```bash
# Test Docker login
echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME

# Test Kubernetes access
kubectl auth can-i get pods --as=system:serviceaccount:default:jenkins

# Test GitHub access
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### Storage Issues

- Clean workspace between builds
- Archive artifacts
- Use caching for dependencies
- Monitor disk usage

### Networking Issues

- Check firewall rules
- Verify DNS resolution
- Test connectivity to registries
- Check cluster network policies

---

## Next Steps

1. **Choose platform** based on requirements
2. **Follow setup steps** for chosen platform
3. **Configure credentials** securely
4. **Test with simple change** (commit to feature branch)
5. **Monitor build execution**
6. **Iterate and optimize**
