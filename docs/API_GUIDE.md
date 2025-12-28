# API Guide - AI Platform

## Overview

The AI Platform API provides access to quantum optimization, computer vision, federated learning, and machine learning inference capabilities. The API supports both REST (via OpenAPI) and GraphQL interfaces.

## Getting Started

### Prerequisites

- API key from your account settings
- Python 3.8+ or Node.js 14+ for client libraries
- Familiarity with HTTP/REST or GraphQL concepts

### Quick Setup

#### Authentication

All requests require a Bearer token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.aiplatform.io/v1/health
```

#### Python Example

```python
import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

response = requests.get(
    "https://api.aiplatform.io/v1/health",
    headers=headers
)
print(response.json())
```

#### JavaScript/Node.js Example

```javascript
const response = await fetch('https://api.aiplatform.io/v1/health', {
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  }
});

const data = await response.json();
console.log(data);
```

## REST API Guide

### Base URLs

- **Production**: `https://api.aiplatform.io/v1`
- **Staging**: `https://staging-api.aiplatform.io/v1`
- **Development**: `https://dev-api.aiplatform.io/v1`
- **Local**: `http://localhost:8000/v1`

### Request Format

All requests should use JSON:

```bash
curl -X POST https://api.aiplatform.io/v1/optimize \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"problem_type": "MAXCUT", "num_qubits": 4}'
```

### Response Format

Successful responses return JSON with status codes:

- `200 OK`: Request succeeded
- `201 Created`: Resource created
- `202 Accepted`: Async request submitted
- `204 No Content`: Success with no response body
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Response Headers

Every response includes rate limiting information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1705335600
```

## Health & Status Endpoints

### Health Check

Get quick service health status:

```bash
curl https://api.aiplatform.io/v1/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

### Detailed Status

Get detailed status of all service components:

```bash
curl https://api.aiplatform.io/v1/status
```

## Quantum Optimization

### Submit Optimization Job

Submit quantum optimization problems (MAXCUT, TSP, VRP, QAOA, QUBO):

```bash
curl -X POST https://api.aiplatform.io/v1/optimize \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "MAXCUT",
    "num_qubits": 4,
    "graph": {
      "nodes": [0, 1, 2, 3],
      "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]
    }
  }'
```

Response (202 Accepted):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

### Get Optimization Result

Retrieve results of a completed job:

```bash
curl https://api.aiplatform.io/v1/optimize/550e8400-e29b-41d4-a716-446655440000
```

### Cancel Optimization

Cancel a running or pending job:

```bash
curl -X POST https://api.aiplatform.io/v1/optimize/550e8400-e29b-41d4-a716-446655440000/cancel
```

## Vision Analysis

### Analyze Single Image

Perform computer vision analysis on a single image:

```bash
curl -X POST https://api.aiplatform.io/v1/vision/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "analysis_type": "detection",
    "confidence_threshold": 0.5
  }'
```

Supported analysis types:
- `detection` - Object detection with bounding boxes
- `classification` - Image classification
- `ocr` - Optical character recognition
- `scene_understanding` - Scene understanding and description

### Batch Image Analysis

Submit multiple images for batch processing:

```bash
curl -X POST https://api.aiplatform.io/v1/vision/batch \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["image_url_1", "image_url_2", "image_url_3"],
    "analysis_type": "classification"
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "total_images": 3,
  "processed_images": 0
}
```

### Get Batch Results

Retrieve results of batch processing:

```bash
curl https://api.aiplatform.io/v1/vision/batch/550e8400-e29b-41d4-a716-446655440000
```

## Federated Learning

### Start Training

Initiate privacy-preserving federated learning:

```bash
curl -X POST https://api.aiplatform.io/v1/federated/train \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-model",
    "num_clients": 10,
    "epochs": 5,
    "privacy_budget": 1.0
  }'
```

### Training Status

Get current training job status:

```bash
curl https://api.aiplatform.io/v1/federated/train/550e8400-e29b-41d4-a716-446655440000
```

## Inference

### Single Prediction

Run inference on a single input:

```bash
curl -X POST https://api.aiplatform.io/v1/infer/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "bert-base-uncased",
    "input": {
      "text": "Hello, world!"
    }
  }'
```

### Batch Inference

Submit batch inference job:

```bash
curl -X POST https://api.aiplatform.io/v1/infer/batch \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "bert-base-uncased",
    "inputs": [
      {"text": "Example 1"},
      {"text": "Example 2"}
    ]
  }'
```

## Model Management

### List Models

Get available models:

```bash
curl "https://api.aiplatform.io/v1/models?task_type=classification"
```

Query parameters:
- `task_type` - Filter by task (classification, detection, etc.)
- `framework` - Filter by framework (pytorch, tensorflow, etc.)

### Model Details

Get detailed information about a model:

```bash
curl https://api.aiplatform.io/v1/models/bert-base-uncased
```

## Project Management

### List Projects

```bash
curl https://api.aiplatform.io/v1/projects
```

### Create Project

```bash
curl -X POST https://api.aiplatform.io/v1/projects \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Project",
    "description": "Project description"
  }'
```

### Get Project

```bash
curl https://api.aiplatform.io/v1/projects/550e8400-e29b-41d4-a716-446655440000
```

### Update Project

```bash
curl -X PUT https://api.aiplatform.io/v1/projects/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Name",
    "description": "Updated description"
  }'
```

### Delete Project

```bash
curl -X DELETE https://api.aiplatform.io/v1/projects/550e8400-e29b-41d4-a716-446655440000
```

## GraphQL Guide

### GraphQL Endpoint

```
https://api.aiplatform.io/graphql
```

### Example Query

```graphql
query {
  health {
    status
    version
    timestamp
  }
  
  optimizationJobs(status: COMPLETED, first: 10) {
    edges {
      node {
        id
        status
        createdAt
      }
    }
  }
}
```

### Example Mutation

```graphql
mutation {
  submitOptimization(input: {
    problemType: MAXCUT
    numQubits: 4
    graph: {
      nodes: [0, 1, 2, 3]
      edges: [[0, 1], [1, 2], [2, 3], [3, 0]]
    }
  }) {
    id
    status
  }
}
```

### Subscriptions

Subscribe to real-time updates:

```graphql
subscription {
  optimizationProgress(jobId: "550e8400-e29b-41d4-a716-446655440000") {
    objectiveValue
    approximationRatio
  }
}
```

## Rate Limiting

### Tier Limits

| Tier | Requests/Hour | Concurrent | Cost |
|------|--------------|-----------|------|
| Standard | 1,000 | 10 | Standard |
| Premium | 10,000 | 100 | 20% discount |
| Enterprise | Unlimited | 1,000 | Custom |

### Cost Per Operation

High-cost operations:
- Quantum optimization: 10 tokens
- Batch analysis: 20 tokens
- Federated training: 15 tokens

Medium-cost operations:
- Vision analysis: 5 tokens
- Batch inference: 10 tokens

Low-cost operations:
- Health check: 0.1 tokens
- Model list: 0.5 tokens

### Handling Rate Limits

When rate limited (429 response):

```python
import time
import requests

response = requests.post(url, headers=headers, json=data)

if response.status_code == 429:
    retry_after = int(response.headers.get('Retry-After', 60))
    print(f"Rate limited. Waiting {retry_after} seconds...")
    time.sleep(retry_after)
    response = requests.post(url, headers=headers, json=data)
```

## Error Handling

### Error Response Format

```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "status_code": 429,
  "message": "Too many requests. Rate limit exceeded.",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "retry_after_seconds": 60
}
```

### Common Error Codes

- `INVALID_REQUEST` - Missing or invalid parameters
- `UNAUTHORIZED` - Missing or invalid authentication
- `FORBIDDEN` - Insufficient permissions
- `NOT_FOUND` - Resource not found
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `SERVICE_ERROR` - Internal server error

## Security Best Practices

1. **Never expose API keys** in client-side code
2. **Use HTTPS** for all requests
3. **Rotate keys** regularly (monthly recommended)
4. **Limit key scopes** to necessary endpoints
5. **Monitor usage** via the admin dashboard
6. **Use VPC endpoints** for enterprise deployments

## Pagination

List endpoints support cursor-based pagination:

```bash
curl "https://api.aiplatform.io/v1/projects?limit=20&offset=0"
```

Response includes pagination info:

```json
{
  "projects": [...],
  "total": 100,
  "limit": 20,
  "offset": 0
}
```

## Webhooks (Premium+)

Configure webhooks for async job completion:

```bash
curl -X POST https://api.aiplatform.io/v1/webhooks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "job.completed",
    "url": "https://example.com/webhook",
    "secret": "webhook_secret_key"
  }'
```

Webhook payload:

```json
{
  "event": "job.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "result": {}
  }
}
```

## Client Libraries

### Python

```bash
pip install aiplatform-sdk
```

```python
from aiplatform import Client

client = Client(api_key="YOUR_API_KEY")
result = client.optimize(problem_type="MAXCUT", num_qubits=4)
```

### JavaScript/Node.js

```bash
npm install aiplatform-sdk
```

```javascript
const { Client } = require('aiplatform-sdk');

const client = new Client({ apiKey: 'YOUR_API_KEY' });
const result = await client.optimize({
  problemType: 'MAXCUT',
  numQubits: 4
});
```

## Support

- **Documentation**: https://docs.aiplatform.io
- **Issues**: https://github.com/aiplatform-team/aiplatform/issues
- **Email**: support@aiplatform.io
- **Slack**: https://aiplatform.slack.com
- **Community**: https://community.aiplatform.io
