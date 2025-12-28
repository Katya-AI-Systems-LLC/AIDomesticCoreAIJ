# API Examples - AI Platform

Complete code examples for using the AI Platform API in various languages.

## Table of Contents

- [Python Examples](#python-examples)
- [JavaScript/Node.js Examples](#javascript-nodejs-examples)
- [cURL Examples](#curl-examples)
- [Real-World Use Cases](#real-world-use-cases)

## Python Examples

### Installation

```bash
pip install requests aiohttp pydantic
```

### Basic Health Check

```python
import requests

# Simple health check
response = requests.get(
    'https://api.aiplatform.io/v1/health',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

print(response.json())
# Output: {'status': 'healthy', 'timestamp': '...', 'version': '1.0.0'}
```

### Quantum Optimization Example

```python
import requests
import json
import time

class AIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.aiplatform.io/v1'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def submit_optimization(self, problem_type, num_qubits, **kwargs):
        """Submit quantum optimization problem"""
        payload = {
            'problem_type': problem_type,
            'num_qubits': num_qubits,
            **kwargs
        }
        
        response = requests.post(
            f'{self.base_url}/optimize',
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 202:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def get_optimization_result(self, job_id):
        """Get optimization result"""
        response = requests.get(
            f'{self.base_url}/optimize/{job_id}',
            headers=self.headers
        )
        return response.json()
    
    def wait_for_completion(self, job_id, max_wait=300, check_interval=5):
        """Wait for job to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            result = self.get_optimization_result(job_id)
            
            if result['status'] in ['completed', 'failed']:
                return result
            
            print(f"Job {job_id} status: {result['status']}")
            time.sleep(check_interval)
        
        raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")


# Usage
client = AIClient('YOUR_API_KEY')

# Submit MAXCUT problem
job = client.submit_optimization(
    problem_type='MAXCUT',
    num_qubits=4,
    graph={
        'nodes': [0, 1, 2, 3],
        'edges': [[0, 1], [1, 2], [2, 3], [3, 0]]
    }
)

print(f"Submitted job: {job['job_id']}")

# Wait for completion
result = client.wait_for_completion(job['job_id'])
print(f"Solution: {result['result']['solution']}")
print(f"Objective value: {result['result']['objective_value']}")
```

### Vision Analysis Example

```python
import base64
import requests
from pathlib import Path

def analyze_image(image_path, analysis_type='detection'):
    """Analyze image using computer vision"""
    api_key = 'YOUR_API_KEY'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    payload = {
        'image': image_data,
        'analysis_type': analysis_type,
        'confidence_threshold': 0.5
    }
    
    response = requests.post(
        'https://api.aiplatform.io/v1/vision/analyze',
        headers=headers,
        json=payload
    )
    
    return response.json()


# Usage
result = analyze_image('/path/to/image.jpg', analysis_type='detection')

print(f"Detections: {result['detections']}")
for detection in result['detections']:
    print(f"- {detection['class_name']}: {detection['confidence']:.2%}")
```

### Batch Image Processing

```python
import requests
import json

def batch_analyze_images(image_urls, analysis_type='classification'):
    """Submit batch image analysis"""
    api_key = 'YOUR_API_KEY'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'images': image_urls,
        'analysis_type': analysis_type
    }
    
    response = requests.post(
        'https://api.aiplatform.io/v1/vision/batch',
        headers=headers,
        json=payload
    )
    
    job_data = response.json()
    job_id = job_data['job_id']
    
    # Poll for results
    while True:
        result_response = requests.get(
            f'https://api.aiplatform.io/v1/vision/batch/{job_id}',
            headers=headers
        )
        
        result_data = result_response.json()
        
        if result_data['status'] == 'completed':
            return result_data
        
        print(f"Progress: {result_data['processed_images']}/{result_data['total_images']}")
        time.sleep(2)


# Usage
images = [
    'https://example.com/image1.jpg',
    'https://example.com/image2.jpg',
    'https://example.com/image3.jpg'
]

results = batch_analyze_images(images)
print(json.dumps(results, indent=2))
```

### Federated Learning Example

```python
import requests

def start_federated_training(model_name, num_clients, epochs):
    """Start federated learning job"""
    api_key = 'YOUR_API_KEY'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model_name': model_name,
        'num_clients': num_clients,
        'epochs': epochs,
        'privacy_budget': 1.0
    }
    
    response = requests.post(
        'https://api.aiplatform.io/v1/federated/train',
        headers=headers,
        json=payload
    )
    
    return response.json()


# Usage
training_job = start_federated_training(
    model_name='federated-classifier',
    num_clients=10,
    epochs=5
)

print(f"Training job started: {training_job['job_id']}")
```

### Async Operations with AsyncIO

```python
import aiohttp
import asyncio

class AsyncAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.aiplatform.io/v1'
    
    async def health_check(self):
        """Check service health"""
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            async with session.get(
                f'{self.base_url}/health',
                headers=headers
            ) as resp:
                return await resp.json()
    
    async def submit_optimization(self, problem_type, num_qubits):
        """Submit optimization asynchronously"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'problem_type': problem_type,
                'num_qubits': num_qubits
            }
            
            async with session.post(
                f'{self.base_url}/optimize',
                headers=headers,
                json=payload
            ) as resp:
                return await resp.json()


# Usage
async def main():
    client = AsyncAIClient('YOUR_API_KEY')
    
    # Check health
    health = await client.health_check()
    print(f"Service status: {health['status']}")
    
    # Submit optimization
    job = await client.submit_optimization('MAXCUT', 4)
    print(f"Job ID: {job['job_id']}")

asyncio.run(main())
```

## JavaScript/Node.js Examples

### Installation

```bash
npm install axios dotenv
```

### Basic Health Check

```javascript
const axios = require('axios');

async function checkHealth() {
    try {
        const response = await axios.get(
            'https://api.aiplatform.io/v1/health',
            {
                headers: {
                    'Authorization': `Bearer ${process.env.API_KEY}`
                }
            }
        );
        
        console.log(response.data);
    } catch (error) {
        console.error('Error:', error.message);
    }
}

checkHealth();
```

### Quantum Optimization Client

```javascript
const axios = require('axios');

class AIClient {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = 'https://api.aiplatform.io/v1';
        this.client = axios.create({
            baseURL: this.baseUrl,
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });
    }
    
    async submitOptimization(problemType, numQubits, options = {}) {
        try {
            const response = await this.client.post('/optimize', {
                problem_type: problemType,
                num_qubits: numQubits,
                ...options
            });
            return response.data;
        } catch (error) {
            throw new Error(`Optimization submission failed: ${error.message}`);
        }
    }
    
    async getResult(jobId) {
        try {
            const response = await this.client.get(`/optimize/${jobId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get result: ${error.message}`);
        }
    }
    
    async waitForCompletion(jobId, maxWait = 300000, interval = 5000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            const result = await this.getResult(jobId);
            
            if (['completed', 'failed'].includes(result.status)) {
                return result;
            }
            
            console.log(`Job ${jobId} status: ${result.status}`);
            await new Promise(resolve => setTimeout(resolve, interval));
        }
        
        throw new Error(`Job ${jobId} did not complete within ${maxWait}ms`);
    }
}

// Usage
const client = new AIClient(process.env.API_KEY);

(async () => {
    const job = await client.submitOptimization('MAXCUT', 4, {
        graph: {
            nodes: [0, 1, 2, 3],
            edges: [[0, 1], [1, 2], [2, 3], [3, 0]]
        }
    });
    
    console.log(`Job submitted: ${job.job_id}`);
    
    const result = await client.waitForCompletion(job.job_id);
    console.log(`Solution: ${result.result.solution}`);
    console.log(`Value: ${result.result.objective_value}`);
})();
```

### Vision Analysis

```javascript
const axios = require('axios');
const fs = require('fs');
const path = require('path');

async function analyzeImage(imagePath, analysisType = 'detection') {
    const imageBuffer = fs.readFileSync(imagePath);
    const imageBase64 = imageBuffer.toString('base64');
    
    const response = await axios.post(
        'https://api.aiplatform.io/v1/vision/analyze',
        {
            image: imageBase64,
            analysis_type: analysisType,
            confidence_threshold: 0.5
        },
        {
            headers: {
                'Authorization': `Bearer ${process.env.API_KEY}`,
                'Content-Type': 'application/json'
            }
        }
    );
    
    return response.data;
}

// Usage
(async () => {
    const result = await analyzeImage('./image.jpg', 'detection');
    
    console.log('Detections:');
    result.detections.forEach(det => {
        console.log(`- ${det.class_name}: ${(det.confidence * 100).toFixed(1)}%`);
    });
})();
```

### GraphQL Query

```javascript
const axios = require('axios');

async function graphqlQuery(query, variables = {}) {
    const response = await axios.post(
        'https://api.aiplatform.io/graphql',
        {
            query,
            variables
        },
        {
            headers: {
                'Authorization': `Bearer ${process.env.API_KEY}`,
                'Content-Type': 'application/json'
            }
        }
    );
    
    return response.data;
}

// Usage
const query = `
    query {
        health {
            status
            version
        }
        
        optimizationJobs(status: COMPLETED, first: 5) {
            edges {
                node {
                    id
                    status
                    createdAt
                }
            }
        }
    }
`;

(async () => {
    const result = await graphqlQuery(query);
    console.log(JSON.stringify(result, null, 2));
})();
```

## cURL Examples

### Health Check

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.aiplatform.io/v1/health
```

### Submit Optimization

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

### Get Result

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.aiplatform.io/v1/optimize/550e8400-e29b-41d4-a716-446655440000
```

### Analyze Image

```bash
# First, encode image to base64
IMAGE_BASE64=$(base64 -i /path/to/image.jpg)

curl -X POST https://api.aiplatform.io/v1/vision/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$IMAGE_BASE64\",
    \"analysis_type\": \"detection\",
    \"confidence_threshold\": 0.5
  }"
```

### GraphQL Query with cURL

```bash
curl -X POST https://api.aiplatform.io/graphql \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ health { status version } }"
  }'
```

## Real-World Use Cases

### Use Case 1: Solve Traveling Salesman Problem

```python
import requests
import numpy as np

def solve_tsp(distance_matrix, num_cities):
    """Solve TSP using quantum optimization"""
    client = AIClient('YOUR_API_KEY')
    
    job = client.submit_optimization(
        problem_type='TSP',
        num_qubits=int(np.ceil(np.log2(num_cities))),
        distance_matrix=distance_matrix.tolist()
    )
    
    result = client.wait_for_completion(job['job_id'])
    return result['result']['solution']

# Example: 5 cities TSP
distance_matrix = np.array([
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0]
])

solution = solve_tsp(distance_matrix, 5)
print(f"Optimal tour: {solution}")
```

### Use Case 2: Batch Image Classification

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def classify_batch(image_urls, batch_size=100):
    """Classify large image batch"""
    all_results = []
    
    for i in range(0, len(image_urls), batch_size):
        batch = image_urls[i:i+batch_size]
        
        result = batch_analyze_images(batch, 'classification')
        all_results.extend(result['results'])
        
        print(f"Processed {i+len(batch)}/{len(image_urls)}")
    
    return all_results

# Usage
image_urls = [f'https://example.com/img_{i}.jpg' for i in range(1000)]
results = classify_batch(image_urls)
```

### Use Case 3: Privacy-Preserving Model Training

```javascript
async function trainFederatedModel(trainingConfig) {
    const client = new AIClient(process.env.API_KEY);
    
    const job = await client.submitFederatedTraining({
        model_name: trainingConfig.modelName,
        num_clients: trainingConfig.numClients,
        epochs: trainingConfig.epochs,
        privacy_budget: trainingConfig.privacyBudget
    });
    
    // Monitor training progress
    let lastRound = -1;
    
    while (true) {
        const status = await client.getTrainingStatus(job.job_id);
        
        if (status.round > lastRound) {
            console.log(
                `Round ${status.round}: accuracy=${status.accuracy.toFixed(4)}, ` +
                `loss=${status.loss.toFixed(4)}`
            );
            lastRound = status.round;
        }
        
        if (status.status === 'completed') {
            return status;
        }
        
        await new Promise(r => setTimeout(r, 5000));
    }
}
```

## Error Handling

### Python

```python
import requests

def robust_api_call(url, headers, data, max_retries=3):
    """Make API call with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Timeout, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
            else:
                raise
```

### JavaScript

```javascript
async function robustApiCall(fn, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            if (error.response?.status === 429) {
                const retryAfter = parseInt(error.response.headers['retry-after'] || 60);
                console.log(`Rate limited. Waiting ${retryAfter}s...`);
                await new Promise(r => setTimeout(r, retryAfter * 1000));
            } else if (attempt < maxRetries - 1) {
                console.log(`Error, retrying... (${attempt + 1}/${maxRetries})`);
                await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 1000));
            } else {
                throw error;
            }
        }
    }
}
```

## Testing

### Python Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def client():
    return AIClient('test-key')

def test_submit_optimization(client):
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 202
        mock_post.return_value.json.return_value = {
            'job_id': 'test-id',
            'status': 'pending'
        }
        
        result = client.submit_optimization('MAXCUT', 4)
        
        assert result['job_id'] == 'test-id'
        assert result['status'] == 'pending'
```

## Conclusion

These examples demonstrate the key features and patterns for using the AI Platform API. For more information, see the complete [API Guide](API_GUIDE.md).
