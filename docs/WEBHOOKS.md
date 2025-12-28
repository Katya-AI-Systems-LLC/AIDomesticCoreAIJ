# Webhooks Guide - AI Platform

## Overview

Webhooks allow your application to receive real-time notifications when events occur on the AI Platform. Instead of polling for job status, webhooks push notifications directly to your endpoint when events happen.

## Event Types

### Job Events

- `optimization.submitted` - Optimization job submitted
- `optimization.started` - Optimization job started processing
- `optimization.completed` - Optimization job completed
- `optimization.failed` - Optimization job failed
- `optimization.cancelled` - Optimization job cancelled

### Vision Events

- `vision.submitted` - Vision analysis submitted
- `vision.completed` - Vision analysis completed
- `vision.failed` - Vision analysis failed
- `batch.submitted` - Batch analysis submitted
- `batch.completed` - Batch analysis completed
- `batch.progress` - Batch analysis progress update
- `batch.failed` - Batch analysis failed

### Training Events

- `training.started` - Federated training started
- `training.progress` - Training progress update
- `training.completed` - Training completed
- `training.failed` - Training failed
- `training.cancelled` - Training cancelled

### Inference Events

- `inference.started` - Inference job started
- `inference.completed` - Inference completed
- `inference.failed` - Inference failed

### Model Events

- `model.deployed` - Model deployed to production
- `model.updated` - Model updated
- `model.deprecated` - Model deprecated

## Setting Up Webhooks

### Create Webhook Endpoint

First, create an endpoint on your server to receive webhook payloads:

**Python (Flask)**

```python
from flask import Flask, request
import hmac
import hashlib
import json

app = Flask(__name__)
WEBHOOK_SECRET = 'your-webhook-secret'

def verify_signature(payload, signature):
    """Verify webhook signature"""
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)

@app.route('/webhook', methods=['POST'])
def webhook():
    payload = request.get_data()
    signature = request.headers.get('X-Webhook-Signature')
    
    # Verify signature
    if not verify_signature(payload, signature):
        return {'error': 'Invalid signature'}, 401
    
    data = json.loads(payload)
    event_type = data['event_type']
    
    # Handle different event types
    if event_type == 'optimization.completed':
        handle_optimization_complete(data)
    elif event_type == 'vision.completed':
        handle_vision_complete(data)
    
    return {'status': 'received'}, 200

def handle_optimization_complete(data):
    job_id = data['data']['job_id']
    result = data['data']['result']
    print(f"Optimization {job_id} completed: {result}")

def handle_vision_complete(data):
    job_id = data['data']['job_id']
    detections = data['data']['detections']
    print(f"Vision analysis {job_id} found {len(detections)} objects")

if __name__ == '__main__':
    app.run(port=5000)
```

**JavaScript (Express)**

```javascript
const express = require('express');
const crypto = require('crypto');
const app = express();

const WEBHOOK_SECRET = 'your-webhook-secret';

app.use(express.json());

function verifySignature(payload, signature) {
  const expectedSignature = crypto
    .createHmac('sha256', WEBHOOK_SECRET)
    .update(payload)
    .digest('hex');
  
  return crypto.timingSafeEqual(signature, expectedSignature);
}

app.post('/webhook', (req, res) => {
  const payload = JSON.stringify(req.body);
  const signature = req.headers['x-webhook-signature'];
  
  // Verify signature
  if (!verifySignature(payload, signature)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }
  
  const { event_type, data } = req.body;
  
  // Handle events
  if (event_type === 'optimization.completed') {
    console.log(`Optimization ${data.job_id} completed`);
  } else if (event_type === 'vision.completed') {
    console.log(`Vision analysis ${data.job_id} completed`);
  }
  
  res.json({ status: 'received' });
});

app.listen(5000);
```

### Register Webhook

Register your webhook endpoint with the API:

```bash
curl -X POST https://api.aiplatform.io/v1/webhooks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook",
    "events": [
      "optimization.completed",
      "vision.completed",
      "training.completed"
    ],
    "secret": "your-webhook-secret"
  }'
```

Response:

```json
{
  "webhook_id": "wh_550e8400e29b41d4",
  "url": "https://example.com/webhook",
  "events": ["optimization.completed", "vision.completed"],
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### List Webhooks

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.aiplatform.io/v1/webhooks
```

### Update Webhook

```bash
curl -X PUT https://api.aiplatform.io/v1/webhooks/wh_550e8400e29b41d4 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/new-webhook",
    "events": ["optimization.completed", "vision.completed", "training.started"]
  }'
```

### Delete Webhook

```bash
curl -X DELETE https://api.aiplatform.io/v1/webhooks/wh_550e8400e29b41d4 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Webhook Payload Format

All webhook payloads follow this format:

```json
{
  "event_id": "evt_550e8400e29b41d4",
  "event_type": "optimization.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "result": {}
  },
  "retry_count": 0
}
```

## Event Examples

### Optimization Completed

```json
{
  "event_id": "evt_...",
  "event_type": "optimization.completed",
  "timestamp": "2024-01-15T10:35:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "problem_type": "MAXCUT",
    "result": {
      "solution": [0, 1, 0, 1],
      "objective_value": 8.0,
      "approximation_ratio": 0.95,
      "execution_time": 300
    }
  }
}
```

### Vision Analysis Completed

```json
{
  "event_id": "evt_...",
  "event_type": "vision.completed",
  "timestamp": "2024-01-15T10:35:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "detections": [
      {
        "class_name": "person",
        "confidence": 0.95,
        "bounding_box": {
          "x": 100,
          "y": 50,
          "width": 200,
          "height": 300
        }
      }
    ]
  }
}
```

### Batch Progress

```json
{
  "event_id": "evt_...",
  "event_type": "batch.progress",
  "timestamp": "2024-01-15T10:35:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "total_images": 100,
    "processed_images": 75,
    "failed_images": 0,
    "progress_percent": 75
  }
}
```

### Training Progress

```json
{
  "event_id": "evt_...",
  "event_type": "training.progress",
  "timestamp": "2024-01-15T10:35:00Z",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "round": 5,
    "accuracy": 0.92,
    "loss": 0.23,
    "total_rounds": 10
  }
}
```

## Security

### Signature Verification

All webhook payloads are signed with HMAC-SHA256:

1. Get the raw request body (before parsing JSON)
2. Create HMAC signature using your webhook secret
3. Compare with `X-Webhook-Signature` header

```python
import hmac
import hashlib

def verify_webhook(payload_bytes, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected)
```

### Headers

Every webhook request includes:

- `X-Webhook-Signature` - HMAC-SHA256 signature
- `X-Webhook-ID` - Unique webhook ID
- `X-Event-ID` - Unique event ID
- `X-Timestamp` - Event timestamp
- `Content-Type: application/json`

## Retry Policy

If your endpoint returns a non-2xx status or times out (30s), webhooks are retried:

- 1st retry: 5 minutes
- 2nd retry: 30 minutes
- 3rd retry: 2 hours
- 4th retry: 8 hours
- 5th retry: 24 hours

After 5 failed attempts, the webhook is disabled. You can manually retry or re-enable from the dashboard.

## Best Practices

### 1. Idempotency

Handle duplicate events by storing event IDs:

```python
import sqlite3

conn = sqlite3.connect('webhooks.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_events (
        event_id TEXT PRIMARY KEY,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

def handle_webhook(event):
    # Check if already processed
    cursor.execute('SELECT 1 FROM processed_events WHERE event_id = ?', (event['event_id'],))
    
    if cursor.fetchone():
        return 'Already processed'
    
    # Process event
    process_event(event)
    
    # Mark as processed
    cursor.execute('INSERT INTO processed_events (event_id) VALUES (?)', (event['event_id'],))
    conn.commit()
    
    return 'Processed'
```

### 2. Async Processing

Process webhooks asynchronously to avoid timeouts:

```python
from celery import shared_task

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    
    # Queue for async processing
    process_webhook_async.delay(data)
    
    return {'status': 'received'}, 202

@shared_task
def process_webhook_async(data):
    # Long-running processing
    handle_optimization_complete(data)
```

### 3. Logging

Log all webhook events for debugging:

```python
import logging

logger = logging.getLogger(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    
    logger.info(f"Received webhook: {data['event_type']}")
    
    try:
        process_event(data)
        logger.info(f"Successfully processed: {data['event_id']}")
    except Exception as e:
        logger.error(f"Failed to process: {data['event_id']}: {str(e)}")
        raise
    
    return {'status': 'received'}, 200
```

### 4. Error Handling

Always return 2xx on success:

```python
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        
        if not verify_signature(request.get_data(), request.headers.get('X-Webhook-Signature')):
            logger.warning("Invalid signature")
            return {'error': 'Invalid signature'}, 401
        
        process_event(data)
        return {'status': 'received'}, 200
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        # Return 2xx so webhook isn't retried
        return {'error': str(e)}, 200
```

## Testing Webhooks

### Manual Test

Send a test webhook:

```bash
curl -X POST https://api.aiplatform.io/v1/webhooks/wh_550e8400e29b41d4/test \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Local Testing

Use ngrok to expose local server:

```bash
# Install ngrok
pip install pyngrok

# Start your server
python app.py

# In another terminal
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Register webhook with public URL
```

## Monitoring

### View Webhook Deliveries

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api.aiplatform.io/v1/webhooks/wh_550e8400e29b41d4/deliveries?limit=50"
```

Response:

```json
{
  "deliveries": [
    {
      "delivery_id": "del_...",
      "event_id": "evt_...",
      "event_type": "optimization.completed",
      "status": "success",
      "status_code": 200,
      "timestamp": "2024-01-15T10:35:00Z",
      "response_time_ms": 145
    }
  ],
  "total": 145,
  "limit": 50
}
```

### Retry Webhook

Manually retry a failed delivery:

```bash
curl -X POST https://api.aiplatform.io/v1/webhooks/wh_550e8400e29b41d4/deliveries/del_550e8400e29b41d4/retry \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Conclusion

Webhooks provide a scalable way to receive real-time notifications from the AI Platform. For more information, see the [API Guide](API_GUIDE.md).
