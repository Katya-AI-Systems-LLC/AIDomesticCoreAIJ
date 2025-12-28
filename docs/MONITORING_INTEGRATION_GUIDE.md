# Monitoring Stack Integration Guide

## Environment Setup

### Development Environment Variables

Create `.env` file in project root:

```bash
# Elasticsearch
ELASTICSEARCH_HOSTS=elasticsearch:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=changeme
ELASTICSEARCH_SSL=false

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_secure_password

# Application Database
DB_HOST=db
DB_USER=postgres
DB_PASSWORD=your_db_password
DB_NAME=aiplatform

# Redis Cache
REDIS_HOST=redis
REDIS_PASSWORD=

# Monitoring
ENVIRONMENT=development
CLUSTER_NAME=local-dev
REGION=local

# Alerting (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
PAGERDUTY_SERVICE_KEY=your-pagerduty-key
OPSGENIE_API_KEY=your-opsgenie-key

# SaaS Monitoring
DD_API_KEY=your_datadog_api_key
DD_SITE=datadoghq.com
NEW_RELIC_LICENSE_KEY=your_new_relic_key

# TLS/Certificates (Production)
ELASTICSEARCH_SSL=true
TLS_CERT_PATH=/etc/ssl/certs/tls.crt
TLS_KEY_PATH=/etc/ssl/private/tls.key
```

## Installation Steps

### 1. Start Monitoring Stack

```bash
# Navigate to project root
cd /path/to/AIDomesticCoreAIJ

# Start all services
docker-compose -f docker-compose.monitoring.yml up -d

# Verify all services are running
docker-compose -f docker-compose.monitoring.yml ps

# Check logs for any errors
docker-compose -f docker-compose.monitoring.yml logs -f
```

### 2. Initialize Elasticsearch

```bash
# Wait for Elasticsearch to be healthy (30-60 seconds)
docker-compose -f docker-compose.monitoring.yml exec elasticsearch curl -s http://localhost:9200

# Create index templates
curl -X PUT "localhost:9200/_index_template/aiplatform-logs" -H 'Content-Type: application/json' -d '{
  "index_patterns": ["aiplatform-logs-*"],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2,
    "index.refresh_interval": "30s"
  },
  "mappings": {
    "properties": {
      "@timestamp": { "type": "date" },
      "level": { "type": "keyword" },
      "service": { "type": "keyword" },
      "trace_id": { "type": "keyword" },
      "message": { "type": "text" }
    }
  }
}'
```

### 3. Configure Grafana Dashboards

```bash
# Access Grafana: http://localhost:3000
# Login: admin / admin

# Steps:
# 1. Go to Configuration > Data Sources
# 2. Add Prometheus datasource (http://prometheus:9090)
# 3. Add Elasticsearch datasource (http://elasticsearch:9200)
# 4. Go to Dashboards > Import
# 5. Upload JSON from monitoring/grafana/dashboards/
```

### 4. Configure Prometheus Targets

```bash
# Verify Prometheus targets: http://localhost:9090/targets

# Update prometheus.yml if targets are down:
# - Replace 'localhost' with actual hostnames
# - For Kubernetes, use service DNS names
# - For Docker Swarm, use service discovery
```

### 5. Set Up Alerting

```bash
# Configure alert receivers in monitoring/alertmanager/alertmanager.yml:
# 1. Set SLACK_WEBHOOK_URL
# 2. Set PAGERDUTY_SERVICE_KEY (if using)
# 3. Configure email SMTP settings

# Reload Alertmanager
curl -X POST http://localhost:9093/-/reload
```

## Integration with Application

### 1. Application Metrics Endpoint

```python
# Add to your Flask/FastAPI application
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
requests_total = Counter(
    'aiplatform_requests_total',
    'Total requests',
    ['endpoint', 'method', 'status']
)

request_duration = Histogram(
    'aiplatform_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

# Add middleware to track metrics
from functools import wraps
import time

def track_metrics(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            status = 200
            return result
        except Exception as e:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            endpoint = request.endpoint or 'unknown'
            requests_total.labels(
                endpoint=endpoint,
                method=request.method,
                status=status
            ).inc()
            request_duration.labels(endpoint=endpoint).observe(duration)
    return decorated_function

# Expose metrics endpoint
from flask import Response

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')
```

### 2. Structured Logging

```python
import json
import logging
from pythonjsonlogger import jsonlogger

# Configure JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)

# Use structured logging
logger.info(
    "Request processed",
    extra={
        "user_id": user_id,
        "endpoint": endpoint,
        "duration_ms": duration,
        "status_code": status_code,
        "trace_id": trace_id
    }
)
```

### 3. Distributed Tracing (Optional)

```python
# Add Jaeger/Datadog tracing
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracer
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Use tracing in your code
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process_request"):
    # Your processing code
    pass
```

## Production Deployment

### Kubernetes Deployment

```yaml
# monitoring-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    name: monitoring

---
# prometheus-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    # Copy content from monitoring/prometheus/prometheus.yml

---
# prometheus-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: storage
          mountPath: /prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: storage
        persistentVolumeClaim:
          claimName: prometheus-storage

---
# prometheus-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  ports:
  - port: 9090
    targetPort: 9090
  selector:
    app: prometheus
  type: LoadBalancer
```

### Helm Chart Integration

```bash
# Add monitoring Helm charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add elastic https://helm.elastic.co
helm repo update

# Install Prometheus Operator
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring

# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace monitoring
```

## Monitoring the Monitoring Stack

### Health Checks

```bash
# Check Prometheus health
curl http://localhost:9090/-/healthy

# Check Alertmanager health
curl http://localhost:9093/-/healthy

# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Check Grafana health
curl http://localhost:3000/api/health
```

### Key Metrics to Monitor

```promql
# Prometheus disk usage
disk_usage_percent = (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100

# Elasticsearch cluster health
elasticsearch_cluster_health_status

# Logstash pipeline delays
logstash_pipeline_queue_size

# Grafana request latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

## Troubleshooting

### Common Issues

**Prometheus not scraping targets**:
```bash
# Check targets at: http://localhost:9090/targets
# Common issues:
# 1. DNS resolution - use container names for Docker
# 2. Port binding - verify services are listening
# 3. Network - ensure services are on same network
# 4. Firewall - check security groups/NSGs

# Debug:
docker-compose -f docker-compose.monitoring.yml exec prometheus \
  curl -v http://api:8000/metrics
```

**Elasticsearch out of memory**:
```bash
# Increase heap size in docker-compose
# Change ES_JAVA_OPTS=-Xms512m -Xmx512m to larger value
# Ensure Docker has enough memory

docker-compose -f docker-compose.monitoring.yml up -d elasticsearch
```

**Logstash not processing logs**:
```bash
# Check Logstash logs
docker-compose -f docker-compose.monitoring.yml logs -f logstash

# Verify input files exist
docker-compose -f docker-compose.monitoring.yml exec logstash \
  ls -la /var/log/
```

## Performance Tuning

### For High Volume

**Prometheus**:
```yaml
global:
  scrape_interval: 30s  # Increase from 15s
  evaluation_interval: 30s

# Or use sampling
metric_relabel_configs:
  - source_labels: [__name__]
    regex: '.*bucket.*'
    action: drop  # Drop high-cardinality metrics
```

**Elasticsearch**:
```yaml
index.number_of_shards: 5  # More shards for more parallelism
index.number_of_replicas: 1  # Fewer replicas if losing data is acceptable
```

**Logstash**:
```yaml
pipeline.batch.size: 2000  # Increase batch size
pipeline.workers: 8  # More workers
queue.max_bytes: 2gb  # Larger queue
```

## Backup and Restore

### Prometheus Backup

```bash
# Backup Prometheus data
docker-compose -f docker-compose.monitoring.yml exec prometheus \
  tar -czf /prometheus/backup.tar.gz /prometheus/

# Copy backup out
docker cp prometheus:/prometheus/backup.tar.gz ./

# Restore
tar -xzf backup.tar.gz -C /path/to/prometheus/
```

### Elasticsearch Snapshot

```bash
# Create snapshot repository
curl -X PUT "localhost:9200/_snapshot/backup" -H 'Content-Type: application/json' -d'{
  "type": "fs",
  "settings": {
    "location": "/usr/share/elasticsearch/backup"
  }
}'

# Create snapshot
curl -X PUT "localhost:9200/_snapshot/backup/snapshot_1?wait_for_completion=true"

# List snapshots
curl "localhost:9200/_snapshot/backup/_all"

# Restore snapshot
curl -X POST "localhost:9200/_snapshot/backup/snapshot_1/_restore"
```

## Cleanup and Maintenance

### Clean Up Old Data

```bash
# Delete old Prometheus data (keep last 30 days)
docker-compose -f docker-compose.monitoring.yml exec prometheus \
  find /prometheus -type f -mtime +30 -delete

# Delete old Elasticsearch indices
curl -X DELETE "localhost:9200/aiplatform-logs-$(date -d '30 days ago' +%Y.%m.%d)"

# Compact Elasticsearch
curl -X POST "localhost:9200/aiplatform-logs-*/_forcemerge?max_num_segments=1"
```

### Restart Services

```bash
# Restart all services
docker-compose -f docker-compose.monitoring.yml restart

# Restart specific service
docker-compose -f docker-compose.monitoring.yml restart prometheus

# Restart with cleanup
docker-compose -f docker-compose.monitoring.yml down
docker-compose -f docker-compose.monitoring.yml up -d
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Setup Guide](https://grafana.com/grafana/download)
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Monitoring](https://kubernetes.io/docs/tasks/debug-application-cluster/resource-metrics-pipeline/)
