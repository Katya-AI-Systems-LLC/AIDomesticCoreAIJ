# AI Platform Monitoring & Observability Guide

## Overview

This guide covers the complete monitoring and observability setup for the AI Platform, including metrics collection, log aggregation, visualization, and alerting.

## Architecture Components

### Metrics & Alerting Stack
- **Prometheus**: Time-series database for metrics collection
- **Alertmanager**: Alert routing and management
- **Grafana**: Visualization and dashboarding

### Log Management Stack
- **Logstash**: Log processing and transformation
- **Elasticsearch**: Distributed search and analytics
- **Kibana**: Log visualization and exploration

### SaaS Monitoring Options
- **Datadog**: Full-stack observability platform
- **New Relic**: APM and monitoring platform

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Kubernetes cluster (for production)
- 4GB RAM minimum for monitoring stack

### Local Development Setup

1. **Start the monitoring stack**:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

2. **Access the services**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Kibana: http://localhost:5601 (elastic/changeme)
- Alertmanager: http://localhost:9093

3. **Import dashboards**:
```bash
# Dashboards are automatically provisioned from monitoring/grafana/dashboards/
# Access Grafana and navigate to Dashboards section
```

## Prometheus Configuration

### Scrape Jobs

The Prometheus configuration includes 15 scrape jobs:

#### 1. Prometheus Self-Monitoring
```yaml
job_name: 'prometheus'
static_configs:
  - targets: ['localhost:9090']
```

#### 2. Kubernetes API Servers
```yaml
job_name: 'kubernetes-apiservers'
kubernetes_sd_configs:
  - role: endpoints
```

#### 3. Kubernetes Nodes
```yaml
job_name: 'kubernetes-nodes'
kubernetes_sd_configs:
  - role: node
```

#### 4. Kubernetes Pods
```yaml
job_name: 'kubernetes-pods'
kubernetes_sd_configs:
  - role: pod
relabel_configs:
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
    action: keep
    regex: 'true'
```

#### 5. Application API Metrics
```yaml
job_name: 'aiplatform-api'
metrics_path: '/metrics'
static_configs:
  - targets: ['api:8000']
```

#### 6-10. Infrastructure Exporters
- PostgreSQL Exporter (port 9187)
- Redis Exporter (port 9121)
- Node Exporter (port 9100)
- cAdvisor (port 8080)
- Docker (port 8080)

### Metric Categories

#### API Metrics
```
http_requests_total - Total HTTP requests
http_request_duration_seconds - Request latency
http_requests_total{status=~"5.."} - Error requests
```

#### Database Metrics
```
pg_stat_activity_count - Active connections
pg_database_size_bytes - Database size
pg_stat_statements_mean_exec_time - Query execution time
pg_replication_lag - Replication delay
```

#### Cache Metrics
```
redis_memory_used_bytes - Memory usage
redis_keyspace_hits_total - Cache hits
redis_keyspace_misses_total - Cache misses
redis_evicted_keys_total - Evicted keys
```

#### Kubernetes Metrics
```
kube_node_status_condition - Node status
kube_pod_status_phase - Pod phase
container_cpu_usage_seconds_total - CPU usage
container_memory_usage_bytes - Memory usage
```

#### Infrastructure Metrics
```
node_cpu_seconds_total - CPU time
node_memory_MemAvailable_bytes - Available memory
node_filesystem_avail_bytes - Filesystem space
node_network_receive_bytes_total - Network input
```

## Alert Rules

### Critical Alerts

#### API High Error Rate
- Condition: Error rate > 5% for 5 minutes
- Impact: Indicates API instability
- Action: Investigate error logs, check dependencies

#### Database Down
- Condition: PostgreSQL unavailable for 1 minute
- Impact: Complete data access loss
- Action: Check database service, verify network connectivity

#### Kubernetes Pod Crash Looping
- Condition: Pod restart rate > 0.1/min for 5 minutes
- Impact: Application instability
- Action: Check pod logs, review resource limits

#### Node Not Ready
- Condition: Kubernetes node not in Ready state for 5 minutes
- Impact: Loss of compute capacity
- Action: SSH to node, check kubelet logs

### Warning Alerts

#### API High Latency
- Condition: P95 latency > 1 second for 5 minutes
- Impact: Degraded user experience
- Action: Check query performance, database load

#### High Memory Usage
- Condition: Memory > 85% for 5 minutes
- Impact: Risk of OOMKill
- Action: Review memory allocation, check for leaks

#### High Disk Usage
- Condition: Disk > 90% for 5 minutes
- Impact: Risk of write failures
- Action: Clean up old logs, increase storage

## Grafana Dashboards

### Available Dashboards

#### 1. API Performance Dashboard
Metrics:
- Request rate (requests/sec)
- Error rate by status code
- Latency percentiles (P50, P95, P99)
- Top endpoints by traffic
- Top endpoints by error rate
- Memory and CPU usage

#### 2. Kubernetes Cluster Dashboard
Metrics:
- Node status (ready/not ready)
- Pod status (running/failed/pending)
- Pod restart count
- CPU and memory usage by namespace
- PVC usage
- Network I/O

#### 3. Infrastructure Dashboard
Metrics:
- Host CPU usage by instance
- Host memory usage by instance
- Disk usage by filesystem
- Load average (1/5/15 min)
- Network I/O by interface
- Database connections
- Cache memory usage

#### 4. Database Performance Dashboard
Metrics:
- Active connections
- Database size
- Cache hit ratio
- Query duration
- Connections by state
- Replication lag
- TPS (transactions per second)

### Dashboard Best Practices

1. **Use appropriate time ranges**:
   - Real-time: 1-5 minute range
   - Troubleshooting: Last 6-24 hours
   - Trending: Last 7-30 days

2. **Set meaningful alert thresholds** in each panel

3. **Use color thresholds** to visualize health status

4. **Create custom panels** for business metrics

## ELK Stack Configuration

### Elasticsearch

#### Index Management
```bash
# Create index with settings
PUT aiplatform-logs
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2,
    "index.refresh_interval": "30s"
  }
}
```

#### Index Lifecycle Management (ILM)
```bash
# Create ILM policy
PUT _ilm/policy/aiplatform-policy
{
  "policy": "aiplatform-policy",
  "phases": {
    "hot": {
      "min_age": "0d",
      "actions": {}
    },
    "warm": {
      "min_age": "7d",
      "actions": {
        "set_priority": {"priority": 50}
      }
    },
    "cold": {
      "min_age": "30d",
      "actions": {
        "set_priority": {"priority": 0}
      }
    }
  }
}
```

#### Security
```bash
# Create role
POST _security/role/aiplatform-logs
{
  "indices": [
    {
      "names": ["aiplatform-logs-*"],
      "privileges": ["read", "view_index_metadata"]
    }
  ]
}
```

### Logstash Pipeline

The Logstash pipeline processes logs from multiple sources:

1. **Input Plugins**:
   - File input (application logs)
   - TCP input (structured logs)
   - UDP input (syslog)

2. **Filter Plugins**:
   - JSON parsing
   - Grok pattern matching
   - Date parsing
   - GeoIP enrichment
   - Fingerprinting for deduplication

3. **Output Plugins**:
   - Elasticsearch (main index)
   - Elasticsearch (error index)
   - Elasticsearch (metrics index)

### Kibana Analysis

#### Index Patterns
Create index patterns to analyze logs:
```
aiplatform-logs-*
aiplatform-errors-*
aiplatform-metrics-*
```

#### Searches and Visualizations
1. **Error Dashboard**: Filter by severity:high
2. **API Access Dashboard**: Analyze HTTP requests and responses
3. **Performance Dashboard**: Track query durations and latencies
4. **Alerts Dashboard**: Monitor alert patterns

## Datadog Integration

### Installation

1. **Install Datadog Agent**:
```bash
DD_API_KEY=your_api_key DD_SITE=datadoghq.com bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_agent.sh)"
```

2. **Configure integrations** in datadog.yaml:
```yaml
integrations:
  postgres:
    enabled: true
  redis:
    enabled: true
  kubernetes:
    enabled: true
```

### Features

- **APM/Tracing**: Distributed tracing across services
- **Logs**: Centralized log collection and analysis
- **Metrics**: Custom and standard metrics collection
- **Profiles**: Continuous profiling for performance analysis
- **RUM**: Real User Monitoring for frontend metrics

### Synthetic Tests

Create synthetic tests to monitor endpoints:
```bash
datadog synthetics create --name "API Health" \
  --request-type GET \
  --url "https://api.example.com/health" \
  --frequency 300
```

## New Relic Integration

### Installation

1. **Install New Relic Agent**:
```bash
pip install newrelic
```

2. **Generate config**:
```bash
newrelic-admin generate-config <license-key> newrelic.ini
```

3. **Start application with agent**:
```bash
NEW_RELIC_CONFIG_FILE=newrelic.ini \
  newrelic-admin run-program python app.py
```

### Features

- **APM**: Application Performance Monitoring
- **Infrastructure**: Host and container metrics
- **Logs**: Integrated log monitoring
- **Alerts**: Intelligent alerting based on baselines
- **Dashboards**: Pre-built dashboards

## Custom Metrics

### Application Level

Expose custom metrics via Prometheus endpoint:

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
requests_total = Counter(
    'aiplatform_requests_total',
    'Total requests',
    ['endpoint', 'method', 'status']
)

# Histograms
request_duration = Histogram(
    'aiplatform_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

# Gauges
active_connections = Gauge(
    'aiplatform_active_connections',
    'Active connections'
)
```

### Business Metrics

Track business-related metrics:
```
aiplatform_ml_predictions_total - Total ML predictions
aiplatform_ml_prediction_accuracy - Prediction accuracy
aiplatform_federated_training_rounds - Training rounds completed
aiplatform_quantum_optimizations - Quantum optimizations performed
```

## Troubleshooting

### Prometheus Issues

**Prometheus not scraping targets**:
1. Check targets at http://localhost:9090/targets
2. Verify network connectivity to target
3. Check Prometheus logs for errors

**High memory usage**:
1. Reduce `scrape_interval` if possible
2. Implement data retention policies
3. Use remote storage for long-term retention

### Elasticsearch Issues

**Out of memory**:
1. Increase heap size: `ES_JAVA_OPTS=-Xmx2g`
2. Implement ILM policies for index cleanup
3. Check for large shards

**Slow queries**:
1. Use query profiling: `?profile=true`
2. Check index mappings
3. Add appropriate indexes

### Logstash Issues

**High CPU usage**:
1. Reduce batch size if possible
2. Check filter complexity
3. Monitor queue depth

**Lost messages**:
1. Enable persistent queue: `queue.type: persisted`
2. Set output batch size appropriately
3. Check error logs

## Maintenance

### Regular Tasks

**Daily**:
- Monitor dashboard for anomalies
- Check alert volume
- Review error logs

**Weekly**:
- Review Grafana dashboard performance
- Check Elasticsearch cluster health
- Verify backups

**Monthly**:
- Update Prometheus scrape configs
- Review and optimize alert rules
- Analyze long-term trends
- Update agent versions

### Backup and Recovery

**Prometheus data**:
```bash
# Backup
tar -czf prometheus-backup.tar.gz /prometheus

# Restore
tar -xzf prometheus-backup.tar.gz -C /
```

**Elasticsearch snapshots**:
```bash
# Create snapshot
PUT _snapshot/my_backup/snapshot_1

# Restore
POST _snapshot/my_backup/snapshot_1/_restore
```

## Security Considerations

1. **Network Security**:
   - Use TLS for all connections
   - Restrict access to monitoring endpoints
   - Use VPN for remote access

2. **Authentication**:
   - Enable Elasticsearch security
   - Set strong Grafana passwords
   - Use API keys for integrations

3. **Data Protection**:
   - Encrypt sensitive metrics
   - Implement log redaction for passwords
   - Use RBAC for access control

## Performance Tuning

### Prometheus
- Increase `scrape_interval` if scraping is slow
- Reduce cardinality of high-cardinality metrics
- Use recording rules for expensive queries

### Elasticsearch
- Tune heap size for workload
- Optimize index mappings
- Use appropriate shard count
- Enable compression for storage

### Grafana
- Cache dashboard JSON
- Reduce query frequency
- Use sampling for large datasets

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Logstash Documentation](https://www.elastic.co/guide/en/logstash/current/index.html)
- [Datadog Documentation](https://docs.datadoghq.com/)
- [New Relic Documentation](https://docs.newrelic.com/)
