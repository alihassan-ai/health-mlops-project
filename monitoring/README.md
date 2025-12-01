# Monitoring Stack - Health MLOps Project

## Overview

Complete monitoring solution using Prometheus and Grafana for tracking model performance, system health, and infrastructure metrics.

---

## 🎯 Components

### 1. Prometheus
- **Purpose:** Metrics collection and time-series database
- **Port:** 9090
- **URL:** http://localhost:9090

**Metrics Collected:**
- Model prediction counts
- API response times
- System resource usage
- Application health

### 2. Grafana
- **Purpose:** Visualization and dashboarding
- **Port:** 3000
- **URL:** http://localhost:3000
- **Default Login:** admin / admin

**Dashboards:**
- Health MLOps Monitoring (main dashboard)
- Model performance metrics
- System health overview

### 3. Node Exporter
- **Purpose:** System-level metrics
- **Port:** 9100
- **Metrics:** CPU, memory, disk, network

---

## 🚀 Quick Start

### Start All Services
```bash
docker-compose up -d
```

### Access Services
- **Gradio Dashboard:** http://localhost:7860
- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090

### Stop Services
```bash
docker-compose down
```

---

## 📊 Grafana Dashboard Setup

### First Time Setup:

1. **Access Grafana:** http://localhost:3000
2. **Login:** admin / admin (change password when prompted)
3. **Data Source:** Already configured (Prometheus)
4. **Import Dashboard:**
   - Go to Dashboards → Import
   - Upload `dashboards/health-mlops-dashboard.json`

### Dashboard Features:

**Panel 1: Model Predictions per Minute**
- Real-time prediction rate
- Shows model usage

**Panel 2: Model Accuracy**
- Current model accuracy
- Performance tracking

**Panel 3: API Response Time**
- 95th percentile latency
- Performance monitoring

**Panel 4: System Health**
- Service status
- Uptime monitoring

---

## 🔍 Prometheus Queries

Useful queries for monitoring:

```promql
# Prediction rate
rate(predictions_total[1m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# System uptime
up

# CPU usage
rate(node_cpu_seconds_total{mode="idle"}[5m])
```

---

## 📈 Monitoring Strategy

### What We Monitor:

1. **Model Performance**
   - Prediction accuracy
   - Inference time
   - Model drift indicators

2. **API Performance**
   - Request rate
   - Response time
   - Error rate

3. **System Health**
   - CPU usage
   - Memory usage
   - Disk space
   - Network I/O

4. **Data Quality**
   - Input data distribution
   - Missing values
   - Anomalies

---

## 🔔 Alerting (Optional)

### Configure Alerts in Prometheus:

Edit `prometheus/alert_rules.yml`:

```yaml
groups:
  - name: health_mlops_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: ModelDrift
        expr: model_accuracy < 0.70
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy below threshold"
```

---

## 🎓 For Presentation

### What to Show:

1. **Open Grafana Dashboard**
   ```
   http://localhost:3000
   ```

2. **Explain Monitoring:**
   - "Real-time monitoring of our ML models"
   - "Track prediction rate, accuracy, and latency"
   - "Automated alerting for issues"

3. **Show Metrics:**
   - Point to prediction rate graph
   - Show system health status
   - Explain how it detects drift

### Key Points:

- ✅ "Complete monitoring stack with Prometheus & Grafana"
- ✅ "Real-time metrics for model performance"
- ✅ "System health monitoring"
- ✅ "Production-ready observability"

---

## 📝 Metrics Endpoints

### Add to Your Application:

For Python/FastAPI:
```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
predictions_counter = Counter('predictions_total', 'Total predictions made')
latency_histogram = Histogram('prediction_duration_seconds', 'Prediction latency')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## 🔧 Configuration Files

**Location:**
```
monitoring/
├── prometheus/
│   └── prometheus.yml          # Prometheus config
├── grafana/
│   ├── provisioning/
│   │   └── datasources.yml     # Data source config
│   └── dashboards/
│       └── health-mlops-dashboard.json  # Dashboard config
└── README.md                   # This file
```

---

## 🎯 Monitoring Checklist

- [x] Prometheus configured
- [x] Grafana dashboards created
- [x] Node Exporter for system metrics
- [x] Docker Compose integration
- [x] Default dashboards provisioned
- [x] Data source auto-configured

---

## 📚 Additional Resources

**Prometheus:**
- Query Language: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Alerting: https://prometheus.io/docs/alerting/latest/overview/

**Grafana:**
- Dashboard Guide: https://grafana.com/docs/grafana/latest/dashboards/
- Panels: https://grafana.com/docs/grafana/latest/panels/

---

## ✅ Verification

Test the monitoring stack:

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:7860/metrics

# Check Grafana health
curl http://localhost:3000/api/health
```

---

**Status:** ✅ READY FOR PRODUCTION
**Last Updated:** December 1, 2025
