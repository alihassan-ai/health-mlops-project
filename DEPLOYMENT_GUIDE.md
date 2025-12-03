# Health MLOps Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Health MLOps Federated Learning system across different environments.

---

## 🚀 Quick Start

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alihassan-ai/health-mlops-project.git
   cd health-mlops-project
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare data:**
   ```bash
   python src/download_data.py
   python src/feature_engineering.py
   ```

5. **Train models:**
   ```bash
   # Train baseline models
   python src/train_baseline.py

   # Train PyTorch models
   python src/train_pytorch.py

   # Train federated learning models
   python src/train_federated.py
   ```

6. **Launch dashboard:**
   ```bash
   python dashboards/gradio_dashboard.py
   # Access at http://localhost:7860
   ```

---

## 🐳 Docker Deployment

### Single Container (Dashboard Only)

```bash
# Build the image
docker build -t health-mlops:latest .

# Run the container
docker run -p 7860:7860 -v $(pwd)/models:/app/models health-mlops:latest
```

### Full Stack with Monitoring

```bash
# Start all services (Dashboard + Prometheus + Grafana)
docker-compose up -d

# Access services
# Dashboard: http://localhost:7860
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- kubectl installed and configured
- Kubernetes cluster (Minikube, EKS, GKE, or AKS)
- Docker images built and pushed to registry

### Local Testing with Minikube

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Deploy to Kubernetes
cd k8s/
kubectl apply -f 00-namespaces.yaml
kubectl apply -f 01-configmaps.yaml
kubectl apply -f 02-persistent-volumes.yaml
kubectl apply -f 03-dashboard-deployment.yaml
kubectl apply -f 04-monitoring-stack.yaml
kubectl apply -f 05-federated-node.yaml
kubectl apply -f 06-autoscaling-ingress.yaml

# Check deployment status
kubectl get pods -n health-mlops-central
kubectl get services -n health-mlops-central

# Access dashboard locally
kubectl port-forward svc/dashboard-service 7860:7860 -n health-mlops-central
```

### Production Cloud Deployment

#### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name health-mlops \
  --region us-east-1 \
  --nodes 3 \
  --node-type m5.2xlarge

# Deploy application
kubectl apply -f k8s/

# Get LoadBalancer URLs
kubectl get services -n health-mlops-central
```

#### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create health-mlops \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --region=us-central1

# Deploy application
kubectl apply -f k8s/

# Get external IPs
kubectl get services -n health-mlops-central
```

#### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group health-mlops-rg \
  --name health-mlops \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3

# Get credentials
az aks get-credentials --resource-group health-mlops-rg --name health-mlops

# Deploy application
kubectl apply -f k8s/
```

---

## 🔐 Security Configuration

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# .env
DOCKER_USERNAME=your-dockerhub-username
DOCKER_PASSWORD=your-dockerhub-token
DATABASE_URL=your-database-connection-string
API_KEY=your-api-key
```

### GitHub Secrets

Configure the following secrets in your GitHub repository:

- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token
- `SLACK_WEBHOOK` - (Optional) Slack webhook for notifications

**Path:** Settings → Secrets and variables → Actions → New repository secret

---

## 📊 Monitoring Setup

### Prometheus Configuration

The Prometheus instance automatically discovers and scrapes metrics from:
- Dashboard application (port 8000)
- Federated learning nodes (port 8001)
- Node exporters (port 9100)

### Grafana Dashboards

1. Access Grafana at `http://localhost:3000`
2. Login with `admin / admin`
3. Dashboard is auto-imported: "Health MLOps Monitoring Dashboard"

**Key Metrics:**
- Predictions per minute
- Model accuracy
- Service health status
- HTTP request rates
- Federated learning node status

---

## 🔄 CI/CD Pipeline

The project includes automated CI/CD with GitHub Actions:

### Workflows

1. **demo-simple.yml** - Quick verification
2. **ci-demo.yml** - Simplified testing
3. **ci-cd.yml** - Full production pipeline (7 stages)

### Pipeline Stages

1. **Code Quality** - Linting, formatting, unit tests
2. **Data Validation** - Schema validation, drift detection
3. **Model Training** - Train and evaluate all models
4. **Docker Build** - Build and test containers
5. **Staging Deployment** - Deploy to staging environment
6. **Production Deployment** - Deploy to production
7. **Monitoring** - Drift detection and alerting

### Triggering Workflows

Workflows trigger automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Daily schedule (drift detection at 2 AM UTC)

---

## 🏥 Federated Learning Deployment

### Hospital Node Setup

Each hospital requires its own deployment:

```bash
# Deploy hospital node
kubectl apply -f k8s/05-federated-node.yaml

# Customize for each hospital
# Edit namespace: hospital-1, hospital-2, etc.
# Set HOSPITAL_ID environment variable
```

### Central Aggregation Server

The central server coordinates federated learning:

```python
# Located in src/train_federated.py
# FederatedServer class handles:
# - Global model initialization
# - Weight aggregation (FedAvg)
# - Model distribution to clients
```

---

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:7860
```

---

## 📈 Scaling Guidelines

### Horizontal Pod Autoscaler (HPA)

The dashboard automatically scales based on:
- **Min replicas:** 3
- **Max replicas:** 20
- **CPU threshold:** 70%
- **Memory threshold:** 80%

### Resource Allocation

**Development:**
- 3 nodes × m5.large
- ~$260/month on AWS

**Production:**
- 5 nodes × m5.2xlarge
- ~$870/month on AWS

---

## 🔧 Troubleshooting

### Common Issues

**1. Dashboard not starting:**
```bash
# Check logs
kubectl logs -f deployment/dashboard -n health-mlops-central

# Common fix: Ensure models are trained
python src/train_baseline.py
```

**2. Prometheus not scraping metrics:**
```bash
# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
# Navigate to http://localhost:9090/targets
```

**3. Federated learning connection issues:**
```bash
# Check network policies
kubectl get networkpolicies -n hospital-1

# Test connectivity
kubectl exec -it <pod-name> -n hospital-1 -- ping federated-aggregator
```

### Health Checks

```bash
# Check all pods
kubectl get pods --all-namespaces

# Check service health
kubectl get services -n health-mlops-central

# View recent events
kubectl get events -n health-mlops-central --sort-by='.lastTimestamp'
```

---

## 📚 Additional Resources

- [Kubernetes Documentation](k8s/README.md)
- [Architecture Overview](markdowns/ARCHITECTURE_CLARIFICATION.md)
- [Research Paper](docs/RESEARCH_PAPER.md)
- [CI/CD Fix Guide](CI_CD_FIX_SUMMARY.md)

---

## 🤝 Support

For issues or questions:
- Create an issue on GitHub
- Check existing documentation in `docs/` and `markdowns/`
- Review workflow runs in GitHub Actions tab

---

**Last Updated:** December 2024
**Version:** 1.0
**Status:** Production Ready
