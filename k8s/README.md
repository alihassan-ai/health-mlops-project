# Kubernetes Deployment Guide - Health MLOps

Complete production-ready Kubernetes manifests for deploying the Health MLOps Federated Learning system.

## 🎯 Why Kubernetes for Health MLOps?

1. **High Availability** - 3 replicas, self-healing, zero-downtime updates
2. **Auto-Scaling** - Scale 3→20 pods based on load (HPA)
3. **Security** - Network policies isolate hospitals, RBAC for access control
4. **Multi-Region** - Deploy in EU/US/Asia for data residency (GDPR)
5. **Federated Learning** - Coordinate training across 5+ hospitals securely

## 📦 Quick Start

### Deploy Everything:
```bash
cd k8s/
kubectl apply -f 00-namespaces.yaml
kubectl apply -f 01-configmaps.yaml
kubectl apply -f 02-persistent-volumes.yaml
kubectl apply -f 03-dashboard-deployment.yaml
kubectl apply -f 04-monitoring-stack.yaml
kubectl apply -f 05-federated-node.yaml
kubectl apply -f 06-autoscaling-ingress.yaml
```

### Access Services (Local):
```bash
kubectl port-forward svc/dashboard-service 7860:7860 -n health-mlops-central
kubectl port-forward svc/grafana-service 3000:3000 -n health-mlops-central
```

## 📚 Files Overview

- **00-namespaces.yaml** - 6 namespaces (1 central + 5 hospitals)
- **01-configmaps.yaml** - Configuration for services
- **02-persistent-volumes.yaml** - Storage (10Gi models, 20Gi metrics)
- **03-dashboard-deployment.yaml** - Dashboard with 3 replicas, LoadBalancer
- **04-monitoring-stack.yaml** - Prometheus + Grafana + RBAC
- **05-federated-node.yaml** - Hospital nodes template
- **06-autoscaling-ingress.yaml** - HPA (3-20 pods) + Ingress + NetworkPolicies

## 🔍 Key Features

- **Auto-Scaling:** HPA scales dashboard 3→20 based on CPU (70%) and memory (80%)
- **Network Isolation:** Hospitals can't access each other's data
- **Resource Limits:** All pods have CPU/memory limits
- **Health Checks:** Liveness & readiness probes on all services
- **Rolling Updates:** Zero-downtime deployments
- **Persistent Storage:** Models and metrics survive pod restarts

## 📊 Monitoring

```bash
# View metrics
kubectl top pods -n health-mlops-central

# Watch auto-scaling
kubectl get hpa dashboard-hpa -n health-mlops-central --watch

# View logs
kubectl logs -f deployment/dashboard -n health-mlops-central
```

## ✅ Production Ready

This configuration includes:
- High availability (multiple replicas)
- Auto-scaling (HPA)
- Security (NetworkPolicies, RBAC)
- Monitoring (Prometheus + Grafana)
- Persistent storage
- Load balancing
- Health checks
