# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- Docker image pushed to registry

## Quick Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=health-mlops
kubectl get services -l app=health-mlops

# View logs
kubectl logs -f deployment/health-mlops-dashboard
```

## Configuration

### 1. Update Docker Image

Edit `k8s/deployment.yaml` and replace:
```yaml
image: yourusername/health-mlops:latest
```

### 2. Configure Storage

The deployment uses PersistentVolumeClaims. Adjust storage sizes in `deployment.yaml` if needed:
- Models: 5Gi
- Data: 10Gi
- Reports: 2Gi

### 3. Setup Ingress (Optional)

Edit `k8s/ingress.yaml` to use your domain:
```yaml
host: health-mlops.yourdomain.com
```

## Scaling

Scale the deployment:
```bash
kubectl scale deployment health-mlops-dashboard --replicas=3
```

## Monitoring

Check pod status:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

## Cleanup

Remove all resources:
```bash
kubectl delete -f k8s/
```
