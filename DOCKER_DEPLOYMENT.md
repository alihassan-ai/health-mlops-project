# Docker Deployment Guide

## Prerequisites

1. **Docker Desktop** installed and running
   - Download: https://www.docker.com/products/docker-desktop/
   - Windows: Requires WSL2
   - Verify: `docker --version`

2. **Docker Compose** (included with Docker Desktop)
   - Verify: `docker-compose --version`

---

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Access dashboard
# Open browser: http://localhost:7860
```

### Option 2: Using Docker directly

```bash
# Build image
docker build -t health-mlops:latest .

# Run container
docker run -d -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  --name health-mlops-dashboard \
  health-mlops:latest

# View logs
docker logs -f health-mlops-dashboard
```

### Option 3: Using deployment script (Linux/Mac)

```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

---

## Container Management

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dashboard
```

### Check Status
```bash
docker-compose ps
```

### Rebuild After Changes
```bash
docker-compose up -d --build
```

---

## Exposed Ports

- **7860**: Gradio Dashboard
- **9090**: Prometheus (if enabled)
- **3000**: Grafana (if enabled)

---

## Volume Mounts

Data persists through volume mounts:

- `./data` → `/app/data` (Datasets)
- `./models` → `/app/models` (Trained models)
- `./reports` → `/app/reports` (Evaluation reports)
- `./logs` → `/app/logs` (Application logs)

---

## Monitoring (Optional)

To enable Prometheus + Grafana monitoring:

1. Uncomment the monitoring services in `docker-compose.yml`
2. Run: `docker-compose up -d`
3. Access:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs dashboard

# Check Docker daemon
docker info
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8080:7860"  # Use port 8080 instead
```

### Memory issues
```bash
# Increase Docker memory in Docker Desktop settings
# Settings → Resources → Memory (recommend 4GB+)
```

### Rebuild from scratch
```bash
# Remove everything and rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

---

## Production Deployment

### Environment Variables

Create `.env` file:
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
LOG_LEVEL=INFO
```

### SSL/HTTPS

Use nginx as reverse proxy:
```yaml
# Add to docker-compose.yml
nginx:
  image: nginx:latest
  ports:
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
```

### Scaling

```bash
# Scale dashboard service
docker-compose up -d --scale dashboard=3
```

---

## CI/CD Integration

See `.github/workflows/docker-deploy.yml` for automated deployment pipeline.

---

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker rmi health-mlops:latest
```

---

## Support

For issues, check:
1. Docker logs: `docker-compose logs -f`
2. Container status: `docker-compose ps`
3. Docker Desktop dashboard 