#  Health MLOps Project with Federated Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange.svg)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end MLOps system for health risk prediction using distributed data sources and privacy-preserving Federated Learning**

---

##  Project Overview

This project implements a complete **MLOps pipeline** that predicts health risks by combining data from wearable devices, air quality sensors, and weather stations, using **Federated Learning** to preserve privacy across multiple hospital nodes.

### Key Innovation
**Privacy-Preserving Machine Learning:** Train AI models across 5 hospital nodes without centralizing patient data, ensuring HIPAA/GDPR compliance.

---

##  Features

###  Data Pipeline
- ✅ Multi-source data ingestion (health wearables, air quality, weather)
- ✅ Automated data validation and quality checks
- ✅ Advanced feature engineering (65+ features)
- ✅ Data drift detection with statistical monitoring

###  Machine Learning
-  **4 Models Trained:**
  - Random Forest (R² = 0.759)
  - XGBoost (R² = 0.740)
  - PyTorch Neural Network (R² = 0.780) ⭐ **Best**
  - Federated Learning (R² = 0.750) 🔒 **Privacy-preserving**
-  Hyperparameter optimization
-  Model versioning and registry

###  MLOps Infrastructure
-  Complete CI/CD pipeline (GitHub Actions)
-  Automated testing (pytest with 14+ tests)
-  Docker containerization
-  Kubernetes deployment manifests
-  Performance monitoring & drift detection
-  REST API (FastAPI)

###  Deployment & Monitoring
-  Interactive dashboard (Gradio)
-  Real-time predictions (<100ms latency)
-  Model comparison interface
-  Monitoring reports and alerts

---

##  Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/health-mlops-project.git
cd health-mlops-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

```bash
# 1. Generate/download data
python src/download_data.py

# 2. Engineer features
python src/feature_engineering.py

# 3. Train models
python src/train_baseline.py
python src/train.py  # PyTorch models

# 4. Launch dashboard
python dashboards/gradio_dashboard.py
# Open browser: http://localhost:7860

# 5. Start API server (alternative)
python src/api.py
# API docs: http://localhost:8000/docs
```

### Using Docker

```bash
# Build and run
docker-compose up -d

# Access:
# Dashboard: http://localhost:7860
```

---

##  Results

### Model Performance

| Model | R² Score | RMSE | MAE | Training Time |
|-------|----------|------|-----|---------------|
| Random Forest | 0.759 | 1.012 | 0.747 | 0.08s |
| XGBoost | 0.740 | 1.051 | 0.793 | 0.17s |
| **PyTorch NN** | **0.780** | **0.959** | **0.747** | 1.40s |
| Federated | 0.750 | 1.13 | 0.82 | 50 rounds |

### Federated Learning Impact
-  **Privacy:** No raw data sharing between hospitals
-  **Performance Drop:** Only 3% vs centralized training
-  **Nodes:** 5 hospital locations
-  **Compliance:** HIPAA/GDPR ready

### Top Predictive Features
1. **health_deterioration** (68%)
2. **body_temp** (6%)
3. **spo2** (4%)
4. **pm25_change** (1%)
5. **heart_rate** (1%)

---

##  Architecture

```
┌─────────────────────────────────────────────┐
│           Data Sources                       │
│  Hospital 1-5  Air Quality  Weather         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│     Data Ingestion & Processing             │
│  Collection → Validation → Feature Eng      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Federated Learning Layer             │
│  Local Training → Aggregation → Global      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           MLOps Pipeline                    │
│  CI/CD → Testing → Building → Monitoring    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      Production Deployment                  │
│  Docker → Kubernetes → API → Dashboard      │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
health-mlops-project/
├── 📁 data/
│   ├── raw/              # Raw data from sources
│   ├── processed/        # Engineered features
│   └── federated/        # Node-specific data
├── 📁 src/
│   ├── download_data.py          # Data generation
│   ├── feature_engineering.py    # Feature creation
│   ├── train_baseline.py         # RF, XGBoost
│   ├── train.py                  # PyTorch models
│   ├── train_federated.py        # Federated learning
│   ├── data_drift_detection.py   # Drift monitoring
│   ├── api.py                    # REST API
│   └── [monitoring scripts]
├── 📁 models/
│   ├── baseline/         # Trained models
│   ├── pytorch/          # Neural networks
│   ├── federated/        # FL models
│   └── evaluation/       # Performance reports
├── 📁 tests/             # Unit tests (pytest)
├── 📁 dashboards/        # Gradio web UI
├── 📁 notebooks/         # EDA & experiments
├── 📁 docs/              # Documentation
├── 📁 k8s/               # Kubernetes configs
├── 📁 .github/           # CI/CD workflows
├── Dockerfile            # Container image
├── docker-compose.yml    # Multi-service setup
└── requirements.txt      # Dependencies
```

---

##  Technologies Used

### Machine Learning
- **Frameworks:** PyTorch, Scikit-Learn, XGBoost
- **Federated Learning:** Flower (flwr)
- **Data Processing:** Pandas, NumPy

### MLOps & Infrastructure
- **CI/CD:** GitHub Actions
- **Containerization:** Docker, Docker Compose
- **Orchestration:** Kubernetes
- **API:** FastAPI, Uvicorn
- **Dashboard:** Gradio, Plotly
- **Testing:** pytest, pytest-cov

### Monitoring & Observability
- **Drift Detection:** SciPy statistical tests
- **Performance Tracking:** Custom metrics
- **Reporting:** HTML reports, CSV logs

---

##  Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data.py -v
```

**Current Status:** ✅ 12/14 tests passing (86%)

---

##  Docker Deployment

### Build Image
```bash
docker build -t health-mlops:latest .
```

### Run Container
```bash
docker run -p 7860:7860 health-mlops:latest
```

### Docker Compose (Recommended)
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

##  Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=health-mlops
kubectl get services

# View logs
kubectl logs -f deployment/health-mlops-dashboard

# Scale deployment
kubectl scale deployment health-mlops-dashboard --replicas=3

# Cleanup
kubectl delete -f k8s/
```

---

##  Monitoring & Drift Detection

### Generate Monitoring Report
```bash
python src/generate_monitoring_report.py
```

### Check Data Drift
```bash
python src/data_drift_detection.py
```

### Monitor Model Performance
```bash
python src/monitor_drift.py
```

Reports are saved to `reports/` directory.

---

##  CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **Code Quality**
   - Linting (flake8)
   - Formatting (black)
   - Unit tests (pytest)

2. **Data Validation**
   - Schema checks
   - Drift detection
   - Quality reports

3. **Model Training**
   - Train all models
   - Performance validation
   - Threshold checks

4. **Docker Build**
   - Build images
   - Run tests
   - Push to registry

5. **Deployment**
   - Staging (develop branch)
   - Production (main branch)
   - Health checks

6. **Monitoring**
   - Track metrics
   - Detect drift
   - Send alerts

---

##  API Usage

### Start API Server
```bash
python src/api.py
# API docs: http://localhost:8000/docs
```

### Make Predictions
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "avg_heart_rate": 85.0,
        "avg_spo2": 96.0,
        "avg_body_temp": 37.2,
        "avg_steps": 6000,
        "avg_pm25": 55.0,
        "avg_pm10": 75.0,
        "avg_no2": 35.0,
        "avg_aqi": 95.0,
        "avg_temperature": 28.0,
        "avg_humidity": 70.0,
        "avg_pressure": 1010.0
    }
)

print(response.json())
```

---

##  Documentation

- **[Project Compliance Report](PROJECT_COMPLIANCE_REPORT.md)** - Full requirement verification
- **[Docker Deployment Guide](DOCKER_DEPLOYMENT.md)** - Container setup
- **[Kubernetes README](k8s/README.md)** - K8s deployment
- **[Model Evaluation Report](docs/Model%20Evaluation%20Report.docx)** - Performance analysis
- **[Project Overview](docs/README.md%20-%20Project%20Overview.md)** - Detailed documentation

---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

##  Team

**Project by:** Saamer Abbas (i220468) and Team

**Built with:** Python, PyTorch, Flower, FastAPI, Docker, Kubernetes

---

##  Acknowledgments

- Flower framework for Federated Learning
- FastAPI for API development
- Gradio for interactive dashboards
- Open-source ML community

---

##  Contact

For questions or issues, please open an issue on GitHub.

---

** If you find this project helpful, please star the repository!**

---

<div align="center">


