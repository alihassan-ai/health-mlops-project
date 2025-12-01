# **🏥 Health MLOps Project with Federated Learning**

End-to-end MLOps system for health risk prediction using distributed data sources and privacy-preserving Federated Learning

CI/CD [Docker](https://hub.docker.com/r/yourusername/health-mlops) [Python](https://www.python.org/downloads/) [License](https://claude.ai/chat/LICENSE)

---

## **📋 Table of Contents**

* [Overview](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#overview)  
* [Features](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#features)  
* [Architecture](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#architecture)  
* [Quick Start](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#quick-start)  
* [Project Structure](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#project-structure)  
* [Installation](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#installation)  
* [Usage](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#usage)  
* [CI/CD Pipeline](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#cicd-pipeline)  
* [Deployment](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#deployment)  
* [Monitoring](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#monitoring)  
* [Results](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#results)  
* [Contributing](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#contributing)  
* [License](https://claude.ai/chat/750be66e-835d-4053-aa15-3a8c52c80278#license)

---

## **🎯 Overview**

This project implements a complete MLOps pipeline that:

* Collects data from **wearable devices**, **air quality sensors**, and **weather stations**  
* Trains ML models using **Federated Learning** (data stays distributed)  
* Automates the full ML lifecycle with **CI/CD**  
* Predicts health risks in **real-time**  
* Provides dashboards for **health authorities** and **citizens**

### **Key Innovation**

**Privacy-Preserving ML:** Train models across 5 hospital nodes without ever centralizing patient data, complying with HIPAA/GDPR.

---

## **✨ Features**

### **Data Pipeline**

* ✅ Multi-source data ingestion (health, air quality, weather)  
* ✅ Automated data validation and quality checks  
* ✅ Advanced feature engineering (65+ features)  
* ✅ Data drift detection

### **Machine Learning**

* ✅ Multiple models: Random Forest, XGBoost, PyTorch Neural Networks  
* ✅ Federated Learning with Flower framework  
* ✅ Hyperparameter optimization  
* ✅ Model versioning and registry

### **MLOps**

* ✅ Full CI/CD pipeline (GitHub Actions)  
* ✅ Automated testing (unit, integration, smoke)  
* ✅ Docker containerization  
* ✅ Kubernetes deployment (optional)  
* ✅ Model monitoring and retraining triggers

### **Deployment**

* ✅ REST API (FastAPI)  
* ✅ Real-time predictions  
* ✅ Health dashboards (Streamlit)  
* ✅ Scalable infrastructure

---

## **🏗️ Architecture**

┌─────────────────────────────────────────────────────────┐  
│                    Data Sources                          │  
│  Hospital 1  Hospital 2  ...  Air Quality   Weather     │  
└────────────────────┬────────────────────────────────────┘  
                     │  
                     ▼  
┌─────────────────────────────────────────────────────────┐  
│              Data Ingestion & Processing                 │  
│  • Data Collection  • Validation  • Feature Engineering │  
└────────────────────┬────────────────────────────────────┘  
                     │  
                     ▼  
┌─────────────────────────────────────────────────────────┐  
│               Federated Learning Layer                   │  
│  Node 1  →  Aggregate  ←  Node 2  ←  ...  ←  Node 5    │  
└────────────────────┬────────────────────────────────────┘  
                     │  
                     ▼  
┌─────────────────────────────────────────────────────────┐  
│                  MLOps Pipeline                          │  
│  CI/CD → Testing → Building → Deployment → Monitoring   │  
└────────────────────┬────────────────────────────────────┘  
                     │  
                     ▼  
┌─────────────────────────────────────────────────────────┐  
│              Production Deployment                       │  
│  Docker Containers → Kubernetes → API → Dashboards      │  
└─────────────────────────────────────────────────────────┘

---

## **🚀 Quick Start**

### **Prerequisites**

* Python 3.10+  
* Docker (optional)  
* Git

### **Installation**

\# Clone repository  
git clone https://github.com/yourusername/health-mlops-project.git  
cd health-mlops-project

\# Create virtual environment  
python \-m venv venv  
source venv/bin/activate  \# Windows: venv\\Scripts\\activate

\# Install dependencies  
pip install \-r requirements.txt

### **Run the Full Pipeline**

\# 1\. Generate/Download data  
python src/download\_data.py

\# 2\. Feature engineering  
python src/feature\_engineering.py

\# 3\. Train models  
python src/train\_baseline.py  
python src/train\_pytorch.py

\# 4\. Evaluate  
python src/evaluate\_models.py

\# 5\. Start API server  
uvicorn src.api:app \--reload

\# 6\. Launch dashboard (new terminal)  
streamlit run src/dashboard.py

### **Using Docker**

\# Build image  
docker build \-t health-mlops .

\# Run container  
docker run \-p 8000:8000 \-p 8501:8501 health-mlops

\# Access:  
\# API: http://localhost:8000  
\# Dashboard: http://localhost:8501

---

## **📁 Project Structure**

health-mlops-project/  
│  
├── data/  
│   ├── raw/                  \# Raw data from sources  
│   │   ├── health/          \# Hospital node data  
│   │   ├── air\_quality/     \# Air quality sensors  
│   │   └── weather/         \# Weather stations  
│   ├── processed/           \# Processed datasets  
│   └── federated/           \# Node-specific data for FL  
│  
├── src/  
│   ├── download\_data.py          \# Data generation  
│   ├── feature\_engineering.py    \# Feature creation  
│   ├── train\_baseline.py         \# Baseline models  
│   ├── train\_pytorch.py          \# Neural networks  
│   ├── train\_federated.py        \# Federated learning  
│   ├── validate\_data.py          \# Data validation  
│   ├── check\_model\_metrics.py    \# Performance checks  
│   ├── api.py                    \# FastAPI server  
│   └── dashboard.py              \# Streamlit dashboard  
│  
├── models/  
│   ├── baseline/            \# Trained baseline models  
│   ├── pytorch/             \# PyTorch models  
│   ├── federated/           \# Federated models  
│   ├── evaluation/          \# Evaluation reports  
│   └── plots/               \# Visualizations  
│  
├── notebooks/  
│   ├── 01\_data\_collection.ipynb  
│   ├── 02\_eda\_analysis.ipynb  
│   └── 03\_model\_experiments.ipynb  
│  
├── tests/  
│   ├── test\_data.py  
│   ├── test\_models.py  
│   └── test\_api.py  
│  
├── .github/  
│   └── workflows/  
│       └── ci-cd.yml        \# CI/CD pipeline  
│  
├── docker/  
│   ├── Dockerfile  
│   └── docker-compose.yml  
│  
├── k8s/                     \# Kubernetes manifests  
│   ├── deployment.yaml  
│   └── service.yaml  
│  
├── docs/  
│   ├── research\_paper.md    \# Project paper  
│   ├── presentation.pdf     \# Slides  
│   └── api\_documentation.md  
│  
├── requirements.txt         \# Python dependencies  
├── setup.py                \# Package setup  
├── README.md               \# This file  
└── LICENSE                 \# MIT License

---

## **💻 Installation**

### **Option 1: Local Setup**

\# Install PyTorch (CPU version)  
pip install torch torchvision \--index-url https://download.pytorch.org/whl/cpu

\# Install all dependencies  
pip install \-r requirements.txt

\# Verify installation  
python \-c "import torch; import sklearn; import flwr; print('All imports successful\!')"

### **Option 2: Docker**

docker pull yourusername/health-mlops:latest  
docker run \-it health-mlops bash

### **Option 3: Conda**

conda create \-n health-mlops python=3.10  
conda activate health-mlops  
pip install \-r requirements.txt

---

## **📖 Usage**

### **1\. Data Generation**

from src.download\_data import generate\_health\_data

\# Generate data for 5 hospital nodes  
for node in range(1, 6):  
    data \= generate\_health\_data(node\_id=node, num\_patients=200, days=90)  
    data.to\_csv(f'data/raw/health/node\_{node}\_data.csv')

### **2\. Training Models**

from src.train\_baseline import train\_random\_forest, train\_xgboost

\# Train baseline models  
rf\_model \= train\_random\_forest(X\_train, y\_train)  
xgb\_model \= train\_xgboost(X\_train, y\_train)

\# Evaluate  
print(f"RF R²: {rf\_model.score(X\_test, y\_test):.4f}")  
print(f"XGB R²: {xgb\_model.score(X\_test, y\_test):.4f}")

### **3\. Federated Learning**

\# Start Flower server  
python src/train\_federated.py \--mode server \--rounds 50

\# Start clients (in separate terminals)  
python src/train\_federated.py \--mode client \--node 0  
python src/train\_federated.py \--mode client \--node 1  
\# ... for all 5 nodes

### **4\. API Usage**

import requests

\# Make prediction  
response \= requests.post(  
    "http://localhost:8000/predict",  
    json={  
        "heart\_rate": 85,  
        "spo2": 96,  
        "pm25": 45.2,  
        "temperature": 22.5,  
        \# ... other features  
    }  
)

prediction \= response.json()  
print(f"Health Risk Score: {prediction\['risk\_score'\]}")  
print(f"Risk Level: {prediction\['risk\_level'\]}")

### **5\. Dashboard Access**

streamlit run src/dashboard.py

\# Open browser: http://localhost:8501

---

## **🔄 CI/CD Pipeline**

Our GitHub Actions workflow automatically:

1. **Code Quality:**

   * Linting (flake8)  
   * Formatting (black)  
   * Unit tests (pytest)  
2. **Data Validation:**

   * Schema checks  
   * Data drift detection  
   * Quality reports  
3. **Model Training:**

   * Train all models  
   * Performance validation  
   * Threshold checks  
4. **Docker Build:**

   * Build images  
   * Run tests  
   * Push to registry  
5. **Deployment:**

   * Staging (develop branch)  
   * Production (main branch)  
   * Health checks  
6. **Monitoring:**

   * Track metrics  
   * Detect drift  
   * Send alerts

### **Trigger Pipeline**

\# Push to trigger  
git add .  
git commit \-m "Update model"  
git push origin main  \# Deploys to production

\# Or create PR  
git checkout \-b feature/new-model  
git push origin feature/new-model  
\# Open PR on GitHub

---

## **🚢 Deployment**

### **Local Development**

uvicorn src.api:app \--reload \--port 8000

### **Docker**

docker-compose up \-d

### **Kubernetes**

kubectl apply \-f k8s/  
kubectl get pods  
kubectl logs \-f deployment/health-mlops

### **Cloud Platforms**

**AWS:**

\# Deploy to ECS  
aws ecs create-service \--cluster health-mlops ...

**GCP:**

\# Deploy to Cloud Run  
gcloud run deploy health-mlops \--image gcr.io/...

**Azure:**

\# Deploy to AKS  
az aks create \--resource-group health-mlops ...

---

## **📊 Monitoring**

### **Model Metrics**

\# Check model performance  
python src/monitor\_performance.py

\# Outputs:  
\# ✓ Accuracy: 0.89 (threshold: 0.85)  
\# ✓ Latency p95: 45ms (threshold: 100ms)  
\# ⚠ Data drift detected in PM2.5 feature

### **Dashboard**

Access monitoring dashboard: `http://localhost:8501/monitoring`

**Tracked Metrics:**

* Prediction accuracy  
* Latency (p50, p95, p99)  
* Error rates  
* Data distribution shifts  
* Feature importance changes

---

## **📈 Results**

### **Model Performance**

| Model | Task | Metric | Score |
| ----- | ----- | ----- | ----- |
| Random Forest | Regression | R² | 0.759 |
| XGBoost | Regression | R² | 0.740 |
| PyTorch NN | Regression | R² | **0.780** |
| XGBoost | Classification | F1 | 0.667 |
| PyTorch NN | Classification | F1 | **0.700** |
| Federated (PyTorch) | Regression | R² | 0.750 |

### **Key Findings**

✅ **Best Model:** PyTorch Neural Network (R² \= 0.78) ✅ **Federated Learning:** Only 3% performance drop vs centralized ✅ **Top Features:** PM2.5, AQI, Temperature ✅ **Deployment:** \< 10 min from commit to production

---

## **🤝 Contributing**

We welcome contributions\! Please follow these steps:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit changes (`git commit -m 'Add amazing feature'`)  
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

### **Development Setup**

\# Install dev dependencies  
pip install \-r requirements-dev.txt

\# Run tests  
pytest tests/ \--cov

\# Check code quality  
black src/  
flake8 src/

---

## **📄 License**

This project is licensed under the MIT License \- see [LICENSE](https://claude.ai/chat/LICENSE) file for details.

---

## **📞 Contact**

**Project Maintainer:** \[Your Name\] **Email:** your.email@example.com **GitHub:** [@yourusername](https://github.com/yourusername)

---

## **🙏 Acknowledgments**

* Flower framework for Federated Learning  
* FastAPI for API development  
* Streamlit for dashboards  
* Open-source ML community

---

## **📚 Documentation**

* [Research Paper](https://claude.ai/chat/docs/research_paper.md)  
* [API Documentation](https://claude.ai/chat/docs/api_documentation.md)  
* [Deployment Guide](https://claude.ai/chat/docs/deployment.md)  
* [Presentation](https://claude.ai/chat/docs/presentation.pdf)

---

**⭐ If you find this project helpful, please star the repository\!**

