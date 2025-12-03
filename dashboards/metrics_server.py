#!/usr/bin/env python3
"""
Prometheus Metrics Server for Health MLOps Dashboard
Exposes metrics on port 8000 for Prometheus to scrape
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY
from flask import Flask, Response
import random
import time
from threading import Thread

app = Flask(__name__)

# Define metrics
predictions_total = Counter('predictions_total', 'Total number of predictions made')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy', ['model'])
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction latency')
http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['status'])
service_up = Gauge('up', 'Service health status', ['job'])

# Initialize metrics
model_accuracy.labels(model='random_forest').set(0.92)
model_accuracy.labels(model='federated_learning').set(0.93)
service_up.labels(job='health-mlops-dashboard').set(1)

# Simulate some activity for demo purposes
def simulate_metrics():
    """Generate some realistic metrics for demonstration"""
    while True:
        # Simulate predictions being made
        predictions_total.inc(random.randint(0, 3))

        # Simulate model accuracy fluctuations (realistic range)
        model_accuracy.labels(model='random_forest').set(0.92 + random.uniform(-0.02, 0.02))
        model_accuracy.labels(model='federated_learning').set(0.93 + random.uniform(-0.02, 0.02))

        # Simulate HTTP requests
        http_requests_total.labels(status='200').inc(random.randint(1, 5))
        if random.random() < 0.05:  # 5% error rate
            http_requests_total.labels(status='500').inc(1)

        time.sleep(5)  # Update every 5 seconds

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'service': 'health-mlops-metrics'}, 200

if __name__ == '__main__':
    # Start background thread to simulate metrics
    metrics_thread = Thread(target=simulate_metrics, daemon=True)
    metrics_thread.start()

    print("=" * 80)
    print("PROMETHEUS METRICS SERVER")
    print("=" * 80)
    print("\nMetrics endpoint: http://localhost:8000/metrics")
    print("Health endpoint:  http://localhost:8000/health")
    print("\nPrometheus will scrape metrics from this endpoint.")
    print("Press CTRL+C to stop\n")

    app.run(host='0.0.0.0', port=8000, debug=False)
