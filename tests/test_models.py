"""
Model Tests - Health MLOps Project
"""
import pytest
import os
import pickle
import torch


class TestModelArtifacts:
    """Test trained model artifacts"""

    def test_baseline_models_exist(self):
        """Check if baseline models are trained and saved"""
        models = [
            'models/baseline/rf_regressor.pkl',
            'models/baseline/xgb_regressor.pkl',
            'models/baseline/rf_classifier.pkl',
            'models/baseline/xgb_classifier.pkl'
        ]

        for model_path in models:
            assert os.path.exists(model_path), f"Model missing: {model_path}"

    def test_pytorch_models_exist(self):
        """Check if PyTorch models are trained and saved"""
        models = [
            'models/pytorch/best_regression_nn.pth',
            'models/pytorch/best_classification_nn.pth'
        ]

        for model_path in models:
            assert os.path.exists(model_path), f"PyTorch model missing: {model_path}"

    def test_federated_models_exist(self):
        """Check if federated learning models exist"""
        models = [
            'models/federated/regression_global_model.pth',
            'models/federated/classification_global_model.pth'
        ]

        for model_path in models:
            assert os.path.exists(model_path), f"Federated model missing: {model_path}"

    def test_model_loading(self):
        """Test if models can be loaded without errors"""
        # Test pickle model loading
        if os.path.exists('models/baseline/rf_regressor.pkl'):
            with open('models/baseline/rf_regressor.pkl', 'rb') as f:
                model = pickle.load(f)
                assert model is not None, "Failed to load Random Forest model"

        # Test PyTorch model loading
        if os.path.exists('models/pytorch/best_regression_nn.pth'):
            state = torch.load('models/pytorch/best_regression_nn.pth',
                             map_location='cpu', weights_only=False)
            assert state is not None, "Failed to load PyTorch model"


class TestModelPerformance:
    """Test model performance metrics"""

    def test_evaluation_reports_exist(self):
        """Check if evaluation reports are generated"""
        reports = [
            'models/evaluation/baseline_report.txt',
            'models/evaluation/pytorch_report.txt'
        ]

        for report_path in reports:
            assert os.path.exists(report_path), f"Report missing: {report_path}"

    def test_baseline_performance_threshold(self):
        """Check if baseline models meet minimum performance threshold"""
        if os.path.exists('models/evaluation/baseline_results.csv'):
            import pandas as pd
            results = pd.read_csv('models/evaluation/baseline_results.csv')

            # Check if R² score is above threshold
            if 'r2_test' in results.columns:
                max_r2 = results['r2_test'].max()
                assert max_r2 >= 0.70, f"Model R² ({max_r2:.3f}) below threshold (0.70)"


class TestModelPredictions:
    """Test model prediction capabilities"""

    def test_model_can_predict(self):
        """Test if models can make predictions"""
        if os.path.exists('models/baseline/rf_regressor.pkl') and \
           os.path.exists('data/processed/centralized_data.pkl'):

            import numpy as np

            # Load model
            with open('models/baseline/rf_regressor.pkl', 'rb') as f:
                model = pickle.load(f)

            # Load test data
            with open('data/processed/centralized_data.pkl', 'rb') as f:
                data = pickle.load(f)
                X_test = data['X_test_scaled'][:5]  # Take 5 samples

            # Make predictions
            predictions = model.predict(X_test)

            assert len(predictions) == 5, "Prediction count mismatch"
            assert not np.isnan(predictions).any(), "Predictions contain NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
