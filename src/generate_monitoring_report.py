"""
Monitoring Report Generator - Health MLOps Project
Generates comprehensive HTML monitoring reports
"""

import pandas as pd
import os
from datetime import datetime


def generate_html_report():
    """Generate HTML monitoring report"""

    print("=" * 80)
    print("GENERATING MONITORING REPORT")
    print("=" * 80)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Health MLOps Monitoring Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .success {{
            background-color: #d4edda;
            border-left-color: #28a745;
        }}
        .error {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Health MLOps Monitoring Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 Model Performance</h2>
"""

    # Load and display model performance
    baseline_path = 'models/evaluation/baseline_results.csv'
    if os.path.exists(baseline_path):
        df = pd.read_csv(baseline_path)

        html_content += """
        <table>
            <tr>
                <th>Model</th>
                <th>R² Score</th>
                <th>RMSE</th>
                <th>MAE</th>
            </tr>
"""
        for idx, row in df.iterrows():
            model = row.get('model', f'Model {idx}')
            r2 = row.get('r2_test', 0)
            rmse = row.get('rmse_test', 0)
            mae = row.get('mae_test', 0)

            html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{r2:.4f}</td>
                <td>{rmse:.4f}</td>
                <td>{mae:.4f}</td>
            </tr>
"""

        html_content += "        </table>\n"

        # Best model highlight
        best_r2 = df['r2_test'].max() if 'r2_test' in df.columns else 0
        status_class = 'success' if best_r2 >= 0.75 else 'warning'

        html_content += f"""
        <div class="metric {status_class}">
            <strong>Best Model R²:</strong> {best_r2:.4f}
            <br>
            <strong>Status:</strong> {'✅ Excellent' if best_r2 >= 0.75 else '⚠️ Needs Monitoring'}
        </div>
"""
    else:
        html_content += """
        <div class="metric error">
            ❌ Model performance data not found
        </div>
"""

    # Data drift section
    html_content += """
        <h2>📈 Data Drift Status</h2>
"""

    drift_path = 'reports/drift/drift_report.csv'
    if os.path.exists(drift_path):
        df_drift = pd.read_csv(drift_path)
        significant_drift = df_drift[df_drift['p_value'] < 0.05] if 'p_value' in df_drift.columns else pd.DataFrame()

        if len(significant_drift) > 0:
            html_content += f"""
        <div class="metric warning">
            ⚠️ Drift detected in {len(significant_drift)} features
        </div>
        <table>
            <tr>
                <th>Feature</th>
                <th>P-Value</th>
                <th>Status</th>
            </tr>
"""
            for idx, row in significant_drift.head(10).iterrows():
                feature = row.get('feature', 'Unknown')
                p_val = row.get('p_value', 0)

                html_content += f"""
            <tr>
                <td>{feature}</td>
                <td>{p_val:.4f}</td>
                <td>⚠️ Significant</td>
            </tr>
"""
            html_content += "        </table>\n"
        else:
            html_content += """
        <div class="metric success">
            ✅ No significant data drift detected
        </div>
"""
    else:
        html_content += """
        <div class="metric warning">
            ⚠️ Drift analysis not yet performed
        </div>
"""

    # Recommendations section
    html_content += """
        <h2>💡 Recommendations</h2>
        <div class="metric">
            <ul>
                <li>Monitor model performance weekly</li>
                <li>Retrain models if R² drops below 0.70</li>
                <li>Investigate features showing significant drift</li>
                <li>Keep federated learning nodes synchronized</li>
                <li>Review data quality metrics regularly</li>
            </ul>
        </div>

        <h2>📋 System Health</h2>
"""

    # Check system health
    checks = [
        ('Data Files', os.path.exists('data/processed/features_engineered.csv')),
        ('Baseline Models', os.path.exists('models/baseline/rf_regressor.pkl')),
        ('PyTorch Models', os.path.exists('models/pytorch/best_regression_nn.pth')),
        ('Federated Models', os.path.exists('models/federated/regression_global_model.pth')),
        ('Evaluation Reports', os.path.exists('models/evaluation/baseline_report.txt'))
    ]

    html_content += "        <table>\n"
    html_content += "            <tr><th>Component</th><th>Status</th></tr>\n"

    for check_name, check_status in checks:
        status_icon = '✅' if check_status else '❌'
        html_content += f"            <tr><td>{check_name}</td><td>{status_icon}</td></tr>\n"

    html_content += "        </table>\n"

    # Close HTML
    html_content += """
    </div>
</body>
</html>
"""

    # Save report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/monitoring_report.html'

    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"\n✓ HTML report generated: {report_path}")
    print(f"  Open in browser: file://{os.path.abspath(report_path)}")
    print("=" * 80)

    return report_path


if __name__ == '__main__':
    generate_html_report()
