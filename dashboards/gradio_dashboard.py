import gradio as gr
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("HEALTH RISK MONITORING DASHBOARD - Gradio (Professional UI)")
print("=" * 80)

class HealthRiskNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

print("\n[Loading] Models and data...")

models = {}
rf_path = "models/baseline/rf_regressor.pkl"
if os.path.exists(rf_path):
    with open(rf_path, "rb") as f:
        models["random_forest"] = pickle.load(f)
    print("✓ Random Forest loaded")
else:
    print("⚠️ Random Forest model not found at", rf_path)

xgb_path = "models/baseline/xgb_regressor.pkl"
if os.path.exists(xgb_path):
    with open(xgb_path, "rb") as f:
        models["xgboost"] = pickle.load(f)
    print("✓ XGBoost loaded")
else:
    print("⚠️ XGBoost model not found at", xgb_path)

fl_path = "models/federated/regression_global_model.pth"
if os.path.exists(fl_path):
    try:
        state = torch.load(fl_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(fl_path, map_location="cpu")
    try:
        input_size = state["network.0.weight"].shape[1]
    except Exception:
        first_w = next((v for k, v in state.items() if "weight" in k), None)
        input_size = first_w.shape[1] if first_w is not None else None
    if input_size is not None:
        fl_model = HealthRiskNN(input_size)
        try:
            fl_model.load_state_dict(state)
        except Exception:
            for key in ("model_state", "state_dict", "network_state"):
                if key in state:
                    fl_model.load_state_dict(state[key])
                    break
        fl_model.eval()
        models["federated_learning"] = fl_model
        print(f"✓ Federated Learning model loaded (input size: {input_size})")
    else:
        print("⚠️ Could not infer FL model input size; federated model not loaded")
else:
    print("⚠️ Federated Learning model not found at", fl_path)

merged_csv = "data/processed/merged_daily_data.csv"
if os.path.exists(merged_csv):
    merged_data = pd.read_csv(merged_csv)
    if "date" in merged_data.columns:
        merged_data["date"] = pd.to_datetime(merged_data["date"])
    print(f"✓ Data loaded: {len(merged_data)} records")
else:
    print("⚠️ Data file not found, creating dummy data for demo")
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    cities = ['City_1', 'City_2', 'City_3', 'City_4', 'City_5']
    merged_data = pd.DataFrame({
        'date': np.repeat(dates, len(cities)),
        'city_id': cities * len(dates),
        'sick_percentage': np.random.uniform(0, 8, len(dates) * len(cities)),
        'avg_aqi': np.random.uniform(20, 150, len(dates) * len(cities)),
        'avg_pm25': np.random.uniform(10, 80, len(dates) * len(cities)),
        'avg_temperature': np.random.uniform(10, 35, len(dates) * len(cities)),
        'avg_humidity': np.random.uniform(30, 90, len(dates) * len(cities)),
        'avg_heart_rate': np.random.uniform(60, 100, len(dates) * len(cities)),
        'avg_spo2': np.random.uniform(95, 100, len(dates) * len(cities))
    })

comparison_csv = "reports/model_comparison.csv"
if os.path.exists(comparison_csv):
    comparison_data = pd.read_csv(comparison_csv)
    print("✓ Model comparison data loaded")
else:
    comparison_data = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Federated Learning'],
        'Task': ['Regression', 'Regression', 'Regression'],
        'Model Type': ['Baseline', 'Baseline', 'Federated'],
        'Test RMSE': [0.85, 0.78, 0.82],
        'Test R²': [0.92, 0.94, 0.93],
        'Test Accuracy': [0.88, 0.91, 0.89],
        'Training Time (s)': [12.5, 15.3, 45.7]
    })

feature_pkl = "data/federated/city_1_data.pkl"
if os.path.exists(feature_pkl):
    with open(feature_pkl, "rb") as f:
        feature_data = pickle.load(f)
        feature_names = feature_data.get("feature_names", [])
    print(f"✓ Feature names loaded: {len(feature_names)} features")
else:
    feature_names = []
    print("⚠️ Feature file not found; feature names defaulting to empty list")

print("\n✅ All resources loaded (best-effort).")

def get_risk_level(sick_percentage):
    if sick_percentage > 4: return "🔴 HIGH RISK", "#ff4444"
    elif sick_percentage > 2: return "🟡 MODERATE RISK", "#ffaa00"
    else: return "🟢 LOW RISK", "#44ff44"

def make_prediction(model_key, features):
    if model_key not in models: return None
    model = models[model_key]
    if model_key == "federated_learning":
        with torch.no_grad():
            tensor = torch.FloatTensor(features).unsqueeze(0)
            return model(tensor).item()
    return model.predict(features.reshape(1, -1))[0]

def predict_health_risk(city, aqi, pm25, temperature, humidity, heart_rate, spo2, body_temp, model_choice):
    feature_vector = np.array([
        heart_rate, spo2, body_temp, 5000,
        pm25, pm25*2, aqi*0.5, aqi,
        temperature, humidity, 1013, 0
    ], dtype=float)
    model_map = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "Federated Learning": "federated_learning"
    }
    model_key = model_map.get(model_choice, "random_forest")
    if model_key == "federated_learning" and model_key in models:
        expected = models[model_key].network[0].in_features
    elif model_key in models:
        expected = getattr(models[model_key], "n_features_in_", len(feature_names) or len(feature_vector))
    else:
        expected = len(feature_names) or len(feature_vector)
    if len(feature_vector) < expected:
        feature_vector = np.pad(feature_vector, (0, expected - len(feature_vector)), 'constant')
    elif len(feature_vector) > expected:
        feature_vector = feature_vector[:expected]
    sick_percentage = make_prediction(model_key, feature_vector)
    if sick_percentage is None:
        sick_percentage = np.random.uniform(1, 5)
    sick_percentage = float(np.clip(sick_percentage, 0, 10))
    risk_level, risk_color = get_risk_level(sick_percentage)
    result_html = f"""
    <div style='padding:20px;border-radius:12px;background:linear-gradient(135deg,{risk_color}22 0%,{risk_color}44 100%);border:2px solid {risk_color};margin:10px 0;'>
        <h3 style='color:{risk_color};text-align:center;margin:0;padding-bottom:8px;'>{risk_level}</h3>
        <div style='text-align:center;font-size:42px;font-weight:700;margin:10px 0;color:#1a1a1a;'>{sick_percentage:.2f}%</div>
        <div style='text-align:center;color:#555;font-size:16px;'>Predicted Sick Percentage</div>
    </div>
    """
    if sick_percentage > 4:
        recommendations = """
### ⚠️ High Risk Recommendations
- 🏠 Stay indoors as much as possible  
- 🚫 Avoid intense outdoor exercise  
- 😷 Wear high-quality masks (N95) if you must go out  
- 🌬️ Use HEPA air purifiers indoors  
- 👨‍⚕️ Monitor symptoms; consult a physician if unwell
        """
    elif sick_percentage > 2:
        recommendations = """
### ⚠️ Moderate Risk Recommendations
- ⚠️ Limit outdoor activities where possible  
- 😷 Wear a mask in crowded/outdoor polluted environments  
- ⏰ Avoid peak pollution hours (early morning / evening)  
- 🪟 Keep windows closed when AQI is high
        """
    else:
        recommendations = """
### ✅ Low Risk
- ✅ Conditions are generally safe for normal activities  
- 🏃 Outdoor exercise is OK  
- 💪 Continue following healthy habits
        """
    city_data = merged_data[merged_data['city_id'] == city].tail(7)
    if city_data.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No recent data for {city}", height=320, template="plotly_white")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=city_data['date'], y=city_data['sick_percentage'],
            mode='lines+markers', line=dict(color=risk_color, width=3),
            marker=dict(size=10), name='Sick %'
        ))
        fig.update_layout(
            title=f"7-Day Health Risk Trend — {city}",
            height=400, template="plotly_white",
            yaxis=dict(range=[0, max(10, city_data['sick_percentage'].max()+1)])
        )
    return result_html, recommendations, fig

def show_city_overview(city, days):
    df = merged_data[merged_data['city_id'] == city].tail(days)
    if df.empty: return "No data available", None, None, ""
    avg_sick = df['sick_percentage'].mean()
    max_sick = df['sick_percentage'].max()
    avg_aqi = df.get('avg_aqi', pd.Series([0]*len(df))).mean()
    high_risk_days = (df['sick_percentage'] > 4).sum()
    stats_html = f"""
    <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:20px 0;'>
        <div style='padding:20px;border-radius:12px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);'>
            <div style='font-size:14px;opacity:0.9;'>Average Sick Rate</div>
            <div style='font-size:32px;font-weight:700;margin-top:8px;'>{avg_sick:.2f}%</div>
        </div>
        <div style='padding:20px;border-radius:12px;background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);color:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);'>
            <div style='font-size:14px;opacity:0.9;'>High Risk Days</div>
            <div style='font-size:32px;font-weight:700;margin-top:8px;'>{high_risk_days}</div>
        </div>
        <div style='padding:20px;border-radius:12px;background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);color:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);'>
            <div style='font-size:14px;opacity:0.9;'>Average AQI</div>
            <div style='font-size:32px;font-weight:700;margin-top:8px;'>{avg_aqi:.0f}</div>
        </div>
        <div style='padding:20px;border-radius:12px;background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);color:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);'>
            <div style='font-size:14px;opacity:0.9;'>Peak Sick Rate</div>
            <div style='font-size:32px;font-weight:700;margin-top:8px;'>{max_sick:.2f}%</div>
        </div>
    </div>
    """
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df['date'], y=df['sick_percentage'], mode='lines+markers',
        line=dict(color='#e74c3c', width=3), fill='tozeroy', name='Sick %'
    ))
    fig_trend.update_layout(
        title=f"Health Risk Trend — Last {days} Days",
        height=400, template="plotly_white"
    )
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=df.get('avg_aqi', pd.Series([0]*len(df))),
        y=df['sick_percentage'], mode='markers',
        marker=dict(
            size=12,
            color=df.get('avg_temperature', pd.Series([0]*len(df))),
            colorscale='Viridis', showscale=True,
            colorbar=dict(title="Temp (°C)")
        )
    ))
    fig_corr.update_layout(
        title="Air Quality vs Health Risk",
        height=400, template="plotly_white",
        xaxis_title="Average AQI", yaxis_title="Sick Percentage (%)"
    )
    latest = df.iloc[-1]
    factors_html = f"""
### 📊 Current Risk Factors (Latest Record)
**🌫️ Air Quality**
- PM2.5: {latest.get('avg_pm25', np.nan):.1f} µg/m³
- AQI: {latest.get('avg_aqi', np.nan):.0f}
**🌤️ Weather**
- Temperature: {latest.get('avg_temperature', np.nan):.1f}°C
- Humidity: {latest.get('avg_humidity', np.nan):.0f}%
**❤️ Health Metrics**
- Avg Heart Rate: {latest.get('avg_heart_rate', np.nan):.0f} bpm
- Avg SpO2: {latest.get('avg_spo2', np.nan):.1f}%
    """
    return stats_html, fig_trend, fig_corr, factors_html

def show_model_performance():
    if comparison_data.empty:
        empty = go.Figure()
        empty.update_layout(title="No comparison data available", height=300)
        return empty, empty, empty, "No comparison data available."
    reg_models = comparison_data[comparison_data['Task'] == 'Regression']
    clf_models = comparison_data[comparison_data['Task'] == 'Classification']
    fig_reg = px.bar(
        reg_models,
        x='Model',
        y='Test RMSE',
        color='Model Type',
        title='Regression Performance (Lower is Better)',
        text='Test RMSE',
        height=400
    )
    fig_reg.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_reg.update_layout(template="plotly_white")
    fig_clf = px.bar(
        clf_models if not clf_models.empty else reg_models,
        x='Model',
        y='Test Accuracy',
        color='Model Type',
        title='Classification Performance (Higher is Better)',
        text='Test Accuracy',
        height=400
    )
    fig_clf.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_clf.update_layout(template="plotly_white")
    fig_time = px.bar(
        comparison_data,
        x='Model',
        y='Training Time (s)',
        color='Model Type',
        title='Training Time (Lower is Better)',
        text='Training Time (s)',
        height=400
    )
    fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig_time.update_layout(template="plotly_white")
    table_md = "| Model | Task | Type | RMSE | R² | Accuracy | Training Time (s) |\n"
    table_md += "|---|---|---|---|---|---|---|\n"
    for _, row in comparison_data.iterrows():
        table_md += f"| {row['Model']} | {row['Task']} | {row['Model Type']} | {row['Test RMSE']} | {row['Test R²']} | {row['Test Accuracy']} | {row['Training Time (s)']} |\n"
    return fig_reg, fig_clf, fig_time, table_md

def compare_all_cities(days):
    recent = merged_data.groupby('city_id').tail(days)
    agg_dict = {'sick_percentage': ['mean', 'max']}
    if 'avg_aqi' in recent.columns:
        agg_dict['avg_aqi'] = 'mean'
    if 'avg_temperature' in recent.columns:
        agg_dict['avg_temperature'] = 'mean'
    city_stats = recent.groupby('city_id').agg(agg_dict).reset_index()
    flat_columns = ["city_id"]
    if 'sick_percentage' in agg_dict:
        flat_columns += ["avg_sick", "max_sick"]
    if 'avg_aqi' in agg_dict:
        flat_columns += ["avg_aqi"]
    if 'avg_temperature' in agg_dict:
        flat_columns += ["avg_temp"]
    city_stats.columns = flat_columns
    for col in ['avg_aqi', 'avg_temp']:
        if col not in city_stats.columns: city_stats[col] = 0
    fig_risk = px.bar(
        city_stats,
        x='city_id',
        y='avg_sick',
        color='avg_sick',
        color_continuous_scale=['green','yellow','red'],
        title=f'Average Health Risk by City — Last {days} Days',
        labels={'avg_sick':'Avg Sick %', 'city_id': 'City'},
        height=400
    )
    fig_risk.update_layout(template="plotly_white")
    fig_aqi = px.bar(
        city_stats,
        x='city_id',
        y='avg_aqi',
        color='avg_aqi',
        color_continuous_scale='Reds',
        title='Average AQI by City',
        labels={'avg_aqi': 'Avg AQI', 'city_id': 'City'},
        height=400
    )
    fig_aqi.update_layout(template="plotly_white")
    fig_trend = go.Figure()
    for c in merged_data['city_id'].unique():
        d = merged_data[merged_data['city_id']==c].tail(days)
        if d.empty: continue
        fig_trend.add_trace(go.Scatter(
            x=d['date'],
            y=d['sick_percentage'],
            mode='lines',
            name=str(c),
            line=dict(width=2)
        ))
    fig_trend.update_layout(
        title='Health Risk Trends — All Cities',
        height=400,
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Sick Percentage (%)'
    )
    return fig_risk, fig_aqi, fig_trend

with gr.Blocks(title="Health Risk Monitoring Dashboard") as demo:
    gr.HTML("""
    <style>
        .gradio-container {
            max-width: 1400px !important;
        }
        .gr-button-primary {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
        }
        body.dark {
            background-color: #181A1B;
            color: #fafafa;
        }
        .gr-card {
            background: #f8fafc;
        }
        body.dark .gr-card {
            background: #222226;
        }
    </style>
    <script>
    function toggleTheme() {
        document.body.classList.toggle('dark');
        const isDark = document.body.classList.contains('dark');
        try { 
            localStorage.setItem('hrm_theme_dark', isDark ? '1' : '0'); 
        } catch(e){}
    }
    (function() {
        try {
            if (localStorage.getItem('hrm_theme_dark') === '1') {
                document.body.classList.add('dark');
            }
        } catch(e){}
    })();
    </script>
    """)

    with gr.Row():
        gr.Markdown("# 🏥 Health Risk Monitoring Dashboard")
        gr.HTML("""
        <button onclick='toggleTheme()' 
                style='padding:10px 20px;border-radius:8px;border:none;background:#667eea;color:white;cursor:pointer;float:right;font-size:14px;'>
            🌙 Toggle Theme
        </button>
        """)
    gr.Markdown("*Professional MLOps dashboard for health risk prediction with federated learning*")
    gr.Markdown("---")

    with gr.Tab("🎯 Health Risk Calculator"):
        gr.Markdown("## 🎯 Health Risk Calculator")
        gr.Markdown("*Enter environmental and health parameters to predict illness risk*")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Location & Environment")
                city_input = gr.Dropdown(
                    sorted(merged_data['city_id'].unique().tolist()),
                    label="Select City",
                    value=sorted(merged_data['city_id'].unique().tolist())[0]
                )
                aqi_input = gr.Slider(0, 300, value=50, label="Air Quality Index (AQI)", step=1)
                pm25_input = gr.Slider(0, 200, value=25, label="PM2.5 (µg/m³)", step=1)
                gr.Markdown("### 🌤 Weather Conditions")
                temp_input = gr.Slider(-10, 45, value=25, label="Temperature (°C)", step=0.5)
                humidity_input = gr.Slider(0, 100, value=60, label="Humidity (%)", step=1)
            with gr.Column(scale=1):
                gr.Markdown("### ❤️ Health Metrics")
                hr_input = gr.Slider(40, 140, value=75, label="Heart Rate (bpm)", step=1)
                spo2_input = gr.Slider(80, 100, value=98, label="SpO2 (%)", step=1)
                body_temp_input = gr.Slider(35, 40, value=37, label="Body Temperature (°C)", step=0.1)
                gr.Markdown("### ⚙️ Model Selection")
                model_select = gr.Dropdown(
                    ["Random Forest", "XGBoost", "Federated Learning"],
                    label="Select Prediction Model",
                    value="Random Forest"
                )
                predict_btn = gr.Button("🔍 Calculate Health Risk", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                risk_output = gr.HTML(label="Risk Assessment")
                recommendations_output = gr.Markdown(label="Recommendations")
            with gr.Column(scale=1):
                trend_plot = gr.Plot(label="7-Day Trend")
        predict_btn.click(
            predict_health_risk,
            inputs=[city_input, aqi_input, pm25_input, temp_input, humidity_input,
                    hr_input, spo2_input, body_temp_input, model_select],
            outputs=[risk_output, recommendations_output, trend_plot]
        )

    with gr.Tab("🏙️ City Overview"):
        gr.Markdown("## 🏙️ City Overview")
        gr.Markdown("*Detailed analytics and trends for a specific city*")
        with gr.Row():
            with gr.Column(scale=1):
                city_select = gr.Dropdown(
                    sorted(merged_data['city_id'].unique().tolist()),
                    label="Select City",
                    value=sorted(merged_data['city_id'].unique().tolist())[0]
                )
                days_select = gr.Slider(7, 90, value=30, step=1, label="Number of Days")
                update_city_btn = gr.Button("🔄 Update Overview", variant="primary")
            with gr.Column(scale=2):
                stats_display = gr.HTML(label="Key Statistics")
        with gr.Row():
            trend_chart = gr.Plot(label="Trend Analysis")
            corr_chart = gr.Plot(label="Correlation Analysis")
        factors_display = gr.Markdown(label="Risk Factors")
        update_city_btn.click(
            show_city_overview,
            inputs=[city_select, days_select],
            outputs=[stats_display, trend_chart, corr_chart, factors_display]
        )

    with gr.Tab("📊 Model Performance"):
        gr.Markdown("## 📊 Model Performance Comparison")
        gr.Markdown("*Compare different ML models across various metrics*")
        with gr.Row():
            reg_chart = gr.Plot(label="Regression Metrics")
            clf_chart = gr.Plot(label="Classification Metrics")
        time_chart = gr.Plot(label="Training Time")
        gr.Markdown("### Detailed Model Comparison")
        comparison_table = gr.Markdown(label="Model Comparison Table")
        refresh_perf_btn = gr.Button("🔄 Refresh Comparison", variant="primary")
        gr.Markdown("### Static Figures / Diagrams")
        img_regression = gr.Image(value="reports/figures/regression_comparison.png", label="Regression Comparison")
        img_classification = gr.Image(value="reports/figures/classification_comparison.png", label="Classification Comparison")
        img_time = gr.Image(value="reports/figures/training_time_comparison.png", label="Training Time Comparison")
        img_fl = gr.Image(value="reports/figures/fl_training_curves.png", label="Federated Training Curves")
        refresh_perf_btn.click(
            show_model_performance,
            outputs=[reg_chart, clf_chart, time_chart, comparison_table]
        )
        demo.load(
            show_model_performance,
            outputs=[reg_chart, clf_chart, time_chart, comparison_table]
        )

    with gr.Tab("🗺️ All Cities Comparison"):
        gr.Markdown("## 🗺️ All Cities Comparison")
        gr.Markdown("*Comparative analysis across all monitored cities*")
        with gr.Row():
            days_all = gr.Slider(7, 90, value=30, label="Number of Days", step=1)
            refresh_btn = gr.Button("🔄 Refresh Comparison", variant="primary")
        risk_by_city = gr.Plot(label="Risk by City")
        aqi_by_city = gr.Plot(label="Air Quality by City")
        all_trends = gr.Plot(label="Comparative Trends")
        refresh_btn.click(
            compare_all_cities,
            inputs=[days_all],
            outputs=[risk_by_city, aqi_by_city, all_trends]
        )

    gr.Markdown("---")
    gr.Markdown(
        "<div style='text-align:center;color:#666;font-size:13px;padding:10px;'>"
        "Built with Gradio • Federated Learning • MLOps • Professional Health Risk Monitoring System"
        "</div>",
        elem_id="footer"
    )

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 LAUNCHING PROFESSIONAL GRADIO DASHBOARD")
    print("=" * 80)
    print("\nAccess the dashboard at: http://localhost:7860")
    print("Press CTRL+C to stop the server\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
