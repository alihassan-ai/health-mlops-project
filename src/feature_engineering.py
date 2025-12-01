"""
Feature Engineering & Federated Learning Data Preparation
Health MLOps Project - PyTorch Compatible
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING & FEDERATED LEARNING PREPARATION (PyTorch)")
print("=" * 80)

# ============================================================================
# STEP 1: Load Merged Data
# ============================================================================
print("\n[STEP 1] Loading merged dataset...")
print("-" * 80)

df = pd.read_csv('data/processed/merged_daily_data.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"✓ Loaded {len(df):,} records")
print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"✓ Cities: {df['city_id'].nunique()}")
print(f"✓ Features: {len(df.columns)}")

# ============================================================================
# STEP 2: Create Advanced Features
# ============================================================================
print("\n[STEP 2] Creating advanced features...")
print("-" * 80)

# -------- Rolling Window Features (Time Series) --------
print("Creating rolling window features...")

# Sort by city and date
df = df.sort_values(['city_id', 'date']).reset_index(drop=True)

# Create lagged features (previous day values)
lag_features = []
for col in ['sick_percentage', 'pm25', 'aqi', 'temperature']:
    df[f'{col}_lag1'] = df.groupby('city_id')[col].shift(1)
    df[f'{col}_lag3'] = df.groupby('city_id')[col].shift(3)
    df[f'{col}_lag7'] = df.groupby('city_id')[col].shift(7)
    lag_features.extend([f'{col}_lag1', f'{col}_lag3', f'{col}_lag7'])

print("✓ Created lagged features (1, 3, 7 days)")

# Rolling averages
rolling_features = []
for col in ['sick_percentage', 'pm25', 'aqi', 'temperature']:
    df[f'{col}_rolling_3d'] = df.groupby('city_id')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df[f'{col}_rolling_7d'] = df.groupby('city_id')[col].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    rolling_features.extend([f'{col}_rolling_3d', f'{col}_rolling_7d'])

print("✓ Created rolling averages (3, 7 days)")

# Rate of change
change_features = []
for col in ['sick_percentage', 'pm25', 'aqi']:
    df[f'{col}_change'] = df.groupby('city_id')[col].diff()
    df[f'{col}_pct_change'] = df.groupby('city_id')[col].pct_change()
    change_features.extend([f'{col}_change', f'{col}_pct_change'])

print("✓ Created rate of change features")

# -------- Interaction Features --------
print("\nCreating interaction features...")

interaction_features = []

# Pollution x Temperature
df['pollution_temp_interaction'] = df['aqi'] * df['temperature']
interaction_features.append('pollution_temp_interaction')

# Pollution x Humidity
df['pollution_humidity_interaction'] = df['pm25'] * df['humidity']
interaction_features.append('pollution_humidity_interaction')

# Weekend x Pollution
df['weekend_pollution'] = df['is_weekend'] * df['aqi']
interaction_features.append('weekend_pollution')

# Temperature extremes
df['temp_extreme'] = ((df['temperature'] < 10) | (df['temperature'] > 30)).astype(int)
interaction_features.append('temp_extreme')

# Pollution spike indicator
df['pollution_spike'] = (df['aqi'] > df['aqi'].quantile(0.75)).astype(int)
interaction_features.append('pollution_spike')

print("✓ Created interaction features")

# -------- Aggregated City-Level Features --------
print("\nCreating city-level aggregate features...")

# Calculate city-level statistics
city_stats = df.groupby('city_id').agg({
    'pm25': ['mean', 'std'],
    'temperature': ['mean', 'std'],
    'sick_percentage': ['mean', 'std']
}).round(2)

city_stats.columns = ['_'.join(col) for col in city_stats.columns]
city_stats = city_stats.reset_index()

# Merge back to main dataframe
df = df.merge(city_stats, on='city_id', how='left')

city_features = [col for col in df.columns if col.endswith('_mean') or col.endswith('_std')]
print(f"✓ Created {len(city_features)} city-level statistics")

# -------- Health Indicator Combinations --------
print("\nCreating health indicator combinations...")

health_combo_features = []

# Respiratory stress indicator
df['respiratory_stress'] = (
    (df['heart_rate'] > 80) & 
    (df['spo2'] < 95)
).astype(int)
health_combo_features.append('respiratory_stress')

# Critical health zone
df['critical_health'] = (
    (df['body_temp'] > 37.5) & 
    (df['heart_rate'] > 85)
).astype(int)
health_combo_features.append('critical_health')

# Health deterioration
df['health_deterioration'] = (
    df['sick_percentage'] > df['sick_percentage_rolling_7d']
).astype(int)
health_combo_features.append('health_deterioration')

print("✓ Created health indicator combinations")

# -------- Weather Impact Features --------
print("\nCreating weather impact features...")

weather_combo_features = []

# Uncomfortable weather
df['uncomfortable_weather'] = (
    ((df['temperature'] < 10) | (df['temperature'] > 30)) |
    (df['humidity'] > 80) |
    (df['precipitation'] > 10)
).astype(int)
weather_combo_features.append('uncomfortable_weather')

# Weather stress score
df['weather_stress'] = (
    np.abs(df['temperature'] - 20) / 10 +
    df['humidity'] / 100 +
    (df['precipitation'] > 0).astype(int)
) / 3
weather_combo_features.append('weather_stress')

print("✓ Created weather impact features")

print(f"\n✓ Total features after engineering: {len(df.columns)}")

# ============================================================================
# STEP 3: Handle Missing Values
# ============================================================================
print("\n[STEP 3] Handling missing values...")
print("-" * 80)

print(f"Missing values before handling:")
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
if len(missing_counts) > 0:
    print(missing_counts.to_string())
else:
    print("  No missing values!")

# Fill missing values in lagged/rolling features
for col in lag_features + rolling_features + change_features:
    if col in df.columns:
        df[col] = df.groupby('city_id')[col].bfill()
        df[col] = df.groupby('city_id')[col].ffill()

# Fill any remaining NaN with 0
df = df.fillna(0)

# Replace inf values
df = df.replace([np.inf, -np.inf], 0)

print(f"✓ Missing values after handling: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 4: Select Features for Modeling
# ============================================================================
print("\n[STEP 4] Selecting features for modeling...")
print("-" * 80)

# Define feature categories
time_features = ['day_of_week', 'is_weekend', 'month', 'week_of_year']

health_features = [
    'num_records', 'heart_rate', 'spo2', 'body_temp', 'steps', 'sleep_hours'
] + health_combo_features

air_quality_features = [
    'pm25', 'pm10', 'no2', 'co', 'o3', 'aqi'
] + [f for f in lag_features if 'pm25' in f or 'aqi' in f] + \
    [f for f in rolling_features if 'pm25' in f or 'aqi' in f] + \
    [f for f in change_features if 'pm25' in f or 'aqi' in f] + \
    ['pollution_spike']

weather_features = [
    'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation'
] + [f for f in lag_features if 'temperature' in f] + \
    [f for f in rolling_features if 'temperature' in f] + \
    ['temp_extreme'] + weather_combo_features

other_interaction_features = [
    'pollution_temp_interaction', 'pollution_humidity_interaction',
    'weekend_pollution', 'health_deterioration'
]

# Combine all features
feature_columns = (
    time_features + 
    health_features + 
    air_quality_features + 
    weather_features + 
    other_interaction_features + 
    city_features
)

# Remove duplicates
feature_columns = list(dict.fromkeys(feature_columns))

# Target variables
target_regression = 'sick_percentage'
target_classification = 'high_risk'

print(f"✓ Selected {len(feature_columns)} features for modeling")
print(f"  - Time features: {len(time_features)}")
print(f"  - Health features: {len(health_features)}")
print(f"  - Air quality features: {len([f for f in feature_columns if any(x in f for x in ['pm', 'aqi', 'no2', 'co', 'o3', 'pollution'])])}")
print(f"  - Weather features: {len([f for f in feature_columns if any(x in f for x in ['temp', 'humidity', 'pressure', 'wind', 'precipitation', 'weather'])])}")

# ============================================================================
# STEP 5: Encode Categorical Variables
# ============================================================================
print("\n[STEP 5] Encoding categorical variables...")
print("-" * 80)

# Encode season
season_encoder = LabelEncoder()
df['season_encoded'] = season_encoder.fit_transform(df['season'])
feature_columns.append('season_encoded')

# Encode pollution level
pollution_encoder = LabelEncoder()
df['pollution_level_encoded'] = pollution_encoder.fit_transform(df['pollution_level'])
feature_columns.append('pollution_level_encoded')

# Encode temperature category
temp_encoder = LabelEncoder()
df['temp_category_encoded'] = temp_encoder.fit_transform(df['temp_category'])
feature_columns.append('temp_category_encoded')

# Encode city_id
city_encoder = LabelEncoder()
df['city_encoded'] = city_encoder.fit_transform(df['city_id'])
feature_columns.append('city_encoded')

print("✓ Encoded 4 categorical variables")

# Save encoders
os.makedirs('models/encoders', exist_ok=True)
encoders = {
    'season': season_encoder,
    'pollution_level': pollution_encoder,
    'temp_category': temp_encoder,
    'city': city_encoder
}

for name, encoder in encoders.items():
    with open(f'models/encoders/{name}_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

print("✓ Saved encoders to models/encoders/")

# ============================================================================
# STEP 6: Save Feature-Engineered Dataset
# ============================================================================
print("\n[STEP 6] Saving feature-engineered dataset...")
print("-" * 80)

df.to_csv('data/processed/features_engineered.csv', index=False)
print(f"✓ Saved: data/processed/features_engineered.csv")
print(f"  Total columns: {len(df.columns)}")
print(f"  Feature columns: {len(feature_columns)}")

# ============================================================================
# STEP 7: Prepare Data for Federated Learning (PyTorch format)
# ============================================================================
print("\n[STEP 7] Preparing data for Federated Learning (PyTorch)...")
print("-" * 80)

# Create directory for federated data
os.makedirs('data/federated', exist_ok=True)

# Split data by city
cities = sorted(df['city_id'].unique())

print(f"Creating {len(cities)} federated learning nodes...")

node_stats = []

for i, city in enumerate(cities):
    # Filter data for this city
    city_data = df[df['city_id'] == city].copy()
    
    # Prepare features and targets
    X = city_data[feature_columns].values.astype(np.float32)
    y_regression = city_data[target_regression].values.astype(np.float32)
    y_classification = city_data[target_classification].values.astype(np.int64)
    dates = city_data['date'].values
    
    # Split into train/test (80/20)
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_reg_train = y_regression[:n_train]
    y_reg_test = y_regression[n_train:]
    y_cls_train = y_classification[:n_train]
    y_cls_test = y_classification[n_train:]
    dates_train = dates[:n_train]
    dates_test = dates[n_train:]
    
    # Save node data (NumPy arrays for PyTorch compatibility)
    node_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_regression_train': y_reg_train,
        'y_regression_test': y_reg_test,
        'y_classification_train': y_cls_train,
        'y_classification_test': y_cls_test,
        'feature_names': feature_columns,
        'dates_train': dates_train,
        'dates_test': dates_test,
        'city_id': city,
        'node_id': i
    }
    
    # Save as pickle
    with open(f'data/federated/node_{i}_{city}.pkl', 'wb') as f:
        pickle.dump(node_data, f)
    
    node_stats.append({
        'node_id': i,
        'city': city,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': len(feature_columns)
    })
    
    print(f"✓ Node {i} ({city}): Train={len(X_train)}, Test={len(X_test)}")

# Save node statistics
node_stats_df = pd.DataFrame(node_stats)
node_stats_df.to_csv('data/federated/node_statistics.csv', index=False)

print(f"\n✓ Created {len(cities)} federated learning nodes")
print("✓ Saved to data/federated/")

# ============================================================================
# STEP 8: Create Centralized Dataset
# ============================================================================
print("\n[STEP 8] Creating centralized dataset...")
print("-" * 80)

# Prepare full dataset
X_full = df[feature_columns].values.astype(np.float32)
y_regression_full = df[target_regression].values.astype(np.float32)
y_classification_full = df[target_classification].values.astype(np.int64)

# Split into train/test (80/20, stratified)
X_train_full, X_test_full, y_reg_train_full, y_reg_test_full, y_cls_train_full, y_cls_test_full = train_test_split(
    X_full, y_regression_full, y_classification_full,
    test_size=0.2,
    random_state=42,
    stratify=y_classification_full
)

print(f"✓ Centralized train set: {len(X_train_full)} samples")
print(f"✓ Centralized test set: {len(X_test_full)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# Save centralized data (PyTorch compatible - NumPy arrays)
centralized_data = {
    'X_train': X_train_full,
    'X_test': X_test_full,
    'X_train_scaled': X_train_scaled.astype(np.float32),
    'X_test_scaled': X_test_scaled.astype(np.float32),
    'y_regression_train': y_reg_train_full,
    'y_regression_test': y_reg_test_full,
    'y_classification_train': y_cls_train_full,
    'y_classification_test': y_cls_test_full,
    'feature_names': feature_columns,
    'scaler': scaler,
    'num_features': len(feature_columns)
}

with open('data/processed/centralized_data.pkl', 'wb') as f:
    pickle.dump(centralized_data, f)

print("✓ Saved: data/processed/centralized_data.pkl")

# Save feature names
with open('data/processed/feature_names.txt', 'w') as f:
    for i, feature in enumerate(feature_columns):
        f.write(f"{i+1}. {feature}\n")

print("✓ Saved: data/processed/feature_names.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)

print(f"\n📊 Dataset Summary:")
print(f"  • Total samples: {len(df):,}")
print(f"  • Total features: {len(feature_columns)}")
print(f"  • Feature data type: float32 (PyTorch compatible)")
print(f"  • Target (Regression): {target_regression}")
print(f"  • Target (Classification): {target_classification}")

print(f"\n🎯 Target Distribution:")
print(f"  • Avg sick percentage: {df[target_regression].mean():.2f}%")
print(f"  • Std sick percentage: {df[target_regression].std():.2f}%")
print(f"  • High risk days: {df[target_classification].sum()} ({df[target_classification].mean()*100:.1f}%)")
print(f"  • Low risk days: {(1-df[target_classification]).sum()} ({(1-df[target_classification]).mean()*100:.1f}%)")

print(f"\n🌐 Federated Learning Setup:")
print(f"  • Number of nodes: {len(cities)}")
print(f"  • Nodes: {', '.join(cities)}")
print(f"  • Data format: NumPy arrays (PyTorch compatible)")
print(f"  • Data split: 80% train, 20% test per node")

print(f"\n📁 Files Created:")
print(f"  • data/processed/features_engineered.csv")
print(f"  • data/processed/centralized_data.pkl")
print(f"  • data/processed/feature_names.txt")
print(f"  • data/federated/node_*_*.pkl (×{len(cities)})")
print(f"  • data/federated/node_statistics.csv")
print(f"  • models/encoders/*.pkl (×4)")

print("\n✅ Data ready for PyTorch modeling!")
print("\nNext steps:")
print("  1. Train baseline models (Sklearn: RF, XGBoost)")
print("  2. Train PyTorch neural networks")
print("  3. Implement Federated Learning with Flower")
print("  4. Compare all approaches")

print("\n" + "=" * 80)