"""
CORRECTED Data Merging Script - Health MLOps Project
Based on successful diagnostic test
"""

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("CREATING MERGED DATASET - Health MLOps Project")
print("=" * 80)

# Create processed directory
os.makedirs('data/processed', exist_ok=True)

# ============================================================================
# STEP 1: Load all datasets from all nodes/cities
# ============================================================================
print("\n[STEP 1] Loading all datasets...")
print("-" * 80)

# Load Health Data from all 5 nodes
health_dfs = []
for node in range(1, 6):
    df = pd.read_csv(f'data/raw/health/node_{node}_health_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date  # Extract just the date
    health_dfs.append(df)
    print(f"✓ Loaded node_{node}_health_data.csv: {len(df):,} records")

df_health = pd.concat(health_dfs, ignore_index=True)
print(f"\n✓ Total Health Records: {len(df_health):,}")

# Load Air Quality Data from all 5 cities
cities = ['New_York', 'Los_Angeles', 'Chicago', 'Houston', 'Phoenix']
air_dfs = []
for city in cities:
    df = pd.read_csv(f'data/raw/air_quality/{city}_air_quality.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date_only'] = df['timestamp'].dt.date  # Extract just the date
    air_dfs.append(df)
    print(f"✓ Loaded {city}_air_quality.csv: {len(df):,} records")

df_air = pd.concat(air_dfs, ignore_index=True)
print(f"\n✓ Total Air Quality Records: {len(df_air):,}")

# Load Weather Data from all 5 cities
weather_dfs = []
for city in cities:
    df = pd.read_csv(f'data/raw/weather/{city}_weather.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date_only'] = df['timestamp'].dt.date  # Extract just the date
    weather_dfs.append(df)
    print(f"✓ Loaded {city}_weather.csv: {len(df):,} records")

df_weather = pd.concat(weather_dfs, ignore_index=True)
print(f"\n✓ Total Weather Records: {len(df_weather):,}")

# ============================================================================
# STEP 2: Map hospital nodes to city IDs
# ============================================================================
print("\n[STEP 2] Mapping hospitals to cities...")
print("-" * 80)

node_to_city = {
    'hospital_1': 'city_1',
    'hospital_2': 'city_2',
    'hospital_3': 'city_3',
    'hospital_4': 'city_4',
    'hospital_5': 'city_5'
}

df_health['city_id'] = df_health['node_id'].map(node_to_city)
print(f"✓ Mapped {df_health['city_id'].nunique()} hospital nodes to cities")

# ============================================================================
# STEP 3: Aggregate data by date and city
# ============================================================================
print("\n[STEP 3] Aggregating data by date and city...")
print("-" * 80)

# Aggregate health data - daily averages per city
print("Aggregating health data...")
health_daily = df_health.groupby(['date_only', 'city_id']).agg({
    'heart_rate': 'mean',
    'spo2': 'mean',
    'body_temp': 'mean',
    'steps': 'mean',
    'sleep_hours': 'mean',
    'health_status': lambda x: (x == 'sick').sum() / len(x) * 100,  # % sick
    'patient_id': 'count'  # number of records
}).reset_index()

health_daily.rename(columns={
    'date_only': 'date',
    'health_status': 'sick_percentage',
    'patient_id': 'num_records'
}, inplace=True)

# Convert date back to datetime for consistency
health_daily['date'] = pd.to_datetime(health_daily['date'])

print(f"✓ Health daily: {len(health_daily):,} records")
print(f"  Date range: {health_daily['date'].min().date()} to {health_daily['date'].max().date()}")

# Aggregate air quality data - daily averages per city
print("\nAggregating air quality data...")
air_daily = df_air.groupby(['date_only', 'city_id']).agg({
    'pm25': 'mean',
    'pm10': 'mean',
    'no2': 'mean',
    'co': 'mean',
    'o3': 'mean',
    'aqi': 'mean'
}).reset_index()

air_daily.rename(columns={'date_only': 'date'}, inplace=True)
air_daily['date'] = pd.to_datetime(air_daily['date'])

print(f"✓ Air quality daily: {len(air_daily):,} records")
print(f"  Date range: {air_daily['date'].min().date()} to {air_daily['date'].max().date()}")

# Aggregate weather data - daily averages per city
print("\nAggregating weather data...")
weather_daily = df_weather.groupby(['date_only', 'city_id']).agg({
    'temperature': 'mean',
    'humidity': 'mean',
    'pressure': 'mean',
    'wind_speed': 'mean',
    'precipitation': 'sum'  # total daily precipitation
}).reset_index()

weather_daily.rename(columns={'date_only': 'date'}, inplace=True)
weather_daily['date'] = pd.to_datetime(weather_daily['date'])

print(f"✓ Weather daily: {len(weather_daily):,} records")
print(f"  Date range: {weather_daily['date'].min().date()} to {weather_daily['date'].max().date()}")

# ============================================================================
# STEP 4: Merge all datasets
# ============================================================================
print("\n[STEP 4] Merging datasets...")
print("-" * 80)

# Merge health + air quality
merged = health_daily.merge(air_daily, on=['date', 'city_id'], how='inner')
print(f"✓ After health + air quality: {len(merged):,} records")

# Merge with weather
merged = merged.merge(weather_daily, on=['date', 'city_id'], how='inner')
print(f"✓ After adding weather: {len(merged):,} records")

if len(merged) == 0:
    print("\n❌ ERROR: Merge resulted in empty dataset!")
    print("This shouldn't happen based on the diagnostic. Please check the data.")
    exit()

print(f"\n✅ Successfully merged all datasets!")

# ============================================================================
# STEP 5: Add derived features
# ============================================================================
print("\n[STEP 5] Adding derived features...")
print("-" * 80)

# Time-based features
merged['day_of_week'] = merged['date'].dt.dayofweek
merged['day_name'] = merged['date'].dt.day_name()
merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
merged['month'] = merged['date'].dt.month
merged['week_of_year'] = merged['date'].dt.isocalendar().week

# Season (Northern Hemisphere)
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

merged['season'] = merged['month'].apply(get_season)

# Pollution severity (based on EPA AQI standards)
def get_pollution_level(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

merged['pollution_level'] = merged['aqi'].apply(get_pollution_level)

# Temperature categories
def get_temp_category(temp):
    if temp < 10:
        return 'Cold'
    elif temp < 20:
        return 'Cool'
    elif temp < 25:
        return 'Moderate'
    elif temp < 30:
        return 'Warm'
    else:
        return 'Hot'

merged['temp_category'] = merged['temperature'].apply(get_temp_category)

# Health risk score (simple composite)
merged['health_risk_score'] = (
    (merged['sick_percentage'] / 100) * 0.4 +
    (merged['aqi'] / 500) * 0.3 +
    (merged['body_temp'] - 36.5) / 3 * 0.3
).clip(0, 1)

# Binary target: High risk if sick_percentage > 15%
merged['high_risk'] = (merged['sick_percentage'] > 15).astype(int)

print("✓ Added derived features:")
print("  - Time features: day_of_week, is_weekend, month, season")
print("  - Categorical: pollution_level, temp_category")
print("  - Target variables: health_risk_score, high_risk")

# ============================================================================
# STEP 6: Reorder columns and save
# ============================================================================
print("\n[STEP 6] Organizing and saving dataset...")
print("-" * 80)

# Reorder columns for better readability
column_order = [
    # Identifiers and time
    'date', 'city_id', 'day_of_week', 'day_name', 'is_weekend', 
    'month', 'week_of_year', 'season',
    
    # Target variables
    'sick_percentage', 'health_risk_score', 'high_risk',
    
    # Health metrics
    'num_records', 'heart_rate', 'spo2', 'body_temp', 'steps', 'sleep_hours',
    
    # Air quality
    'pm25', 'pm10', 'no2', 'co', 'o3', 'aqi', 'pollution_level',
    
    # Weather
    'temperature', 'temp_category', 'humidity', 'pressure', 
    'wind_speed', 'precipitation'
]

merged = merged[column_order]

# Round numeric columns for readability
numeric_cols = ['sick_percentage', 'health_risk_score', 'heart_rate', 'spo2', 
                'body_temp', 'steps', 'sleep_hours', 'pm25', 'pm10', 'no2', 
                'co', 'o3', 'aqi', 'temperature', 'humidity', 'pressure', 
                'wind_speed', 'precipitation']

for col in numeric_cols:
    merged[col] = merged[col].round(2)

# Save to CSV
output_file = 'data/processed/merged_daily_data.csv'
merged.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# ============================================================================
# STEP 7: Summary and statistics
# ============================================================================
print("\n" + "=" * 80)
print("MERGED DATASET SUMMARY")
print("=" * 80)

print(f"\n📊 Dataset Overview:")
print(f"  • Total Records: {len(merged):,}")
print(f"  • Date Range: {merged['date'].min().date()} to {merged['date'].max().date()}")
print(f"  • Number of Cities: {merged['city_id'].nunique()}")
print(f"  • Number of Features: {len(merged.columns)}")
print(f"  • File Size: {os.path.getsize(output_file) / 1024:.2f} KB")

print(f"\n🏥 Health Statistics:")
print(f"  • Average Sick %: {merged['sick_percentage'].mean():.2f}%")
print(f"  • High Risk Days: {merged['high_risk'].sum()} ({merged['high_risk'].mean()*100:.1f}%)")
print(f"  • Avg Heart Rate: {merged['heart_rate'].mean():.1f} bpm")
print(f"  • Avg SpO2: {merged['spo2'].mean():.1f}%")

print(f"\n🌫️ Air Quality Statistics:")
print(f"  • Average PM2.5: {merged['pm25'].mean():.2f} µg/m³")
print(f"  • Average AQI: {merged['aqi'].mean():.2f}")
print(f"  • Pollution Levels:")
for level in merged['pollution_level'].value_counts().sort_index().items():
    print(f"    - {level[0]}: {level[1]} days")

print(f"\n🌤️ Weather Statistics:")
print(f"  • Avg Temperature: {merged['temperature'].mean():.1f}°C")
print(f"  • Avg Humidity: {merged['humidity'].mean():.1f}%")
print(f"  • Total Precipitation: {merged['precipitation'].sum():.2f} mm")

print(f"\n📈 Sample Data (First 10 rows):")
print(merged.head(10).to_string())

print(f"\n📊 Basic Statistics:")
print(merged.describe().to_string())

print("\n" + "=" * 80)
print("✅ MERGE COMPLETE! Dataset ready for modeling")
print("=" * 80)
print(f"\n📁 Saved to: {output_file}")
print("\nNext steps:")
print("  1. Exploratory Data Analysis with visualizations")
print("  2. Feature engineering and selection")
print("  3. Prepare for Federated Learning")
print("  4. Build and train models")