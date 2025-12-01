"""
Fixed Data Merging Script - Health MLOps Project
Properly merges health, air quality, and weather data
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("FIXING DATA MERGE - Health MLOps Project")
print("=" * 80)

# ============================================================================
# STEP 1: Load all datasets
# ============================================================================
print("\n[STEP 1] Loading datasets...")

# Load Health Data
health_dfs = []
for node in range(1, 6):
    df = pd.read_csv(f'data/raw/health/node_{node}_health_data.csv')
    health_dfs.append(df)
    print(f"✓ Loaded node_{node}: {len(df):,} records")

df_health = pd.concat(health_dfs, ignore_index=True)
print(f"\n✓ Total Health Records: {len(df_health):,}")

# Load Air Quality Data
cities = ['New_York', 'Los_Angeles', 'Chicago', 'Houston', 'Phoenix']
air_dfs = []
for city in cities:
    df = pd.read_csv(f'data/raw/air_quality/{city}_air_quality.csv')
    air_dfs.append(df)

df_air = pd.concat(air_dfs, ignore_index=True)
print(f"✓ Total Air Quality Records: {len(df_air):,}")

# Load Weather Data
weather_dfs = []
for city in cities:
    df = pd.read_csv(f'data/raw/weather/{city}_weather.csv')
    weather_dfs.append(df)

df_weather = pd.concat(weather_dfs, ignore_index=True)
print(f"✓ Total Weather Records: {len(df_weather):,}")

# ============================================================================
# STEP 2: Debug - Check data formats
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 2] Checking data formats...")
print("=" * 80)

print("\n--- Health Data Sample ---")
print(df_health[['node_id', 'date', 'patient_id']].head())
print(f"\nHealth date type: {df_health['date'].dtype}")
print(f"Unique nodes: {df_health['node_id'].unique()}")

print("\n--- Air Quality Data Sample ---")
print(df_air[['city_id', 'timestamp']].head())
print(f"\nAir timestamp type: {df_air['timestamp'].dtype}")
print(f"Unique cities: {df_air['city_id'].unique()}")

print("\n--- Weather Data Sample ---")
print(df_weather[['city_id', 'timestamp']].head())
print(f"\nWeather timestamp type: {df_weather['timestamp'].dtype}")

# ============================================================================
# STEP 3: Fix date formats and city mappings
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 3] Converting dates and mapping cities...")
print("=" * 80)

# Convert health dates to datetime
df_health['date'] = pd.to_datetime(df_health['date'])
print(f"✓ Converted health dates to datetime")

# Map hospital nodes to city IDs
node_to_city = {
    'hospital_1': 'city_1',
    'hospital_2': 'city_2',
    'hospital_3': 'city_3',
    'hospital_4': 'city_4',
    'hospital_5': 'city_5'
}

df_health['city_id'] = df_health['node_id'].map(node_to_city)
print(f"✓ Mapped hospitals to cities")
print(f"  Unique city_ids in health data: {df_health['city_id'].unique()}")

# Convert air quality timestamps to datetime and extract date
df_air['timestamp'] = pd.to_datetime(df_air['timestamp'])
df_air['date'] = df_air['timestamp'].dt.date
df_air['date'] = pd.to_datetime(df_air['date'])
print(f"✓ Extracted dates from air quality timestamps")

# Convert weather timestamps to datetime and extract date
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
df_weather['date'] = df_weather['timestamp'].dt.date
df_weather['date'] = pd.to_datetime(df_weather['date'])
print(f"✓ Extracted dates from weather timestamps")

# ============================================================================
# STEP 4: Aggregate data by date and city
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 4] Aggregating data by date and city...")
print("=" * 80)

# Aggregate health data
print("\nAggregating health data...")
health_daily = df_health.groupby(['date', 'city_id']).agg({
    'heart_rate': 'mean',
    'spo2': 'mean',
    'body_temp': 'mean',
    'steps': 'mean',
    'sleep_hours': 'mean',
    'health_status': lambda x: (x == 'sick').sum() / len(x) * 100,  # % sick
    'patient_id': 'count'  # number of patients
}).reset_index()

# Rename columns for clarity
health_daily.rename(columns={
    'health_status': 'sick_percentage',
    'patient_id': 'num_patients'
}, inplace=True)

print(f"✓ Health daily aggregated: {len(health_daily)} records")
print(f"  Date range: {health_daily['date'].min()} to {health_daily['date'].max()}")
print(f"  Cities: {health_daily['city_id'].unique()}")

# Aggregate air quality data
print("\nAggregating air quality data...")
air_daily = df_air.groupby(['date', 'city_id']).agg({
    'pm25': 'mean',
    'pm10': 'mean',
    'no2': 'mean',
    'co': 'mean',
    'o3': 'mean',
    'aqi': 'mean'
}).reset_index()

print(f"✓ Air quality daily aggregated: {len(air_daily)} records")
print(f"  Date range: {air_daily['date'].min()} to {air_daily['date'].max()}")
print(f"  Cities: {air_daily['city_id'].unique()}")

# Aggregate weather data
print("\nAggregating weather data...")
weather_daily = df_weather.groupby(['date', 'city_id']).agg({
    'temperature': 'mean',
    'humidity': 'mean',
    'pressure': 'mean',
    'wind_speed': 'mean',
    'precipitation': 'sum'  # total daily precipitation
}).reset_index()

print(f"✓ Weather daily aggregated: {len(weather_daily)} records")
print(f"  Date range: {weather_daily['date'].min()} to {weather_daily['date'].max()}")
print(f"  Cities: {weather_daily['city_id'].unique()}")

# ============================================================================
# STEP 5: Merge all datasets
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Merging all datasets...")
print("=" * 80)

# Check for overlapping dates and cities before merge
print("\nChecking data compatibility...")
health_keys = set(zip(health_daily['date'], health_daily['city_id']))
air_keys = set(zip(air_daily['date'], air_daily['city_id']))
weather_keys = set(zip(weather_daily['date'], weather_daily['city_id']))

print(f"Health unique (date, city) pairs: {len(health_keys)}")
print(f"Air quality unique (date, city) pairs: {len(air_keys)}")
print(f"Weather unique (date, city) pairs: {len(weather_keys)}")

common_keys = health_keys & air_keys & weather_keys
print(f"Common (date, city) pairs across all datasets: {len(common_keys)}")

# Perform merges
print("\nPerforming inner merge on health + air quality...")
merged = health_daily.merge(air_daily, on=['date', 'city_id'], how='inner')
print(f"✓ After health+air merge: {len(merged)} records")

print("\nPerforming inner merge with weather...")
merged = merged.merge(weather_daily, on=['date', 'city_id'], how='inner')
print(f"✓ After adding weather: {len(merged)} records")

if len(merged) == 0:
    print("\n⚠️ WARNING: Merge resulted in 0 records!")
    print("\nDEBUGGING INFO:")
    
    # Show sample dates from each dataset
    print("\nSample dates from Health:")
    print(health_daily[['date', 'city_id']].head(10))
    
    print("\nSample dates from Air Quality:")
    print(air_daily[['date', 'city_id']].head(10))
    
    print("\nSample dates from Weather:")
    print(weather_daily[['date', 'city_id']].head(10))
    
    # Try left merge to see what's missing
    print("\nTrying LEFT merge to diagnose...")
    test_merge = health_daily.merge(air_daily, on=['date', 'city_id'], how='left', indicator=True)
    print(test_merge['_merge'].value_counts())
    
else:
    print("\n✅ SUCCESS! Merged dataset created")
    
    # ============================================================================
    # STEP 6: Add additional features
    # ============================================================================
    print("\n" + "=" * 80)
    print("[STEP 6] Adding derived features...")
    print("=" * 80)
    
    # Add day of week
    merged['day_of_week'] = merged['date'].dt.dayofweek
    merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
    
    # Add month
    merged['month'] = merged['date'].dt.month
    
    # Add season (simplified for Northern Hemisphere)
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
    
    # Add pollution severity categories
    merged['pollution_level'] = pd.cut(merged['aqi'], 
                                       bins=[0, 50, 100, 150, 200, 300, 500],
                                       labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 
                                              'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    
    # Add temperature categories
    merged['temp_category'] = pd.cut(merged['temperature'],
                                     bins=[-np.inf, 10, 20, 25, 30, np.inf],
                                     labels=['Cold', 'Cool', 'Moderate', 'Warm', 'Hot'])
    
    print("✓ Added derived features:")
    print("  - day_of_week, is_weekend")
    print("  - month, season")
    print("  - pollution_level")
    print("  - temp_category")
    
    # ============================================================================
    # STEP 7: Save merged dataset
    # ============================================================================
    print("\n" + "=" * 80)
    print("[STEP 7] Saving merged dataset...")
    print("=" * 80)
    
    # Reorder columns for better readability
    column_order = [
        'date', 'city_id', 'day_of_week', 'is_weekend', 'month', 'season',
        # Health metrics
        'sick_percentage', 'num_patients', 'heart_rate', 'spo2', 'body_temp', 'steps', 'sleep_hours',
        # Air quality
        'pm25', 'pm10', 'no2', 'co', 'o3', 'aqi', 'pollution_level',
        # Weather
        'temperature', 'temp_category', 'humidity', 'pressure', 'wind_speed', 'precipitation'
    ]
    
    merged = merged[column_order]
    
    # Save to CSV
    output_file = 'data/processed/merged_daily_data.csv'
    merged.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    # ============================================================================
    # STEP 8: Summary statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("MERGED DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Total Records: {len(merged):,}")
    print(f"📅 Date Range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"🏙️ Cities: {merged['city_id'].nunique()}")
    print(f"📏 Features: {len(merged.columns)}")
    
    print("\n🔢 Dataset Info:")
    print(merged.info())
    
    print("\n📈 Sample Data:")
    print(merged.head(10))
    
    print("\n📊 Basic Statistics:")
    print(merged.describe())
    
    print("\n✅ Merged dataset is ready for modeling!")
    print("=" * 80)