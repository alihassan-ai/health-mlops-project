"""
Quick preview of downloaded datasets
"""

import pandas as pd
import os

print("=" * 70)
print("DATASET PREVIEW")
print("=" * 70)

# Check if data exists
if not os.path.exists('data/raw/health'):
    print("\n❌ No data found! Please run src/download_data.py first")
    exit()

# Preview Health Data
print("\n[1] HEALTH DATA (Hospital Node 1)")
print("-" * 70)
df_health = pd.read_csv('data/raw/health/node_1_health_data.csv')
print(df_health.head(10))
print(f"\nShape: {df_health.shape[0]} rows × {df_health.shape[1]} columns")
print(f"Columns: {list(df_health.columns)}")
print(f"\nHealth Status Distribution:")
print(df_health['health_status'].value_counts())

# Preview Air Quality Data
print("\n" + "=" * 70)
print("[2] AIR QUALITY DATA (New York)")
print("-" * 70)
df_air = pd.read_csv('data/raw/air_quality/New_York_air_quality.csv')
print(df_air.head(10))
print(f"\nShape: {df_air.shape[0]} rows × {df_air.shape[1]} columns")
print(f"Columns: {list(df_air.columns)}")
print(f"\nAir Quality Stats:")
print(df_air[['pm25', 'pm10', 'aqi']].describe())

# Preview Weather Data
print("\n" + "=" * 70)
print("[3] WEATHER DATA (New York)")
print("-" * 70)
df_weather = pd.read_csv('data/raw/weather/New_York_weather.csv')
print(df_weather.head(10))
print(f"\nShape: {df_weather.shape[0]} rows × {df_weather.shape[1]} columns")
print(f"Columns: {list(df_weather.columns)}")
print(f"\nWeather Stats:")
print(df_weather[['temperature', 'humidity', 'precipitation']].describe())

print("\n" + "=" * 70)
print("✓ Data preview complete!")
print("=" * 70)