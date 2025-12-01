"""
Diagnostic Script - Find out why merge is failing
"""

import pandas as pd
import os

print("=" * 80)
print("DATA DIAGNOSIS - Finding the merge issue")
print("=" * 80)

# Check if files exist
print("\n[CHECK 1] Verifying files exist...")
print("-" * 80)

files_to_check = [
    'data/raw/health/node_1_health_data.csv',
    'data/raw/air_quality/New_York_air_quality.csv',
    'data/raw/weather/New_York_weather.csv'
]

for file in files_to_check:
    exists = os.path.exists(file)
    print(f"{'✓' if exists else '✗'} {file}: {'EXISTS' if exists else 'NOT FOUND'}")

if not all(os.path.exists(f) for f in files_to_check):
    print("\n❌ Some files are missing! Run src/download_data.py first")
    exit()

# Load sample data
print("\n[CHECK 2] Loading sample data...")
print("-" * 80)

df_health = pd.read_csv('data/raw/health/node_1_health_data.csv')
df_air = pd.read_csv('data/raw/air_quality/New_York_air_quality.csv')
df_weather = pd.read_csv('data/raw/weather/New_York_weather.csv')

print(f"✓ Health data loaded: {len(df_health)} rows")
print(f"✓ Air quality loaded: {len(df_air)} rows")
print(f"✓ Weather loaded: {len(df_weather)} rows")

# Check columns
print("\n[CHECK 3] Column names...")
print("-" * 80)
print(f"Health columns: {list(df_health.columns)}")
print(f"Air columns: {list(df_air.columns)}")
print(f"Weather columns: {list(df_weather.columns)}")

# Check date formats
print("\n[CHECK 4] Date/Timestamp samples...")
print("-" * 80)

print("\nHealth 'date' column:")
print(df_health['date'].head())
print(f"Type: {df_health['date'].dtype}")

print("\nAir Quality 'timestamp' column:")
print(df_air['timestamp'].head())
print(f"Type: {df_air['timestamp'].dtype}")

print("\nWeather 'timestamp' column:")
print(df_weather['timestamp'].head())
print(f"Type: {df_weather['timestamp'].dtype}")

# Convert and compare
print("\n[CHECK 5] Converting dates...")
print("-" * 80)

df_health['date'] = pd.to_datetime(df_health['date'])
df_air['timestamp'] = pd.to_datetime(df_air['timestamp'])
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])

print(f"Health date range: {df_health['date'].min()} to {df_health['date'].max()}")
print(f"Air timestamp range: {df_air['timestamp'].min()} to {df_air['timestamp'].max()}")
print(f"Weather timestamp range: {df_weather['timestamp'].min()} to {df_weather['timestamp'].max()}")

# Extract dates for air and weather
df_air['date'] = df_air['timestamp'].dt.date
df_weather['date'] = df_weather['timestamp'].dt.date
df_health['date_only'] = df_health['date'].dt.date

print("\nAfter extracting date component:")
print(f"Health dates: {df_health['date_only'].min()} to {df_health['date_only'].max()}")
print(f"Air dates: {df_air['date'].min()} to {df_air['date'].max()}")
print(f"Weather dates: {df_weather['date'].min()} to {df_weather['date'].max()}")

# Check for overlap
print("\n[CHECK 6] Checking date overlap...")
print("-" * 80)

health_dates = set(df_health['date_only'])
air_dates = set(df_air['date'])
weather_dates = set(df_weather['date'])

print(f"Unique health dates: {len(health_dates)}")
print(f"Unique air dates: {len(air_dates)}")
print(f"Unique weather dates: {len(weather_dates)}")

common_dates = health_dates & air_dates & weather_dates
print(f"\n✓ Common dates across all three: {len(common_dates)}")

if len(common_dates) > 0:
    print(f"Sample common dates: {sorted(list(common_dates))[:5]}")
else:
    print("\n❌ NO COMMON DATES FOUND!")
    print("\nThis is the problem. Let's investigate...")
    
    print("\nSample health dates:")
    print(sorted(list(health_dates))[:5])
    
    print("\nSample air dates:")
    print(sorted(list(air_dates))[:5])
    
    print("\nSample weather dates:")
    print(sorted(list(weather_dates))[:5])

# Check city IDs
print("\n[CHECK 7] Checking city/node IDs...")
print("-" * 80)

print(f"Health node_id values: {df_health['node_id'].unique()}")
print(f"Air city_id values: {df_air['city_id'].unique()}")
print(f"Weather city_id values: {df_weather['city_id'].unique()}")

# Try a simple aggregation and merge
print("\n[CHECK 8] Attempting simple merge...")
print("-" * 80)

# Add city mapping to health
df_health['city_id'] = 'city_1'  # Since this is New York data (node 1)

# Aggregate
health_daily = df_health.groupby(['date_only', 'city_id']).agg({
    'heart_rate': 'mean'
}).reset_index()
health_daily.rename(columns={'date_only': 'date'}, inplace=True)

air_daily = df_air.groupby(['date', 'city_id']).agg({
    'pm25': 'mean'
}).reset_index()

weather_daily = df_weather.groupby(['date', 'city_id']).agg({
    'temperature': 'mean'
}).reset_index()

print(f"Health daily: {len(health_daily)} rows")
print(f"Air daily: {len(air_daily)} rows")
print(f"Weather daily: {len(weather_daily)} rows")

print("\nHealth daily sample:")
print(health_daily.head())

print("\nAir daily sample:")
print(air_daily.head())

print("\nWeather daily sample:")
print(weather_daily.head())

# Convert date columns to same type
health_daily['date'] = pd.to_datetime(health_daily['date'])
air_daily['date'] = pd.to_datetime(air_daily['date'])
weather_daily['date'] = pd.to_datetime(weather_daily['date'])

print("\nAttempting merge...")
merged = health_daily.merge(air_daily, on=['date', 'city_id'], how='inner')
print(f"After health+air: {len(merged)} rows")

if len(merged) > 0:
    merged = merged.merge(weather_daily, on=['date', 'city_id'], how='inner')
    print(f"After adding weather: {len(merged)} rows")
    
    if len(merged) > 0:
        print("\n✅ MERGE SUCCESSFUL!")
        print("\nMerged data sample:")
        print(merged.head())
        print("\n✓ The data CAN be merged. The issue was in the main script.")
        print("I'll create a fixed version now...")
    else:
        print("\n❌ Weather merge failed")
else:
    print("\n❌ Health+Air merge failed")
    print("\nChecking what's different...")
    
    print("\nHealth daily dates:")
    print(sorted(health_daily['date'].unique())[:5])
    
    print("\nAir daily dates:")
    print(sorted(air_daily['date'].unique())[:5])

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)