"""
Health MLOps Project - Data Collection Script
Downloads open datasets for health monitoring, air quality, and weather
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create data directories if they don't exist
os.makedirs('data/raw/health', exist_ok=True)
os.makedirs('data/raw/air_quality', exist_ok=True)
os.makedirs('data/raw/weather', exist_ok=True)

print("=" * 60)
print("HEALTH MLOPS PROJECT - DATA COLLECTION")
print("=" * 60)

# ============================================================================
# 1. SIMULATED WEARABLE/HEALTH DATA (Multiple Nodes/Hospitals)
# ============================================================================
print("\n[1/3] Generating simulated wearable health data from 5 nodes...")

def generate_health_data(node_id, num_patients=1000, days=90):
    """
    Simulate wearable health data from a hospital/clinic node
    Features: heart_rate, steps, sleep_hours, spo2, temperature
    """
    np.random.seed(node_id * 42)
    
    dates = pd.date_range(
        end=datetime.now(), 
        periods=days, 
        freq='D'
    )
    
    data = []
    for patient_id in range(num_patients):
        for date in dates:
            # Simulate realistic health metrics
            base_hr = np.random.randint(60, 80)
            health_status = np.random.choice(['healthy', 'at_risk', 'sick'], p=[0.7, 0.2, 0.1])
            
            if health_status == 'sick':
                heart_rate = base_hr + np.random.randint(20, 40)
                spo2 = np.random.randint(88, 95)
                temp = round(np.random.uniform(37.5, 39.5), 1)
            elif health_status == 'at_risk':
                heart_rate = base_hr + np.random.randint(10, 20)
                spo2 = np.random.randint(94, 97)
                temp = round(np.random.uniform(37.0, 37.8), 1)
            else:
                heart_rate = base_hr + np.random.randint(-5, 10)
                spo2 = np.random.randint(95, 100)
                temp = round(np.random.uniform(36.5, 37.2), 1)
            
            data.append({
                'node_id': f'hospital_{node_id}',
                'patient_id': f'P{node_id}_{patient_id:04d}',
                'date': date,
                'heart_rate': heart_rate,
                'steps': np.random.randint(2000, 12000),
                'sleep_hours': round(np.random.uniform(4, 9), 1),
                'spo2': spo2,
                'body_temp': temp,
                'health_status': health_status
            })
    
    return pd.DataFrame(data)

# Generate data from 5 different nodes (hospitals/cities)
for node in range(1, 6):
    df = generate_health_data(node, num_patients=200, days=90)
    filename = f'data/raw/health/node_{node}_health_data.csv'
    df.to_csv(filename, index=False)
    print(f"  ✓ Generated {filename} - {len(df)} records")

print("  ✓ Health data generation complete!")

# ============================================================================
# 2. AIR QUALITY DATA
# ============================================================================
print("\n[2/3] Generating air quality data for 5 cities...")

def generate_air_quality_data(city_id, days=90):
    """
    Simulate air quality sensor data
    Features: PM2.5, PM10, NO2, CO, O3, AQI
    """
    np.random.seed(city_id * 123)
    
    dates = pd.date_range(
        end=datetime.now(), 
        periods=days * 24,  # Hourly data
        freq='H'
    )
    
    data = []
    for date in dates:
        # Simulate pollution levels (higher during certain hours)
        hour = date.hour
        is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
        
        base_pm25 = np.random.uniform(20, 50)
        if is_rush_hour:
            pm25 = base_pm25 * np.random.uniform(1.5, 2.5)
        else:
            pm25 = base_pm25 * np.random.uniform(0.8, 1.2)
        
        pm10 = pm25 * np.random.uniform(1.3, 1.8)
        no2 = np.random.uniform(10, 80)
        co = np.random.uniform(0.2, 2.5)
        o3 = np.random.uniform(20, 100)
        
        # Calculate simple AQI (Air Quality Index)
        aqi = max(pm25 * 2, pm10, no2, o3)
        
        data.append({
            'city_id': f'city_{city_id}',
            'timestamp': date,
            'pm25': round(pm25, 2),
            'pm10': round(pm10, 2),
            'no2': round(no2, 2),
            'co': round(co, 3),
            'o3': round(o3, 2),
            'aqi': round(aqi, 2)
        })
    
    return pd.DataFrame(data)

cities = ['New_York', 'Los_Angeles', 'Chicago', 'Houston', 'Phoenix']
for i, city in enumerate(cities, 1):
    df = generate_air_quality_data(i, days=90)
    filename = f'data/raw/air_quality/{city}_air_quality.csv'
    df.to_csv(filename, index=False)
    print(f"  ✓ Generated {filename} - {len(df)} records")

print("  ✓ Air quality data generation complete!")

# ============================================================================
# 3. WEATHER DATA
# ============================================================================
print("\n[3/3] Generating weather data for 5 cities...")

def generate_weather_data(city_id, days=90):
    """
    Simulate weather station data
    Features: temperature, humidity, pressure, wind_speed, precipitation
    """
    np.random.seed(city_id * 456)
    
    dates = pd.date_range(
        end=datetime.now(), 
        periods=days * 24,
        freq='H'
    )
    
    data = []
    base_temp = [15, 22, 18, 25, 30][city_id - 1]  # Different base temps per city
    
    for i, date in enumerate(dates):
        # Temperature varies by hour and day
        hour_effect = -5 * np.cos((date.hour - 14) * np.pi / 12)
        day_effect = 3 * np.sin(i * 2 * np.pi / (24 * 30))
        
        temp = base_temp + hour_effect + day_effect + np.random.normal(0, 2)
        humidity = np.clip(70 + np.random.normal(0, 15), 20, 100)
        pressure = np.random.normal(1013, 10)
        wind_speed = np.abs(np.random.normal(15, 8))
        precipitation = max(0, np.random.exponential(2) if np.random.rand() > 0.8 else 0)
        
        data.append({
            'city_id': f'city_{city_id}',
            'timestamp': date,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1),
            'wind_speed': round(wind_speed, 1),
            'precipitation': round(precipitation, 2)
        })
    
    return pd.DataFrame(data)

for i, city in enumerate(cities, 1):
    df = generate_weather_data(i, days=90)
    filename = f'data/raw/weather/{city}_weather.csv'
    df.to_csv(filename, index=False)
    print(f"  ✓ Generated {filename} - {len(df)} records")

print("  ✓ Weather data generation complete!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETE!")
print("=" * 60)
print("\nDataset Summary:")
print("-" * 60)

# Count total records
health_records = sum([len(pd.read_csv(f'data/raw/health/node_{i}_health_data.csv')) for i in range(1, 6)])
air_records = sum([len(pd.read_csv(f'data/raw/air_quality/{city}_air_quality.csv')) for city in cities])
weather_records = sum([len(pd.read_csv(f'data/raw/weather/{city}_weather.csv')) for city in cities])

print(f"Health Data:       {health_records:,} records across 5 hospital nodes")
print(f"Air Quality Data:  {air_records:,} records across 5 cities")
print(f"Weather Data:      {weather_records:,} records across 5 cities")
print(f"\nTotal Records:     {health_records + air_records + weather_records:,}")
print("-" * 60)

print("\n📂 All data saved in 'data/raw/' directory")
print("\nNext Steps:")
print("  1. Run EDA (Exploratory Data Analysis) notebook")
print("  2. Process and merge datasets")
print("  3. Start building the model")