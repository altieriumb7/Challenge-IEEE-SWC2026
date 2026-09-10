import pandas as pd
import os

data_path = r"C:\Users\Umberto\.cache\kagglehub\datasets\mobeenfatimah\cityflow-smart-urban-mobility-and-traffic-iot\versions\1\smart_city_traffic_mobility.csv"
df = pd.read_csv(data_path)

print("\n--- 1. Spatial Fairness (by City Zone) ---")
zone_stats = df.groupby('city_zone').agg({
    'average_wait_time': 'mean',
    'emission_estimate': 'mean',
    'green_light_duration': 'mean',
    'vehicle_count': 'mean',
    'public_transport_count': 'mean'
}).sort_values('average_wait_time', ascending=False)
print(zone_stats.round(2))

print("\n--- 2. Infrastructure Fairness (Schools/Hospitals) ---")
infra_stats = df.groupby(['nearby_school', 'nearby_hospital']).agg({
    'emission_estimate': 'mean',
    'heavy_vehicle_count': 'mean'
}).round(2)
print(infra_stats)

print("\n--- 3. Modal Fairness (Public Transport vs Private) ---")
print("Correlation Matrix:")
print(df[['green_light_duration', 'vehicle_count', 'public_transport_count', 'heavy_vehicle_count', 'average_wait_time']].corr().round(3))
