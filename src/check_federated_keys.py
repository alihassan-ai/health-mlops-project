"""
Quick check of federated data structure
"""
import pickle

# Check first node file
with open('data/federated/city_1_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in federated data:")
for key in data.keys():
    print(f"  - {key}")
    if hasattr(data[key], 'shape'):
        print(f"    Shape: {data[key].shape}")