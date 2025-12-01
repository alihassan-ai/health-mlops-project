"""
Check column names in result files
"""
import pandas as pd
import os

print("Checking column names in result files...\n")

# Check baseline results
baseline_file = 'models/evaluation/baseline_results.csv'
if os.path.exists(baseline_file):
    df = pd.read_csv(baseline_file)
    print("BASELINE RESULTS columns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
else:
    print("Baseline results not found")

print("\n" + "="*80 + "\n")

# Check pytorch results
pytorch_file = 'models/evaluation/pytorch_results.csv'
if os.path.exists(pytorch_file):
    df = pd.read_csv(pytorch_file)
    print("PYTORCH RESULTS columns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
else:
    print("PyTorch results not found")