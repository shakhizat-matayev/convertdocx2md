import pandas as pd

# Path adjusted for your new v1.0 structure
df = pd.read_parquet('output/entities.parquet')

# Filter for anything the AI tagged as an anomaly
anomalies = df[df['type'].str.contains('anomaly', case=False, na=False)]

print(f"Total entities found: {len(df)}")
print(f"Anomalies found: {len(anomalies)}")
print("\n--- List of Extracted Anomalies ---")
print(anomalies[['title', 'description']].to_string())