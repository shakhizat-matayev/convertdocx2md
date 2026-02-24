import pandas as pd
df_rel = pd.read_parquet('output/relationships.parquet')

# Find entities that are connected to MORE THAN ONE well
shared = df_rel.groupby('target')['source'].nunique()
shared_entities = shared[shared > 1]

print("--- Entities Shared Between Wells ---")
print(shared_entities)