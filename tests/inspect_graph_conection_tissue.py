import pandas as pd
# Check the relationships table
df_rel = pd.read_parquet('output/relationships.parquet')

# Look for connections to Spruce-W07
well_links = df_rel[df_rel['source'].str.contains('Spruce-W07', case=False)]

print(f"Total relationships found: {len(df_rel)}")
print("\n--- Spruce-W07 Connections ---")
print(well_links[['source', 'target', 'description']].to_string())