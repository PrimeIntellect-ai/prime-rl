import pandas as pd

df = pd.read_parquet("grep_dataset_20k.parquet")
print(f"Starting rows: {len(df)}")

# Remove long ground truths (100+ chars - these are model refusals/errors)
df = df[df['ground_truth'].str.len() <= 100]
print(f"After removing >100 char: {len(df)}")

# Cap repeats at 10
df = (
    df.sample(frac=1, random_state=42)  # shuffle first so we keep random samples
    .groupby('ground_truth', as_index=False)
    .head(10)
)
print(f"After capping repeats at 10: {len(df)}")

# Verify
counts = df['ground_truth'].value_counts()
print(f"\nMax repeats now: {counts.max()}")
print(f"Unique ground truths: {df['ground_truth'].nunique()}")

df.to_parquet("grep_dataset_20k_clean.parquet", index=False)
print(f"\nSaved to grep_dataset_20k_clean.parquet")
