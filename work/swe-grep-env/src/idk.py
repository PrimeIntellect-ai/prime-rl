import pandas as pd

df = pd.read_parquet("grep_dataset_20k_v2.parquet")

# Check a few rows in detail
for i in range(5):
    print(f"\n{'='*60}")
    print(f"Row {i}")
    print(f"Query: {df.iloc[i]['user_query']}")
    print(f"Answer: {df.iloc[i]['answer']}")
    print(f"Files: {df.iloc[i]['gt_files']}")
    print(f"Lines: {df.iloc[i]['gt_lines']}")

# Check for empty/malformed entries
print(f"\n{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Empty answers: {(df['answer'] == '').sum()}")
print(f"Empty files: {df['gt_files'].apply(lambda x: len(x) == 0).sum()}")
print(f"Empty lines: {df['gt_lines'].apply(lambda x: len(x) == 0).sum()}")
