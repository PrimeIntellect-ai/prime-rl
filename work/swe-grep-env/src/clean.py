import pandas as pd
from collections import Counter

df = pd.read_parquet("grep_dataset_20k.parquet")

print(f"Total rows: {len(df)}")
print(f"Unique ground truths: {df['ground_truth'].nunique()}")
print(f"Uniqueness ratio: {df['ground_truth'].nunique() / len(df):.1%}")

# Value counts
counts = df['ground_truth'].value_counts()

print(f"\n=== Top 10 most repeated ===")
print(counts.head(10).to_string())

print(f"\n=== Repetition distribution ===")
print(f"Appear 1x (unique):  {(counts == 1).sum()}")
print(f"Appear 2-5x:         {((counts >= 2) & (counts <= 5)).sum()}")
print(f"Appear 6-20x:        {((counts >= 6) & (counts <= 20)).sum()}")
print(f"Appear 21-100x:      {((counts >= 21) & (counts <= 100)).sum()}")
print(f"Appear 100+x:        {(counts > 100).sum()}")

# Ground truth length stats (short ones are often garbage)
df['gt_len'] = df['ground_truth'].str.len()
print(f"\n=== Ground truth length ===")
print(df['gt_len'].describe())

print(f"\n=== Suspiciously short (<3 chars) ===")
short = df[df['gt_len'] < 3]['ground_truth'].value_counts().head(10)
print(short.to_string() if len(short) > 0 else "None")

print(f"\n=== Suspiciously long (>100 chars) ===")
long_gt = df[df['gt_len'] > 100]
print(f"Count: {len(long_gt)}")
if len(long_gt) > 0:
    print(long_gt['ground_truth'].head(3).to_string())

# Cleanup preview
max_repeat = 10  # tune this
to_drop = counts[counts > max_repeat].index
affected = df[df['ground_truth'].isin(to_drop)]
print(f"\n=== Cleanup preview (max {max_repeat} repeats) ===")
print(f"Ground truths exceeding limit: {len(to_drop)}")
print(f"Rows that would be affected: {len(affected)}")
print(f"Rows remaining after dedup: {len(df) - len(affected) + len(to_drop) * max_repeat}")

# Uncomment to actually clean:
# def cap_repeats(df, col, max_n):
#     return df.groupby(col).apply(lambda x: x.head(max_n)).reset_index(drop=True)
# 
# df_clean = cap_repeats(df, 'ground_truth', max_repeat)
# df_clean = df_clean[df_clean['gt_len'] >= 3]  # drop tiny ones
# df_clean = df_clean[df_clean['gt_len'] <= 100]  # drop huge ones
# df_clean.drop(columns=['gt_len']).to_parquet("grep_dataset_clean.parquet")
# print(f"Saved {len(df_clean)} rows to grep_dataset_clean.parquet")
