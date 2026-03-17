import pandas as pd

# The TSV has two schemas:
#   - Old rows: 30 fields (no distance_metric/feature_set_name/n_features)
#   - New rows: 33 fields (3 extra metadata columns at the end)
# Read with 33 columns so all rows load correctly.
# Old rows get NaN for the last 3 columns — that's fine.

# First, peek at the header to get the base column names
with open('results/results_analogue.tsv') as f:
    header = f.readline().strip().split('\t')

# Add the 3 extra column names for the longer rows
extra_cols = ['distance_metric', 'feature_set_name', 'n_features']
all_cols = header + [c for c in extra_cols if c not in header]

df = pd.read_csv(
    'results/results_analogue.tsv',
    sep='\t',
    names=all_cols,
    skiprows=1,          # skip original header
    engine='python',
    on_bad_lines='skip', # skip any truly malformed rows
)

before = len(df)
df_clean = df.drop_duplicates(subset=['experiment_name'], keep='first')
df_clean.to_csv('results/results_analogue.tsv', sep='\t', index=False)
print(f'Read {before} rows. Removed {before - len(df_clean)} duplicates. Kept {len(df_clean)}.')
