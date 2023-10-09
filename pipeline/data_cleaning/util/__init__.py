import numpy as np

# Rename columns
def standardize_col (col_name: str) -> str:
    out = col_name.lower()
    out = out.replace('.', ' ')
    out = out.replace('%', 'pct')
    out = out.strip().replace(' ', '_')
    out = out.replace('__', '_')
    return out

def time_to_sec (x):
    if x != '--':
        parts = x.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    return np.nan