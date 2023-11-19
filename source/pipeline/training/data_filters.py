import pandas as pd

def remove_wmma (df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.filter(like='women').any(axis=1)]
