from typing import Tuple
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

fights_df = pd.read_csv('scrape_ufc_stats/ufc_fight_stats.csv')
fight_results_df = pd.read_csv('scrape_ufc_stats/ufc_fight_results.csv')
fighters_df = pd.read_csv('scrape_ufc_stats/ufc_fighter_tott.csv')
events_df = pd.read_csv('scrape_ufc_stats/ufc_event_details.csv')

def _load_ufc_stats_data (root_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fights_df = pd.read_csv(f'{root_dir}/ufc_fight_stats.csv')
    fight_results_df = pd.read_csv(f'{root_dir}/ufc_fight_results.csv')
    fighters_df = pd.read_csv(f'{root_dir}/ufc_fighter_tott.csv')
    events_df = pd.read_csv(f'{root_dir}/ufc_event_details.csv')
    return fights_df, fight_results_df, fighters_df, events_df

def clean_fights_df (fights_df: pd.DataFrame) -> pd.DataFrame:
    """ Drop null """
    fights_df.dropna(inplace=True)

    """ Rename columns """
    def standardize_col (col_name: str) -> str:
        out = col_name.lower()
        out = out.replace('.', ' ')
        out = out.replace('%', 'pct')
        out = out.strip().replace(' ', '_')
        out = out.replace('__', '_')
        return out

    fights_df.rename(columns={i: standardize_col(i) for i in fights_df.columns}, inplace=True)

    """ Clean columns """
    # Remove 'Round' from round col
    fights_df['round'] = fights_df['round'].map(lambda x: int(x.split(' ')[1])) # 'Round 1' -> 1

    # Separate out columns with 'x of y' in them into two cols (e.g. 'head' and 'head_att' for head attempts)
    def separate_ratio_col (df: pd.DataFrame, col_name: str) -> pd.Series:
        df[f'{col_name}_att'] = df[col_name].map(lambda x: int(x.split(' of ')[1]))
        df[col_name] = df[col_name].map(lambda x: int(x.split(' of ')[0]))

    separate_ratio_col(fights_df, 'sig_str')
    separate_ratio_col(fights_df, 'total_str')
    separate_ratio_col(fights_df, 'td')
    separate_ratio_col(fights_df, 'head')
    separate_ratio_col(fights_df, 'body')
    separate_ratio_col(fights_df, 'leg')
    separate_ratio_col(fights_df, 'distance')
    separate_ratio_col(fights_df, 'clinch')
    separate_ratio_col(fights_df, 'ground')

    # Remove percent signs
    def remove_percent (x):
        if x != '---':
            return float(x.replace('%', ''))
        else:
            return np.nan
    fights_df.sig_str_pct = fights_df.sig_str_pct.map(remove_percent)
    fights_df.td_pct = fights_df.td_pct.map(remove_percent)

    def ctrl_to_sec (x):
        if x != '--':
            parts = x.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        return np.nan
        
    fights_df.ctrl = fights_df.ctrl.map(ctrl_to_sec)

    return fights_df