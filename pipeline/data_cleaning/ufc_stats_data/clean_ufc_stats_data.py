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