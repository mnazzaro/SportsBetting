from typing import Optional

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

""" Compile round-level stats in fights_df into fight-level stats """

def _get_bout_fighter_output_trend (group, columns):
    res = pd.DataFrame()
    # res.set_index(group.index)
    for col in columns:
        x = group['round'].values.astype(float)
        y = group[col].values.astype(float)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        res[f'{col}_trend'] = [m]
    return res

def _find_fighter_in_keys (name: str, urls: dict) -> str:
    """ Looser check for name in url keys (e.g. Michelle Waterson matches Michelle Waterson-Gomez)"""
    try:
        return urls['name']
    except:
        for key, val in urls.items():
            if name in key:
                return val
        
def _make_urls (group: pd.DataFrame, fight_details):
    print (f'EVENT: {group.iloc[0]["event"]}, BOUT: {group.iloc[0]["bout"]}')
    try:
        url = fight_details[(fight_details['event'] == group.iloc[0]['event']) & (fight_details['bout'] == group.iloc[0]['bout'])].iloc[0]['url']
    except:
        reverse_bout = ' vs. '.join(group.iloc[0]['bout'].split(' vs. ')[::-1])
        url = fight_details[(fight_details['event'] == group.iloc[0]['event']) & \
                      (fight_details['bout'] == reverse_bout)].iloc[0]['url']
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')
    urls = {x.text.strip(): x['href'] for x in soup.find_all('a', attrs={'class': 'b-link b-fight-details__person-link'})}
    group.insert(len(group.columns), 'url', 
                        group['fighter'].map(lambda x: _find_fighter_in_keys(x, urls)))
    return group

def make_fighter_cumulative_df (fights_df: pd.DataFrame, fight_results_df: pd.DataFrame, 
                                events_df: pd.DataFrame, fight_details_df: pd.DataFrame,
                                write_fpath: Optional[str] = None,
                                load_fpath: Optional[str] = None):
    if load_fpath:
        return pd.read_csv(load_fpath)

    sum_columns = list(set(fights_df.columns) - set(['round', 'opponent_name', 'event', 'bout', 'fighter']))
    fight_stats_df = fights_df.groupby(['event', 'bout', 'fighter'])[sum_columns].sum().reset_index()
    trends = fights_df.groupby(['event', 'bout', 'fighter'])[[*sum_columns, 'round']].apply(lambda x: _get_bout_fighter_output_trend(x, sum_columns))

    fight_stats_df = fight_stats_df.merge(events_df[['event', 'date']], how='left', on='event')
    fight_stats_df = fight_stats_df.merge(fight_results_df, how='left', on=['event', 'bout', 'fighter'])
    fight_stats_df = fight_stats_df.merge(trends, how='left', on=['event', 'bout', 'fighter'])

    fight_stats_df.rename(columns={'time': 'last_round_time'}, inplace=True)
            
    fight_stats_df['total_time'] = fight_stats_df.apply(lambda x: (x['round'] - 1) * 300 + x['last_round_time'], axis=1)

    # fight_stats_df.to_csv('fig_cum.csv')
    # print (fight_stats_df.head())

    # fight_stats_df = pd.read_csv('fig_cum.csv') TODO: Remove this but have a feeling (since I haven't tested since removing) we will need to do a to_numeric or something. Idk why else it would be here

    print ('STARTING URLS IN MAKING FIG CUM')

    fight_stats_df = fight_stats_df.groupby(['event', 'bout']).apply(lambda x: _make_urls(x, fight_details_df)).reset_index(drop=True) # this reset index should remove the unnamed: 0 but it's untested

    if write_fpath:
        fight_stats_df.to_csv(write_fpath)

    return fight_stats_df

