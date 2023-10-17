from typing import Tuple, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np


def _get_fighter_progression (group):
    x = [float(i) for i in range(len(group))]
    y = group.astype(float)
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m


def make_train_test_sets (fighters_df: pd.DataFrame, fight_stats_df: pd.DataFrame, 
                          fight_results_df: pd.DataFrame, load_train_fpath: Optional[str]=None,
                          load_test_fpath: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    
    fighter_cumulative_df = fight_stats_df.sort_values(by=['fighter', 'date'], ascending=[True, True])
    split_date = fighter_cumulative_df.sort_values(by='date').loc[int(len(fighter_cumulative_df.index) * 0.85)].date
    non_quant_stat_columns = list(set(['ko_tko', 'unanimous_decision', 'split_decision', 'submission', 'dr_stoppage', 'other']).union(
                             set(filter(lambda x: 'weight' in x, fight_stats_df.columns))))
    stat_columns = list(set(fight_stats_df.columns) - set(['event', 'bout', 'fighter', 'date', 'method', 'time_format', 'referee', 'details', 'outcome', 'url'])
                         - set(non_quant_stat_columns))

    failed_fights = []
    def _get_cum_fighter_stats (fighter: str, url: str, date: datetime, suffix: str) -> pd.DataFrame:
        name = pd.Series([fighter, url], ['fighter', 'url'])
        tott = fighters_df[fighters_df['url'] == url][['weight', 'height', 'reach', 'age', 'stance_open_stance', \
                                                               'stance_orthodox', 'stance_sideways', 'stance_southpaw', \
                                                                'stance_switch', 'wins', 'losses', 'draws', 'wl_percentage', 'nc']].squeeze()
        base_stats = fighter_cumulative_df[(fighter_cumulative_df['fighter'] == fighter) & (fighter_cumulative_df['date'] < date)][stat_columns]
        mean = base_stats.mean().add_suffix('_mean')
        trend = base_stats.apply(_get_fighter_progression, axis=0, result_type='reduce', raw=True).add_suffix('_progression')
        non_quant_stats = fighter_cumulative_df[(fighter_cumulative_df['fighter'] == fighter) & (fighter_cumulative_df['date'] == date)][non_quant_stat_columns].squeeze()
        days_temp = fight_stats_df[(fight_stats_df['fighter'] == fighter) & (fight_stats_df['date'] == date)]
        if len(days_temp.index) > 0:
            days_since_last_fight = pd.Series([days_temp.iloc[0]['days_since_last_fight']], ['days_since_last_fight'])
        else:
            days_since_last_fight = pd.Series([np.nan], ['days_since_last_fight'])
        return pd.concat([name, tott, non_quant_stats, days_since_last_fight, mean, trend]).add_suffix(suffix)

    def _make_training_row (group: pd.DataFrame, failures: List[str]):
        try:
            f1 = _get_cum_fighter_stats(group.iloc[0]['fighter'], group.iloc[0]['url'], group.iloc[0]['date'], '_red')
            f2 = _get_cum_fighter_stats(group.iloc[1]['fighter'], group.iloc[1]['url'], group.iloc[1]['date'], '_blue')
            result = fight_results_df[(fight_results_df['event'] == group.name[0]) & (fight_results_df['bout'] == group.name[1])].iloc[0]
            if result['fighter'] == group.iloc[0]['fighter']:
                if result.loc['outcome'] == 'W':
                    outcome = pd.Series([1], index=['outcome'])
                else:
                    outcome = pd.Series([0], index=['outcome'])
            else:
                if result.loc['outcome'] == 'W':
                    outcome = pd.Series([1], index=['outcome'])
                else:
                    outcome = pd.Series([0], index=['outcome'])
            return pd.DataFrame(pd.concat([f1, f2, outcome])).transpose().reset_index(drop=True).set_index(['url_red', 'url_blue'])
        except Exception as e:
            print (f'{group.name}  ERROR: {e}')
            failures.append(group.name)
            return None


    if load_train_fpath:
        train = pd.read_csv(load_train_fpath)
    else:
        train = fighter_cumulative_df[fighter_cumulative_df['date'] < split_date].groupby(['event', 'bout']).filter(lambda x: len(x.index) == 2) \
            .groupby(['event', 'bout']).apply(lambda x: _make_training_row(x, failed_fights)) # .reset_index().drop(columns=['level_2'])
    if load_test_fpath:
        test = pd.read_csv(load_test_fpath)
    else:
        test = fighter_cumulative_df[fighter_cumulative_df['date'] >= split_date].groupby(['event', 'bout']).filter(lambda x: len(x.index) == 2) \
            .groupby(['event', 'bout']).apply(lambda x: _make_training_row(x, failed_fights)) # .reset_index().drop(columns=['level_2'])
    
    return test, train



    # .groupby(['fighter']).filter(lambda x: len(x.index) > 2) \

# fight_stats_df['days_since_last_fight'] = (fight_stats_df.date - fight_stats_df.groupby('fighter')['date'].shift(1)).map(lambda x: float(x.days))


# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

