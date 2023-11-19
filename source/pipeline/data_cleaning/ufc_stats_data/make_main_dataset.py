from typing import Tuple, List, Optional
from datetime import datetime
from random import random

import pandas as pd
import numpy as np


def _get_fighter_progression (group):
    x = [float(i) for i in range(len(group))]
    y = group.astype(float)
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def get_k_factor (p1_games, p2_games):
    min_games = min(p1_games, p2_games)
    if min_games == 0:
        return 100
    if min_games > 15:
        return 20
    return 100 - ((20*min_games)**0.75)

def update_elo(player1_rating, player2_rating, k_factor, result):

    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    exp_score_p1 = expected_score(player1_rating, player2_rating)
    exp_score_p2 = expected_score(player2_rating, player1_rating)

    new_rating_p1 = player1_rating + k_factor * (result - exp_score_p1)
    new_rating_p2 = player2_rating + k_factor * ((1 - result) - exp_score_p2)

    return new_rating_p1, new_rating_p2

def make_main_dataset (fighters_df: pd.DataFrame, fight_stats_df: pd.DataFrame, 
                          fight_results_df: pd.DataFrame, write_fpath: Optional[str]=None,
                          load_fpath: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if load_fpath:
        return pd.read_csv(load_fpath)
    
    fighter_cumulative_df = fight_stats_df.sort_values(by='date', ascending=True)
    split_date = fighter_cumulative_df.sort_values(by='date').loc[int(len(fighter_cumulative_df.index) * 0.85)].date
    print(split_date)
    non_quant_stat_columns = list(set(['ko_tko', 'unanimous_decision', 'split_decision', 'submission', 'dr_stoppage', 'other']).union(
                             set(filter(lambda x: 'weight' in x, fight_stats_df.columns))))
    stat_columns = list(set(fight_stats_df.columns) - set(['event', 'bout', 'fighter', 'date', 'method', 'time_format', 'referee', 'details', 'outcome', 'url'])
                         - set(non_quant_stat_columns))

    failed_fights = []
    fighter_elo_data = {}
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
        # TODO: Add ufc_fights_red/blue, elo_red/blue. Then, we can calculate current elo in get_prediction_data
        try:
            url_red = group.iloc[0]['url']
            url_blue = group.iloc[1]['url']
            f1 = _get_cum_fighter_stats(group.iloc[0]['fighter'], url_red, group.iloc[0]['date'], '_red')
            f2 = _get_cum_fighter_stats(group.iloc[1]['fighter'], url_blue, group.iloc[1]['date'], '_blue')
            result = fight_results_df[(fight_results_df['event'] == group.name[0]) & (fight_results_df['bout'] == group.name[1])].iloc[0]
            if result['fighter'] == group.iloc[0]['fighter']:
                if result.loc['outcome'] == 'W':
                    outcome = pd.Series([1], index=['outcome'])
                else:
                    outcome = pd.Series([0], index=['outcome'])
            else:
                if result.loc['outcome'] == 'W':
                    outcome = pd.Series([0], index=['outcome'])
                else:
                    outcome = pd.Series([1], index=['outcome'])
            date = pd.Series([group.iloc[0]['date']], ['date'])
            
            
            return pd.DataFrame(pd.concat([f1, f2, date, outcome])).transpose().reset_index(drop=True).set_index(['url_red', 'url_blue'])
        except Exception as e:
            print (f'{group.name}  ERROR: {e}')
            failures.append(group.name)
            return None

    all: pd.DataFrame = fighter_cumulative_df.groupby(['event', 'bout']).filter(lambda x: len(x.index) == 2) \
            .groupby(['event', 'bout']).apply(lambda x: _make_training_row(x, failed_fights)).sort_values(by='date')
    # ones_proportion = len(all[all['outcome'] == 1].index) / len(all.index)
    # size = int((ones_proportion - 0.5) * len(all[all['outcome'] == 1].index))
    # random_indices = np.random.choice(all[all['outcome'] == 1].index, size=size, replace=False)

    fighter_elo_data = {}
    elo_red = []
    games_red = []
    elo_blue = []
    games_blue = []
    for index, row in all.iterrows():
        # try:
            url_red = index[2]
            print (url_red)
            url_blue = index[3]
            if url_red in fighter_elo_data:
                elo1 = fighter_elo_data[url_red][0]
                games1 = fighter_elo_data[url_red][1]
            else:
                elo1 = 1000
                games1 = 0
            if url_blue in fighter_elo_data:
                elo2 = fighter_elo_data[url_blue][0]
                games2 = fighter_elo_data[url_blue][1]
            else:
                elo2 = 1000
                games2 = 0
            elo_red.append(elo1)
            elo_blue.append(elo2)
            games_red.append(games1)
            games_blue.append(games2)

            k = get_k_factor(games1, games2)
            elo1, elo2 = update_elo(elo1, elo2, k, row['outcome'])
            fighter_elo_data[url_red] = (elo1, games1 + 1)
            fighter_elo_data[url_blue] = (elo2, games2 + 1)
        # except Exception as e:
        #     print (e)
        #     elo_red.append(np.nan)
        #     elo_blue.append(np.nan)
        #     games_red.append(np.nan)
        #     games_blue.append(np.nan)


    all['elo_red'] = elo_red
    all['bouts_red'] = games_red
    all['elo_blue'] = elo_blue
    all['bouts_blue'] = games_blue

    reds = list(filter(lambda x: x.endswith('red'), all.columns))
    blues = list(map(lambda x: x[:-4] + '_blue', reds))
    for col in reds:
        if col != 'fighter_red':
            all[col[:-4] + '_direct_difference'] = all[col] - all[col[:-4]+'_blue']
            
    # all.loc[random_indices, reds], all.loc[random_indices, blues] = all.loc[random_indices, blues].values, all.loc[random_indices, reds].values
    # all.loc[random_indices, 'outcome'] = all.loc[random_indices].outcome.map(lambda x: 0 if x == 1 else 1)
    swapped: pd.DataFrame = all.copy()
    print (swapped.head())
    swapped[reds], swapped[blues] = swapped[blues], swapped[reds]
    swapped[list(map(lambda x: x[:-4] + '_direct_difference', filter(lambda x: x != 'fighter_red', reds)))] = swapped[list(map(lambda x: x[:-4] + '_direct_difference', filter(lambda x: x != 'fighter_red', reds)))] * -1
    swapped['outcome'] = swapped.outcome.map(lambda x: 0 if x == 1 else 1)
    swapped.index = swapped.index.swaplevel('url_red', 'url_blue')
    print (all[['fighter_red', 'fighter_blue', 'outcome']].head())
    print (swapped[['fighter_red', 'fighter_blue', 'outcome']].head())
    all = pd.concat([all, swapped]).sort_values(by='date')
   
    if write_fpath:
        all.to_csv(write_fpath)

    return all
