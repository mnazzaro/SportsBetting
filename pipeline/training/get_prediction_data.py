from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

def _get_fighter_progression (group):
    x = [float(i) for i in range(len(group))]
    y = group.astype(float)
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def update_elo(player1_rating, player2_rating, k_factor, result):

    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    exp_score_p1 = expected_score(player1_rating, player2_rating)
    exp_score_p2 = expected_score(player2_rating, player1_rating)

    new_rating_p1 = player1_rating + k_factor * (result - exp_score_p1)
    new_rating_p2 = player2_rating + k_factor * ((1 - result) - exp_score_p2)

    return new_rating_p1, new_rating_p2

def get_k_factor (p1_games, p2_games):
    min_games = min(p1_games, p2_games)
    if min_games == 0:
        return 100
    if min_games > 15:
        return 20
    return 100 - ((20*min_games)**0.75)


def get_cum_fighter_stats (fighters_df: pd.DataFrame, fight_stats_df: pd.DataFrame, fighter: str, url: str, date: datetime, suffix: str) -> pd.Series:
    fighter_cumulative_df = fight_stats_df.sort_values(by='date', ascending=True)
    non_quant_stat_columns = list(set(['ko_tko', 'unanimous_decision', 'split_decision', 'submission', 'dr_stoppage', 'other']).union(
                             set(filter(lambda x: 'weight' in x, fight_stats_df.columns))))
    stat_columns = list(set(fight_stats_df.columns) - set(['event', 'bout', 'fighter', 'date', 'method', 'time_format', 'referee', 'details', 'outcome', 'url'])
                         - set(non_quant_stat_columns))
    name = pd.Series([fighter, url], ['fighter', 'url'])
    
    try:
        tott = fighters_df[fighters_df['url'] == url].iloc[-1][['weight', 'height', 'reach', 'age', 'stance_open_stance', \
                                                                'stance_orthodox', 'stance_sideways', 'stance_southpaw', \
                                                                'stance_switch', 'wins', 'losses', 'draws', 'wl_percentage', 'nc']].squeeze()
    except: 
        print (f"Couldn't find {fighters_df['url'] == url} for {fighter}")
        tott = fighters_df[fighters_df['fighter'] == fighter][['weight', 'height', 'reach', 'age', 'stance_open_stance', \
                                                                'stance_orthodox', 'stance_sideways', 'stance_southpaw', \
                                                                'stance_switch', 'wins', 'losses', 'draws', 'wl_percentage', 'nc']].squeeze()
    base_stats = fighter_cumulative_df[fighter_cumulative_df['fighter'] == fighter][stat_columns]
    mean = base_stats.mean().add_suffix('_mean')
    trend = base_stats.apply(_get_fighter_progression, axis=0, result_type='reduce', raw=True).add_suffix('_progression')
    fighter_stats = fighter_cumulative_df[(fighter_cumulative_df['fighter'] == fighter)]
    non_quant_stats = fighter_stats.loc[fighter_stats['date'] == fighter_stats['date'].max()][non_quant_stat_columns].squeeze() #TODO: this shit ain't 1
    try:
        days_temp = fight_stats_df[(fight_stats_df['fighter'] == fighter)]['date'].max()
        days_since_last_fight = pd.Series([int((date - days_temp).days / 365)], ['days_since_last_fight'])
        print (f"worked for {fighter}")
    except Exception as e:
        print (e)
        print (f"Didn't work for {fighter}")
        days_since_last_fight = pd.Series([np.nan], ['days_since_last_fight'])
    return pd.concat([name, tott, non_quant_stats, days_since_last_fight, mean, trend]).add_suffix(suffix)

def make_matchup_df (fighters_df: pd.DataFrame, fight_stats_df: pd.DataFrame, all_data: pd.DataFrame, group: pd.DataFrame, fight_date: datetime):
    # try:
        url_red = group.iloc[0]['url']
        url_blue = group.iloc[1]['url']
        f1 = get_cum_fighter_stats(fighters_df, fight_stats_df,
            group.iloc[0]['fighter'], url_red, fight_date, '_red')
        f2 = get_cum_fighter_stats(fighters_df, fight_stats_df,
            group.iloc[1]['fighter'], url_blue, fight_date, '_blue')
        date = pd.Series([fight_date], ['date'])

        latest_fight_red = all_data[all_data['fighter_red'] == group.iloc[0]['fighter']]
        latest_fight_red = latest_fight_red.loc[latest_fight_red['date'] == latest_fight_red['date'].max()].iloc[0]
        latest_fight_blue = all_data[all_data['fighter_red'] == group.iloc[1]['fighter']]
        latest_fight_blue = latest_fight_blue.loc[latest_fight_blue['date'] == latest_fight_blue['date'].max()].iloc[0]
        k = get_k_factor(latest_fight_red['bouts_red'], latest_fight_red['bouts_blue'])
        elo1_red, _ = update_elo(latest_fight_red['elo_red'], latest_fight_red['elo_blue'], k, latest_fight_red['outcome'])
        bouts_red = latest_fight_red['bouts_red'] + 1
        elo_red = pd.Series([elo1_red, bouts_red], ['elo_red', 'bouts_red'])

        k = get_k_factor(latest_fight_blue['bouts_red'], latest_fight_blue['bouts_blue'])
        elo1_blue, _ = update_elo(latest_fight_blue['elo_red'], latest_fight_blue['elo_blue'], k, latest_fight_blue['outcome'])
        bouts_blue = latest_fight_blue['bouts_red'] + 1
        elo_blue = pd.Series([elo1_blue, bouts_blue], ['elo_blue', 'bouts_blue'])

        elo_direct_diffs = pd.Series([elo1_red - elo1_blue, bouts_red - bouts_blue], ['elo_direct_difference', 'bouts_direct_difference'])
        # f1 = pd.concat([f1, elo_red])
        # f2 = pd.concat([f2, elo_blue])
        stat_cols = list(set(list(f1.index)) - set(['fighter_red', 'url_red']))
        blue_stat_cols = list(map(lambda x: (x[:-4] + '_blue'), stat_cols))
        print (f'{f1["fighter_red"]}: {len(f1.index)}')
        print (f'{f2["fighter_blue"]}: {len(f2.index)}')
        direct_diffs_index = list(map(lambda x: x[:-4] + '_direct_difference', stat_cols))
        direct_diffs = pd.Series(f1[stat_cols].values - f2[blue_stat_cols].values, direct_diffs_index)
        ret = pd.DataFrame(pd.concat([f1, f2, date, elo_red, elo_blue, direct_diffs, elo_direct_diffs])).transpose().reset_index(drop=True).set_index(['url_red', 'url_blue'])
        return ret
    # except Exception as e:
    #     print (str(e))
        
def make_matchup (fighters_df: pd.DataFrame, fight_stats_df: pd.DataFrame, fighter1: str, fighter2: str, fight_date: datetime):
    group = pd.DataFrame(
        data=list(zip(
            [fighter1, fighter2],
            [fight_stats_df[fight_stats_df['fighter'] == fighter1].iloc[0]['url'], 
             fight_stats_df[fight_stats_df['fighter'] == fighter2].iloc[0]['url']]
        )),
        columns=['fighter', 'url']
    )
    all_data = pd.read_csv('all_training_data.csv')
    return make_matchup_df(fighters_df, fight_stats_df, all_data, group, fight_date)