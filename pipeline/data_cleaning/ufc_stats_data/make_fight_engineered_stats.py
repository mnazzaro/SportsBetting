
""" Make fight-level engineered features """
from typing import Optional
import pandas as pd

def make_fight_engineered_stats (fight_stats_df: pd.DataFrame, load_fpath: Optional[str]=None):
    if load_fpath:
        return pd.read_csv(load_fpath)
    
    try: # TODO: Why is this popping up still
        fight_stats_df.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass

    round_quant_columns = list(set(fight_stats_df.columns) - \
                            set(
                                ['event', 'bout', 'fighter', 'round', 'date', 
                                'method', 'last_round_time', 'time_format', 'total_time', 'referee', 'outcome', 'ko_tko',
                                'unanimous_decision', 'split_decision', 'submission', 
                                'dr_stoppage', 'other', 'details', 'url']
                                ) - \
                            set(filter(lambda x: x.endswith('_trend'), fight_stats_df.columns)) - \
                            set(filter(lambda x: x.startswith('weightclass_'), fight_stats_df.columns)))

    for col in round_quant_columns:
        if not col.endswith('_allowed'):
            """ Make differential columns """
            fight_stats_df[f'{col}_diff'] = fight_stats_df[col] - fight_stats_df[f'{col}_allowed']

        """ Make per minute columns """
        fight_stats_df[f'{col}_per_min'] = fight_stats_df[col] / fight_stats_df['total_time'] * 60

    """ Make striking accuracy (ratio) columns """
    accuracy_columns = list(set(round_quant_columns) - set(['kd', 'kd_allowed', 'ctrl', 'ctrl_allowed', 'rev', 'rev_allowed']))
    for col in accuracy_columns:
        if (not col.endswith('_att')) and (not col.endswith('_att_allowed')):
            if col.endswith('_allowed'):
                name = col.split('_allowed')[0]
                fight_stats_df[f'{col}_accuracy'] = fight_stats_df[col] / fight_stats_df[f'{name}_att_allowed']
            else:
                fight_stats_df[f'{col}_accuracy'] = fight_stats_df[col] / fight_stats_df[f'{col}_att']

    """ Make days since last fight column """
    fight_stats_df = fight_stats_df.sort_values(by=['fighter', 'date'], ascending=[True, True])
    fight_stats_df.date = pd.to_datetime(fight_stats_df.date)
    fight_stats_df['days_since_last_fight'] = (fight_stats_df.date - fight_stats_df.groupby('fighter')['date'].shift(1)).map(lambda x: float(x.days)) # TODO: This should prob be by url

    return fight_stats_df