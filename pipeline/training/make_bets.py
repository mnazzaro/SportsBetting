import pandas as pd
from typing import Tuple, List
from datetime import datetime
from .get_prediction_data import make_matchup
from .test_against_odds import get_implied_odds, make_kelly_bet

def make_bets (fighters_df: pd.DataFrame, fight_stats_df: pd.DataFrame, date: datetime, 
               fighter_tuples: List[Tuple[str, str]], odds_tuples: List[Tuple[int, int]],
               calibrated_model, cols, kelly_threshold: float, kelly_fraction: float,
               bankroll: float):
    """
    Steps:
    1. Make matchups from list of fighter tuples
    2. Augment data
    3. Make predictions and mean
    4. Apply threshold and kelly criterion
    5. Print results
    """
    dfs = []
    dfs_swapped = []
    for f1, f2 in fighter_tuples:
        dfs.append(make_matchup(fighters_df, fight_stats_df, f1, f2, date))
        dfs_swapped.append(make_matchup(fighters_df, fight_stats_df, f2, f1, date))
    
    df = pd.concat(dfs, ignore_index=True)
    df_swapped = pd.concat(dfs_swapped, ignore_index=True)

    df['predictions'] = calibrated_model.predict_proba(df[cols].apply(pd.to_numeric))[:, -1]
    df_swapped['predictions'] = calibrated_model.predict_proba(df_swapped[cols].apply(pd.to_numeric))[:, -1]
    
    ret_odds = []
    for index, row in df.iterrows():
        fighter_red_prob = (row['predictions'] + 1 - df_swapped.loc[index]['predictions']) / 2
        fighter_blue_prob = (1 - row['predictions'] + df_swapped.loc[index]['predictions']) / 2
        ret_odds.append(fighter_red_prob)
        red_implied_odds = get_implied_odds (odds_tuples[index][0])
        blue_implied_odds = get_implied_odds (odds_tuples[index][1])
        prob_r_str = "{:.2f}".format(fighter_red_prob * 100)
        prob_b_str = "{:.2f}".format(fighter_blue_prob * 100)
        imp_odd_r_str = "{:.2f}".format(red_implied_odds * 100)
        imp_odd_b_str = "{:.2f}".format(blue_implied_odds * 100)
        if fighter_red_prob - red_implied_odds >= kelly_threshold:
            print (f"""BET {make_kelly_bet(bankroll, kelly_fraction, red_implied_odds, fighter_red_prob)} on {row['fighter_red']} with {prob_r_str}% chance against {imp_odd_r_str}% implied odds""")
        elif fighter_blue_prob - blue_implied_odds >= kelly_threshold:
            print (f"""BET {make_kelly_bet(bankroll, kelly_fraction, blue_implied_odds, fighter_blue_prob)} on {row['fighter_blue']} with {prob_b_str}% chance against {imp_odd_b_str}% implied odds""")
        else:
            print (f"""DO NOT BET ON {row['fighter_red']} vs. {row['fighter_blue']}\n    -{row['fighter_red']} has {prob_r_str}% chance against {imp_odd_r_str}% implied odds\n    -{row['fighter_blue']} has {prob_b_str}% chance against {imp_odd_b_str}% implied odds""")
            
    return ret_odds
            
    