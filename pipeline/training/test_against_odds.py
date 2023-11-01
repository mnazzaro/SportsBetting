import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def clean_odds_data (fpath: str):
    odds_df = pd.read_csv(fpath)
    odds_df = odds_df[odds_df['bet_type'] == 'Ordinary']
    odds_df['mean_odds'] = odds_df['BetOnline']
    # odds_df['mean_odds'] = odds_df[['5Dimes', 'BetDSI', 'BookMaker', 'SportBet', 
    #                                 'Bet365', 'Sportsbook', 'William_H', 'Pinnacle', 
    #                                 'SportsInt', 'BetOnline', 'Intertops']].mean(axis=1)
    return odds_df.copy()

def _get_implied_odds (american_odds: int) -> float:
    if american_odds < 0:
        return -(american_odds / (100 - american_odds))
    return 100 / (100 + american_odds)

def make_kelly_bet (bankroll: float, kelly_fraction: float, implied_odds: float, p: float) -> float:
    q = 1 - p
    b = 1 / implied_odds - 1
    kelly_criterion = (b * p - q) / b
    return bankroll * kelly_fraction * kelly_criterion

def calculate_mean_event_return (bankrolls):
    daily_returns = [bankrolls[i+1] / bankrolls[i] - 1 for i in range(len(bankrolls)-1)]
    return np.mean(daily_returns) * 100

def calculate_event_return_stdv (bankrolls):
    daily_returns = [bankrolls[i+1] / bankrolls[i] - 1 for i in range(len(bankrolls)-1)]
    return np.std(daily_returns)

def calculate_sharpe_ratio(bankrolls):
    """
    Calculate the Sharpe Ratio from a list of bankrolls.
    
    Assumes a risk-free rate of 0.
    """
    # Calculate daily returns
    daily_returns = [bankrolls[i+1] / bankrolls[i] - 1 for i in range(len(bankrolls)-1)]

    # Calculate the average daily return
    avg_daily_return = np.mean(daily_returns)

    # Calculate the standard deviation of the daily returns
    std_dev = np.std(daily_returns)

    # Calculate the Sharpe Ratio
    # Assuming a risk-free rate of 0
    if std_dev == 0:  # To handle the case where standard deviation is 0
        return 0
    sharpe_ratio = avg_daily_return / std_dev

    return sharpe_ratio

def compare_predictions_to_odds_groupby_date (test_data: pd.DataFrame, odds_data: pd.DataFrame, 
                                 initial_bankroll: float, kelly_threshold: float,
                                 kelly_fraction: float) -> float:
    bankrolls = [initial_bankroll]
    dates = []
    wins = 0
    losses = 0
    not_placed = 0
    for date, group in test_data.groupby('date'):
        gains = []
        cur_bankroll = bankrolls[-1]
        for _, row in group.iterrows():
            if not (odds_data[(odds_data['fighter1'] == row['fighter_red']) & (odds_data['fighter2'] == row['fighter_blue']) & (odds_data['Card_Date'] == row['date'])].empty):
                odds = odds_data[(odds_data['fighter1'] == row['fighter_red']) & 
                        (odds_data['fighter2'] == row['fighter_blue']) & 
                        (odds_data['Card_Date'] == row['date'])]
                if row['fighter_red'] == odds.iloc[0]['Bet']:
                    if row['prediction'] - _get_implied_odds (odds.iloc[0]['mean_odds']) >= kelly_threshold:
                        bet_size = make_kelly_bet(cur_bankroll, kelly_fraction, _get_implied_odds (odds.iloc[0]['mean_odds']), row['prediction'])
                        if row['outcome'] == 1:
                            print (f"WON ${bet_size} BET ON {row['fighter_red']}")
                            wins += 1
                            gains.append(((1 / _get_implied_odds(odds.iloc[0]['mean_odds'])) - 1) * bet_size)
                        else:
                            print (f"LOST ${bet_size} BET ON {row['fighter_red']}")
                            losses += 1
                            gains.append(-bet_size)
                    else:
                        not_placed += 1
                        print (f"NOT BETTING ON {row['fighter_red']}")
                        if (1 - row['prediction']) - _get_implied_odds (odds.iloc[1]['mean_odds']) >= kelly_threshold:
                            bet_size = make_kelly_bet(cur_bankroll, kelly_fraction, _get_implied_odds (odds.iloc[1]['mean_odds']), 1 - row['prediction'])
                            if row['outcome'] == 0:
                                print (f"WON ${bet_size} BET ON {row['fighter_blue']}")
                                wins += 1
                                gains.append(((1 / _get_implied_odds(odds.iloc[1]['mean_odds'])) - 1) * bet_size)
                            else:
                                print (f"LOST ${bet_size} BET ON {row['fighter_blue']}")
                                losses += 1
                                gains.append(-bet_size)
                        else:
                            not_placed += 1
                            print (f"NOT BETTING ON {row['fighter_blue']}")
                else:
                    if (1 - row['prediction']) - _get_implied_odds (odds.iloc[0]['mean_odds']) >= kelly_threshold:
                        bet_size = make_kelly_bet(cur_bankroll, kelly_fraction, _get_implied_odds (odds.iloc[0]['mean_odds']), 1 - row['prediction'])
                        if row['outcome'] == 0:
                            print (f"WON ${bet_size} BET ON {row['fighter_blue']}")
                            wins += 1
                            gains.append(((1 / _get_implied_odds(odds.iloc[0]['mean_odds'])) - 1) * bet_size)
                        else:
                            print (f"LOST ${bet_size} BET ON {row['fighter_blue']}")
                            losses += 1
                            gains.append(-bet_size)
                    else:
                        print (f"NOT BETTING ON {row['fighter_blue']}")
                        not_placed += 1
                        if row['prediction'] - _get_implied_odds (odds.iloc[1]['mean_odds']) >= kelly_threshold:
                            bet_size = make_kelly_bet(cur_bankroll, kelly_fraction, _get_implied_odds (odds.iloc[1]['mean_odds']), row['prediction'])
                            if row['outcome'] == 1:
                                print (f"WON ${bet_size} BET ON {row['fighter_red']}")
                                wins += 1
                                gains.append(((1 / _get_implied_odds(odds.iloc[1]['mean_odds'])) - 1) * bet_size)
                            else:
                                print (f"LOST ${bet_size} BET ON {row['fighter_red']}")
                                losses += 1
                                gains.append(-bet_size)
                        else:
                            print (f"NOT BETTING ON {row['fighter_red']}")
        bankrolls.append(cur_bankroll + sum(gains))
        dates.append(date)

    dates = list(map(lambda x: datetime.strptime(x, f'%Y-%m-%d'), dates))
    first_date = dates[0] - timedelta(days=1)
    dates.insert(0, first_date)

    for i, d in enumerate(dates):
        if d > datetime(2020, 3, 20):
            stop_index = i
            break

    dates = dates[:stop_index]
    bankrolls = bankrolls[:stop_index]

    print (f"WINS: {wins}")
    print (f"LOSSES: {losses}")
    print (f"NOT PLACED: {not_placed}")
    print (f"W/L RATIO: {wins / losses}")
    print (f"ACCURACY: {wins / (wins + losses)}")
    print (f"BET RATIO: {1 - (not_placed / (wins + losses + not_placed))}")
    print ("FINAL BANKROLL: ${:.2f}".format(bankrolls[-1]))
    print ("MEAN EVENT RETURN: {:.2f}%".format(calculate_mean_event_return(bankrolls)))
    print ("STANDARD DEVIATION: {:.2f}%".format(calculate_event_return_stdv(bankrolls)*100))
    print (f"SHARPE RATIO: {calculate_sharpe_ratio(bankrolls)}")

    plt.plot(dates, bankrolls)
    plt.show()

def compare_predictions_to_odds (test_data: pd.DataFrame, odds_data: pd.DataFrame, 
                                 initial_bankroll: float, kelly_threshold: float,
                                 kelly_fraction: float) -> float:
    bankrolls = [initial_bankroll]
    dates = []
    print (odds_data.head())
    for _, row in test_data.iterrows():
        if not (odds_data[(odds_data['fighter1'] == row['fighter_red']) & (odds_data['fighter2'] == row['fighter_blue']) & (odds_data['Card_Date'] == row['date'])].empty):
            odds = odds_data[(odds_data['fighter1'] == row['fighter_red']) & 
                     (odds_data['fighter2'] == row['fighter_blue']) & 
                     (odds_data['Card_Date'] == row['date'])]
            if row['fighter_red'] == odds.iloc[0]['Bet']:
                if row['prediction'] - _get_implied_odds (odds.iloc[0]['mean_odds']) >= kelly_threshold:
                    bet_size = make_kelly_bet(bankrolls[-1], kelly_fraction, _get_implied_odds (odds.iloc[0]['mean_odds']), row['prediction'])
                    if row['outcome'] == 1:
                        print (f"WON ${bet_size} BET ON {row['fighter_red']}")
                        bankrolls.append(bankrolls[-1] + (((1 / _get_implied_odds(odds.iloc[0]['mean_odds'])) - 1) * bet_size))
                    else:
                        print (f"LOST ${bet_size} BET ON {row['fighter_red']}")
                        bankrolls.append(bankrolls[-1] - bet_size)
                    dates.append(row['date'])
                else:
                    print (f"NOT BETTING ON {row['fighter_red']}")
                    if (1 - row['prediction']) - _get_implied_odds (odds.iloc[1]['mean_odds']) >= kelly_threshold:
                        bet_size = make_kelly_bet(bankrolls[-1], kelly_fraction, _get_implied_odds (odds.iloc[1]['mean_odds']), 1 - row['prediction'])
                        if row['outcome'] == 0:
                            print (f"WON ${bet_size} BET ON {row['fighter_blue']}")
                            bankrolls.append(bankrolls[-1] + (((1 / _get_implied_odds(odds.iloc[1]['mean_odds'])) - 1) * bet_size))
                        else:
                            print (f"LOST ${bet_size} BET ON {row['fighter_blue']}")
                            bankrolls.append(bankrolls[-1] - bet_size)
                        dates.append(row['date'])
                    else:
                        print (f"NOT BETTING ON {row['fighter_blue']}")
            else:
                if (1 - row['prediction']) - _get_implied_odds (odds.iloc[0]['mean_odds']) >= kelly_threshold:
                    bet_size = make_kelly_bet(bankrolls[-1], kelly_fraction, _get_implied_odds (odds.iloc[0]['mean_odds']), 1 - row['prediction'])
                    if row['outcome'] == 0:
                        print (f"WON ${bet_size} BET ON {row['fighter_blue']}")
                        bankrolls.append(bankrolls[-1] + (((1 / _get_implied_odds(odds.iloc[0]['mean_odds'])) - 1) * bet_size))
                    else:
                        print (f"LOST ${bet_size} BET ON {row['fighter_blue']}")
                        bankrolls.append(bankrolls[-1] - bet_size)
                    dates.append(row['date'])
                else:
                    print (f"NOT BETTING ON {row['fighter_blue']}")
                    if row['prediction'] - _get_implied_odds (odds.iloc[1]['mean_odds']) >= kelly_threshold:
                        bet_size = make_kelly_bet(bankrolls[-1], kelly_fraction, _get_implied_odds (odds.iloc[1]['mean_odds']), row['prediction'])
                        if row['outcome'] == 1:
                            print (f"WON ${bet_size} BET ON {row['fighter_red']}")
                            bankrolls.append(bankrolls[-1] + (((1 / _get_implied_odds(odds.iloc[1]['mean_odds'])) - 1) * bet_size))
                        else:
                            print (f"LOST ${bet_size} BET ON {row['fighter_red']}")
                            bankrolls.append(bankrolls[-1] - bet_size)
                        dates.append(row['date'])
                    else:
                        print (f"NOT BETTING ON {row['fighter_red']}")

    dates = list(map(lambda x: datetime.strptime(x, f'%Y-%m-%d'), dates))
    first_date = dates[0] - timedelta(days=1)
    dates.insert(0, first_date)

    for i, d in enumerate(dates):
        if d > datetime(2020, 6, 1):
            stop_index = i
            break

    dates = dates[:stop_index]
    bankrolls = bankrolls[:stop_index]

    print (f"FINAL BANKROLL: {bankrolls[-1]}")
    print (f"MEAN EVENT RETURN: {calculate_mean_event_return(bankrolls)}")
    print (f"ROI: {(bankrolls[-1] / initial_bankroll) / bankrolls[-1]}")
    print (f"SHARPE RATIO: {calculate_sharpe_ratio(bankrolls)}")

    


    plt.plot(dates, bankrolls)
    plt.show()


