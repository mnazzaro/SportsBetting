import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def clean_odds_data (fpath: str):
    odds_df = pd.read_csv(fpath)
    odds_df = odds_df[odds_df['bet_type'] == 'Ordinary']
    odds_df['mean_odds'] = odds_df['Bet365']
    # odds_df[['5Dimes', 'BetDSI', 'BookMaker', 'SportBet', 
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

def calculate_sharpe_ratio(bankrolls):
    """
    Calculate the Sharpe Ratio from a list of bankrolls.
    
    Assumes a risk-free rate of 0.
    """
    import numpy as np

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

    print (f"FINAL BANKROLL: {bankrolls[-1]}")
    print (f"SHARPE RATIO: {calculate_sharpe_ratio(bankrolls)}")

    dates = list(map(lambda x: datetime.strptime(x, f'%Y-%m-%d'), dates))
    first_date = dates[0] - timedelta(days=1)
    dates.insert(0, first_date)
    plt.plot(dates, bankrolls)
    plt.show()


