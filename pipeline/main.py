from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pytz
import matplotlib.pyplot as plt

import config

from data_cleaning.ufc_stats_data.csv_handlers.handle_fight_results_df import FightResultsHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_fighters_df import FightersHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_fights_df import FightStatsHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_events_df import EventsHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_fight_details_df import FightDetailsHandler
from data_cleaning.ufc_stats_data.make_fighter_cumulative_df import make_fighter_cumulative_df
from data_cleaning.ufc_stats_data.make_fight_engineered_stats import make_fight_engineered_stats
from pipeline.data_cleaning.ufc_stats_data.make_main_dataset import make_main_dataset

from training.xgb import train_xgb, train_xgb_all
from training.get_prediction_data import make_matchup
from training.data_filters import remove_wmma
from training.make_bets import make_bets
from training.data_manager import full_set, train_test_sets


if __name__=='__main__':
    print(f"Starting FightResultsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    fight_results_df = FightResultsHandler(f'{config.RAW_DATA_PATH}/ufc_fight_results.csv')()

    print(f"Starting FightersHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    # fighters_df = FightersHandler(f'{config.RAW_DATA_PATH}/ufc_fighter_tott.csv')(f'{config.CLEAN_DATA_PATH}/fighters_df.csv')
    fighters_df = FightersHandler(f'{config.CLEAN_DATA_PATH}/fighters_df.csv', preload=True)()

    print(f"Starting FightStatsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    fight_stats_df = FightStatsHandler(f'{config.RAW_DATA_PATH}/ufc_fight_stats.csv')(f'{config.CLEAN_DATA_PATH}/fight_stats_df.csv')

    print(f"Starting EventsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    events_df = EventsHandler(f'{config.RAW_DATA_PATH}/ufc_event_details.csv')()

    print(f"Starting FightDetailsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    fight_details_df = FightDetailsHandler(f'{config.RAW_DATA_PATH}/ufc_fight_details.csv')()

    print(f"Starting make cumulative df at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    cumulative_df = make_fighter_cumulative_df(fight_stats_df, fight_results_df, events_df, fight_details_df, 
                                               write_fpath=f'{config.CLEAN_DATA_PATH}/fight_stats_with_url.csv',
                                               load_fpath=f'{config.CLEAN_DATA_PATH}/fight_stats_with_url.csv')

    print (cumulative_df.head())

    all_fight_level_stats = make_fight_engineered_stats(cumulative_df, write_fpath=f'{config.CLEAN_DATA_PATH}/all_fight_level_stats.csv')
    
    all_data = make_main_dataset (fighters_df, all_fight_level_stats, fight_results_df, 
                                  write_fpath=f'{config.TRAINING_DATA_PATH}/dataset.csv',
                                  load_fpath=f'{config.TRAINING_DATA_PATH}/dataset.csv')

    # train, test = train_test_split(pd.read_csv('all_training_data.csv'), shuffle=False, test_size=0.25)
    # train, test = remove_wmma(train), remove_wmma(test)

    # all = pd.read_csv('all_training_data.csv')
    # cols, model = train_xgb(train, test)

    all = pd.read_csv(f'{config.TRAINING_DATA_PATH}/all_training_data.csv')
    cols, model = train_xgb_all(all)

    with full_set(all_data) as (X, y):
        train_xgb_all(X, y)

    print ('testing time')

    # odds = []
    # for i in range(40):
    #     cols, model = train_xgb_all(all)

    #     # for j in range(10):
    #     #     print ('\n')
    #     #     accuracies = []
    #     #     for i in range(50, 100, 5):
    #     #         train, test = train_test_split(all, shuffle=False, test_size=(1-(i/100)))
    #     #         cols, model, accuracy = train_xgb(train, test)
    #     #         print (f'ACCURACY WHEN TRAINING ON {i}% of the data: {accuracy}')
    #     #         accuracies.append(accuracy)
    #     #     plt.plot(range(50, 100, 5), accuracies)
    #     # plt.show()

    #     odds.append(make_bets(fighters_df, engineered_fight_level_stats, datetime(2023, 11, 4),
    #               [
    #                   ('Derrick Lewis', 'Jailton Almeida'),
    #                   ('Rodrigo Nascimento', "Don'Tale Mayes"),
    #                   ('Armen Petrosyan', 'Rodolfo Vieira'),
    #               ],
    #               [
    #                   (360, -500), 
    #                   (-200, 165),
    #                   (-110, -110),

    #               ], model, cols, 0.03, 0.25, 1000))
        
    # odds = np.array(odds)
    # mean_odds = []
    # for i in range(odds.shape[1]):
    #     mean_odds.append(np.mean(odds[:, i]))

    # print (mean_odds)
    make_bets(fighters_df, all_fight_level_stats, datetime(2023, 11, 11),
              [
                  ('Jiri Prochazka', 'Alex Pereira'),
                  ('Tom Aspinall', "Sergei Pavlovich"),
                  ('Jessica Andrade', 'Mackenzie Dern'),
                  ('Matt Frevola', 'Benoit Saint Denis'),
                  ('Diego Lopes', 'Pat Sabatini'),
                  ('Tabatha Ricci', 'Loopy Godinez'),
                  ('Jared Gordon', 'Mark Madsen'),
                  ('John Castaneda', 'Kyung Ho Kang'),
                  ('Dennis Buzukja', 'Jamall Emmers')
              ],
              [
                  (105, -125), 
                  (-115, -105),
                  (170, -210),
                  (185, -225),
                  (100, -120),
                  (145, -175),
                  (-225, 180),
                  (-135, 110),
                  (200, -250)

              ], model, cols, 0.03, 0.15, 1007)

    print("FINISHED")