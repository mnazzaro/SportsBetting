from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import pytz

from data_cleaning.ufc_stats_data.csv_handlers.handle_fight_results_df import FightResultsHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_fighters_df import FightersHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_fights_df import FightStatsHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_events_df import EventsHandler
from data_cleaning.ufc_stats_data.csv_handlers.handle_fight_details_df import FightDetailsHandler
from data_cleaning.ufc_stats_data.make_fighter_cumulative_df import make_fighter_cumulative_df
from data_cleaning.ufc_stats_data.make_fight_engineered_stats import make_fight_engineered_stats
from data_cleaning.ufc_stats_data.make_train_test_sets import make_train_test_sets

from training.xgb import train_xgb
from training.data_filters import remove_wmma


if __name__=='__main__':
    print(f"Starting FightResultsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    fight_results_df = FightResultsHandler('../scrape_ufc_stats/ufc_fight_results.csv')()

    print(f"Starting FightersHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    #fighters_df = FightersHandler('../scrape_ufc_stats/ufc_fighter_tott.csv')('fighters_df')
    fighters_df = FightersHandler('fighters_df.csv', preload=True)()

    print(f"Starting FightStatsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    fight_stats_df = FightStatsHandler('../scrape_ufc_stats/ufc_fight_stats.csv')()

    print(f"Starting EventsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    events_df = EventsHandler('../scrape_ufc_stats/ufc_event_details.csv')()

    print(f"Starting FightDetailsHandler at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    fight_details_df = FightDetailsHandler('../scrape_ufc_stats/ufc_fight_details.csv')()

    print(f"Starting make cumulative df at {datetime.now(tz=pytz.timezone('US/Eastern'))}")
    cumulative_df = make_fighter_cumulative_df(fight_stats_df, fight_results_df, events_df, fight_details_df, load_fpath='fight_stats_with_url.csv')

    print (cumulative_df.head())

    engineered_fight_level_stats = make_fight_engineered_stats(cumulative_df)
    
    # engineered_fight_level_stats.to_csv('engineered_fight_level_stats.csv') # TODO: these types of big dataframe editing functions should write themselves to file

    # all_data = make_train_test_sets(fighters_df, engineered_fight_level_stats, fight_results_df)
                                       # load_train_fpath='train.csv', load_test_fpath='test.csv')

    train, test = train_test_split(pd.read_csv('all_training_data.csv').drop(columns=['date']), shuffle=False)
    print (train.head())
    # train, test = remove_wmma(train), remove_wmma(test)
    train_xgb(train, test)

    print("FINISHED")