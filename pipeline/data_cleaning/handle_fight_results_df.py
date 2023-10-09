import pandas as pd
import numpy as np

from .csv_handler import CSVHandler
from .util import standardize_col, time_to_sec

class FighterResultsHandler (CSVHandler):

    def __init__ (self):
        super().__init__('ufc_fight_results.csv') # TODO: This is not the real path atm

    def clean (self):
        self.df.rename(columns={i: standardize_col(i) for i in self.df.columns}, inplace=True)

        self.df['fighter_left'] = self.df.bout.map(lambda x: x.split(' vs. ')[0])
        self.df['fighter_right'] = self.df.bout.map(lambda x: x.split(' vs. ')[1])
        self.df['outcome_left'] = self.df.outcome.map(lambda x: x.split('/')[0].strip())
        self.df['outcome_right'] = self.df.outcome.map(lambda x: x.split('/')[1].strip())

        # left = self.df[['fighter_left', 'outcome_left']].copy().rename(columns={'fighter_left': 'fighter', 'outcome_left': 'outcome'})
        # right = self.df[['fighter_right', 'outcome_right']].copy().rename(columns={'fighter_right': 'fighter', 'outcome_right': 'outcome'})

        # fighter_record_df = pd.concat([left, right], axis=0, ignore_index=True)
        left = self.df[['event', 'bout', 'weightclass', 'method', 'round', 'time', 'time_format', 'referee', 'details', 'fighter_left', 'outcome_left']] \
            .rename(columns={'fighter_left': 'fighter', 'outcome_left': 'outcome'})
        right = self.df[['event', 'bout', 'weightclass', 'method', 'round', 'time', 'time_format', 'referee', 'details', 'fighter_right', 'outcome_right']] \
            .rename(columns={'fighter_right': 'fighter', 'outcome_right': 'outcome'})

        self.df = pd.concat([left, right])

        normal_methods = set(['KO/TKO', 'Decision - Unanimous', 'Decision - Split', 'Submission', "TKO - Doctor's Stoppage"])
        self.df.method = self.df.method.map(lambda x: x.strip())
        self.df['ko_tko'] = (self.df.method == 'KO/TKO').astype(int)
        self.df['unanimous_decision'] = (self.df.method == 'Decision - Unanimous').astype(int)
        self.df['split_decision'] = (self.df.method == 'Decision - Split').astype(int)
        self.df['submission'] = (self.df.method == 'Submission').astype(int)
        self.df['dr_stoppage'] = (self.df.method == "TKO - Doctor's Stoppage").astype(int)
        self.df['other'] = self.df.method.map(lambda x: 0 if x in normal_methods else 0)

        self.df.event = self.df.event.apply(lambda x: x.strip())

        self.df.bout = self.df.bout.apply(lambda x: x.strip())
        self.df.bout = self.df.bout.apply(lambda x: x.replace('  ', ' '))

        self.df.fighter = self.df.fighter.apply(lambda x: x.strip())

        self.df = self.df[(self.df['time_format'] == '5 Rnd (5-5-5-5-5)') | (self.df['time_format'] == '3 Rnd (5-5-5)')]
        self.df.time = self.df.time.map(time_to_sec)

        self.df.weightclass = self.df.weightclass.map(lambda x: x.lower().replace(' ', '_'), na_action='ignore')
        self.df = pd.get_dummies(self.df, columns=['weightclass'])

        # TODO: We need to count weightclasses