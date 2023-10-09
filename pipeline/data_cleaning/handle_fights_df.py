import pandas as pd
import numpy as np

from .csv_handler import CSVHandler
from .util import standardize_col, time_to_sec

class FightStatsHandler (CSVHandler):

    def __init__ (self):
        super().__init__('ufc_fight_stats.csv') # TODO: This is not the real path atm
    
    # Separate out columns with 'x of y' in them into two cols (e.g. 'head' and 'head_att' for head attempts)
    def _separate_ratio_col (self, col_name: str) -> pd.Series:
        self.df[f'{col_name}_att'] = self.df[col_name].map(lambda x: int(x.split(' of ')[1]))
        self.df[col_name] = self.df[col_name].map(lambda x: int(x.split(' of ')[0]))

    # Remove percent signs
    def _remove_percent (self, x):
        if x != '---':
            return float(x.replace('%', ''))
        else:
            return np.nan
        
    def clean (self):

        """ Drop null """
        self.df.dropna(inplace=True)

        self.df.rename(columns={i: standardize_col(i) for i in self.df.columns}, inplace=True)

        """ Clean columns """
        # Remove 'Round' from round col
        self.df['round'] = self.df['round'].map(lambda x: int(x.split(' ')[1])) # 'Round 1' -> 1

        self._separate_ratio_col(self.df, 'sig_str')
        self._separate_ratio_col(self.df, 'total_str')
        self._separate_ratio_col(self.df, 'td')
        self._separate_ratio_col(self.df, 'head')
        self._separate_ratio_col(self.df, 'body')
        self._separate_ratio_col(self.df, 'leg')
        self._separate_ratio_col(self.df, 'distance')
        self._separate_ratio_col(self.df, 'clinch')
        self._separate_ratio_col(self.df, 'ground')

        
        self.df.sig_str_pct = self.df.sig_str_pct.map(self._remove_percent)
        self.df.td_pct = self.df.td_pct.map(self._remove_percent)
            
        self.df.ctrl = self.df.ctrl.map(time_to_sec)