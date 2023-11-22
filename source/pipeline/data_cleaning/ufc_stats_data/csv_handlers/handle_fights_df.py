import pandas as pd
import numpy as np

from .csv_handler import CSVHandler
from .util import standardize_col, time_to_sec

class FightStatsHandler (CSVHandler):
    
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
        
    def _get_opponent_name (self, x):
        fighters = np.unique(x)
        temp = []
        for fighter in x:
            if fighter == fighters[0]:
                temp.append(fighters[1])
            else:
                temp.append(fighters[0])
        return temp
        
    def clean (self):


        self.df.rename(columns={i: standardize_col(i) for i in self.df.columns}, inplace=True)

        """ Clean columns """
        # Remove 'Round' from round col
        self.df['round'] = self.df['round'].map(lambda x: int(x.split(' ')[1])) # 'Round 1' -> 1

        self._separate_ratio_col('sig_str')
        self._separate_ratio_col('total_str')
        self._separate_ratio_col('td')
        self._separate_ratio_col('head')
        self._separate_ratio_col('body')
        self._separate_ratio_col('leg')
        self._separate_ratio_col('distance')
        self._separate_ratio_col('clinch')
        self._separate_ratio_col('ground')

        self.df.sig_str_pct = self.df.sig_str_pct.map(self._remove_percent)
        self.df.td_pct = self.df.td_pct.map(self._remove_percent)
            
        self.df.ctrl = self.df.ctrl.map(time_to_sec)

        """ Aggregate Round Data """
        opponent_mapping = self.df.groupby(['event', 'bout', 'round'])['fighter'].transform(self._get_opponent_name)
        self.df['opponent_name'] = opponent_mapping

        self.df = self.df.merge(self.df, left_on=['event', 'bout', 'round', 'opponent_name'], right_on=['event', 'bout', 'round', 'fighter'], suffixes=('', '_allowed'))

        self.df.drop(columns=['fighter_allowed', 'sig_str_pct', 'td_pct', 'sig_str_pct_allowed', 'td_pct_allowed', 'opponent_name_allowed'], inplace=True)
