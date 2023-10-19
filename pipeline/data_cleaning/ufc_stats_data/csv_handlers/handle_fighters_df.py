import pandas as pd
import numpy as np
from datetime import datetime
import requests
import re

from .csv_handler import CSVHandler
from .util import standardize_col

class FightersHandler (CSVHandler):

    def __init__ (self, fpath: str, preload: bool = False):
        super().__init__(fpath, preload) # TODO: This is not the real path atm
        self.RECORD_RE = re.compile(r'Record:\s?(\d+)-(\d+)-(\d+)(\s\(NC\s(\d+)\))?')
        self.HEIGHT_RE = re.compile(r'(\d)\' (\d+)"')

    def _get_fighter_records (self, url: str):
            req = requests.get(url)
            record = re.search(self.RECORD_RE, req.text)
            if len(record.groups()) > 4:
                nc = record.group(5)
            else:
                nc = 0
            return {
                'wins': int(record.group(1)),
                'losses': int(record.group(2)),
                'wl_percentage': int(record.group(1)) / (int(record.group(2)) + int(record.group(1))),
                'draws': int(record.group(3)),
                'nc': nc
            }
    
    def _height_to_int (self, height: str) -> int:
        match = re.match(self.HEIGHT_RE, height)
        if match:
            return 12 * int(match.group(1)) + int(match.group(2))
        return np.nan
        
    def _reach_to_int (self, reach: str) -> int:
        if reach == '--':
            return np.nan
        return int(reach.replace('"', ''))

    def _weight_to_int (self, weight: str) -> int:
        if weight == '--':
            return np.nan
        return int(weight.split(' ')[0])

    def _dob_to_age (self, dob: str) -> int:
        if dob == '--':
            return np.nan
        return int((datetime.now() - datetime.strptime(dob, '%b %d, %Y')).days / 365)

    def clean (self):
        self.df.rename(columns={i: standardize_col(i) for i in self.df.columns}, inplace=True)

        self.df.height = self.df.height.map(self._height_to_int)
        self.df.weight = self.df.weight.map(self._weight_to_int)
        self.df.reach = self.df.reach.map(self._reach_to_int)
        self.df.stance = self.df.stance.map(lambda x: x.lower().replace(' ', '_'), na_action='ignore')
        self.df['age'] = self.df.dob.map(self._dob_to_age)
        self.df = pd.get_dummies(self.df, columns=['stance'])

        self.df['record'] = self.df.url.map(self._get_fighter_records)
        self.df[['wins', 'losses', 'wl_percentage', 'draws', 'nc']] = self.df.record.apply(pd.Series)
        self.df.drop(columns=['record'], inplace=True)