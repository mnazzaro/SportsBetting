from .csv_handler import CSVHandler
from .util import standardize_col

class FightDetailsHandler (CSVHandler):

    def clean (self):
        self.df.rename(columns={i: standardize_col(i) for i in self.df.columns}, inplace=True)
