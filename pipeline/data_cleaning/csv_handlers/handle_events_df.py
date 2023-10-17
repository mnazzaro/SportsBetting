from datetime import datetime

from .csv_handler import CSVHandler
from .util import standardize_col

class EventsHandler (CSVHandler):

    def clean (self):
        self.df.rename(columns={i: standardize_col(i) for i in self.df.columns}, inplace=True)
        self.df.date = self.df.date.map(lambda x: datetime.strptime(x, '%B %d, %Y'))