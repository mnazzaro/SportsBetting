import pandas as pd
from abc import ABC, abstractmethod

class CSVHandler (ABC):

    def __init__ (self, fpath: str):
        self.df = pd.read_csv (fpath).dropna()

    @abstractmethod
    def clean ():
        """ Clean the dataframe and store in self.pd """
        pass

    @abstractmethod
    def run_tests ():
        """ Run qa tests on dataframe """
        pass

    def write (self, fpath: str) -> str:
        """ Write current self.df to fpath and return fpath """
        self.df.to_csv(fpath)
        return fpath