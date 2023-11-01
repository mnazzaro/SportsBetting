from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd

class CSVHandler (ABC):

    def __init__ (self, fpath: str, preload: bool = False):
        if preload:
            self.df = pd.read_csv (fpath)
        else:
            self.df = pd.read_csv (fpath)
        self.preload = preload

    @abstractmethod
    def clean ():
        """ Clean the dataframe and store in self.pd """
        pass

    # @abstractmethod
    # def run_tests ():
    #     """ Run qa tests on dataframe """
    #     pass

    def write (self, fpath: str) -> str:
        """ Write current self.df to fpath and return fpath """
        self.df.to_csv(fpath)
        return fpath
    
    def __call__ (self, write_fpath: Optional[str] = None):
        if self.preload:
            return self.df
        self.clean()
        #self.run_tests()
        if write_fpath:
            self.write(write_fpath)
        return self.df