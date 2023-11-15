from contextlib import contextmanager

import pandas as pd
from sklearn.model_selection import train_test_split

def _X_feature_selector (dataset: pd.DataFrame):
    decisions = set(['ko_tko', 'unanimous_decision', 'split_decision', 'submission', 'dr_stoppage', 'other'])
    return list(set(dataset.columns) - set(['event', 'date', 'bout', 'outcome', 'fighter_red', 'fighter_blue', 'url_red', 'url_blue']) -
                           decisions - set(map(lambda x: x + '_direct_difference', decisions)))

def _y_feature_selector (): return 'outcome'

@contextmanager
def train_test_sets (dataset: pd.DataFrame, test_frac=0.2, shuffle=True):
    X_features = _X_feature_selector(dataset)
    y_features = _y_feature_selector
    train, test = train_test_split(dataset, shuffle=shuffle, test_size=test_frac)
    yield train[X_features], train[y_features], test[X_features], test[y_features]

@contextmanager
def full_set (dataset: pd.DataFrame, shuffle=True):
    X_features = _X_feature_selector(dataset)
    y_features = _y_feature_selector
    if shuffle:
        temp = dataset.sample(frac=1).reset_index(drop=True)
        yield temp[X_features], temp[y_features]
    else:
        yield dataset[X_features], temp[y_features]
