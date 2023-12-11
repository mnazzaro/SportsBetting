import os

""" DATA PATHS """

# Root path for all fs objects
DATA_PATH = '/home/markn/Documents/SportsBetting/data'

# Categories of data
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw_data')
CLEAN_DATA_PATH = os.path.join(DATA_PATH, 'clean_data')
TRAINING_DATA_PATH = os.path.join(DATA_PATH, 'training_data')
REPORTS_PATH = os.path.join(DATA_PATH, 'reports')

""" Configurations """

TEST_MODE = False