from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import pandas as pd

import time
import re

df = pd.read_csv('all_training_data.csv')


regex = re.compile(r'UFC \d{2,3}')
mask = df['event'].apply(lambda x: bool(1 if re.match(regex, x) else 0))
main_events = df[mask]['event'].map(lambda x: re.match(regex, x).group(0))
# main_events.iloc[0]
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
driver.get('https://bestfightodds.com')

def enter_search (search: str):
    element = driver.find_element(By.ID, 'search-box1')
    element.send_keys(search)
    element.send_keys(Keys.ENTER)

enter_search(main_events.iloc[2000])
time.sleep(10)