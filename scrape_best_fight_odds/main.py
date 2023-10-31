from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import json

import pandas as pd

import time
import re

import requests
import base64

df = pd.read_csv('all_training_data.csv')

regex = re.compile(r'UFC \d{2,3}')
mask = df['event'].apply(lambda x: bool(1 if re.match(regex, x) else 0))
main_events = df[mask]['event'].map(lambda x: re.match(regex, x).group(0))
# main_events.iloc[0]
options = webdriver.ChromeOptions()
options.add_argument('user-agent=Chrome/118.0.0.0')
# options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
# driver.get('https://bestfightodds.com')

def enter_search (search: str):
    element = driver.find_element(By.ID, 'search-box1')
    element.send_keys(search)
    element.send_keys(Keys.ENTER)

def get_page_source (event: str) -> pd.DataFrame:
    api_url_base = 'https://www.bestfightodds.com/api/ggd?'
    driver.get('https://bestfightodds.com')
    search_bar = driver.find_element(By.ID, 'search-box1')
    search_bar.send_keys(event)
    search_bar.send_keys(Keys.ENTER)
    fighters = []
    encoded_data = []
    time.sleep(5)
    soup = BeautifulSoup(driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML"), 'html.parser')
    with open('temp.html', 'w') as html:
        html.write(str(soup))
    table = soup.find('table', attrs={'class': 'odds-table'})
    rows = table.find('tbody').find_all('tr')
    for i in rows:
        print (i)
        name = i.find('th').find('a').find('span').text
        print (name)
        fighters.append(name)
        data_li = json.loads(i.find('td', {'class': ['button-cell']})['data-li'])
        req = requests.get(f'{api_url_base}m={data_li[0]}&p={data_li[1]}')
        encoded_data.append(req.text)
        print (encoded_data)
#string to decode (found on the webpage)
# string = """LExRPzI+NlFpUXw2Mj9RW1E1MkUyUWksTFFJUWlgZGdmX2FiZ2FjX19fW1FKUWlhXWVjW1E1MkUyezIzNj1EUWlMUUl=..."""  
# string_decoded = base64.b64decode(string)

# #decoding string, found in the javascript code of the webpage
# m = b'!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
# w=len(m)
# print (bytes(m[(m.index(i)+w//2)%w] for i in string_decoded)) # display results

    
print (get_page_source('UFC 249'))