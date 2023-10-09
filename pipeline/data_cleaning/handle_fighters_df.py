

fighters_df.rename(columns={i: standardize_col(i) for i in fighters_df.columns}, inplace=True)

HEIGHT_RE = re.compile(r'(\d)\' (\d+)"')
def height_to_int (height: str) -> int:
    match = re.match(HEIGHT_RE, height)
    if match:
        return 12 * int(match.group(1)) + int(match.group(2))
    return np.nan
    
def reach_to_int (reach: str) -> int:
    if reach == '--':
        return np.nan
    return int(reach.replace('"', ''))

def weight_to_int (weight: str) -> int:
    if weight == '--':
        return np.nan
    return int(weight.split(' ')[0])

def dob_to_age (dob: str) -> int:
    if dob == '--':
        return np.nan
    return int((datetime.now() - datetime.strptime(dob, '%b %d, %Y')).days / 365)

fighters_df.height = fighters_df.height.map(height_to_int)
fighters_df.weight = fighters_df.weight.map(weight_to_int)
fighters_df.reach = fighters_df.reach.map(reach_to_int)
fighters_df.stance = fighters_df.stance.map(lambda x: x.lower().replace(' ', '_'), na_action='ignore')
fighters_df['age'] = fighters_df.dob.map(dob_to_age)
fighters_df = pd.get_dummies(fighters_df, columns=['stance'])

import requests
from bs4 import BeautifulSoup

RECORD_RE = re.compile(r'Record:\s?(\d+)-(\d+)-(\d+)(\s\(NC\s(\d+)\))?')

def get_fighter_records (url: str):
    req = requests.get(url)
    record = re.search(RECORD_RE, req.text)
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

def get_fighter_has_records (url: str):
    req = requests.get(url)
    record = re.search(RECORD_RE, req.text)
    return 0 if record is None else 1

# fighters_df['wins'] = outcomes
# fighters_df['has_record'] = fighters_df.url.map(get_fighter_has_records)
fighters_df['record'] = fighters_df.url.map(get_fighter_records)
fighters_df[['wins', 'losses', 'wl_percentage', 'draws', 'nc']] = fighters_df.record.apply(pd.Series)
fighters_df.drop(columns=['record'], inplace=True)

fighters_df.head()