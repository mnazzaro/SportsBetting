from typing import Optional
from enum import Enum

class OddsConversion (Enum):
    AMERICAN2IMPLIED = lambda x:  -(x / (100 - x)) if x < 0 else 100 / (100 + x)
    DECIMAL2IMPLIED = lambda x: 1 / x
    AMERICAN2DECIMAL = lambda x: (100 - x) / -x if x < 0 else (100 + x) / x

def convert_odds (value: float, conversion_type: OddsConversion):
    return conversion_type(value)

def calculate_vig (odds1: float, odds2: float, conversion: Optional[OddsConversion] = None):
    if conversion:
        return conversion(odds1) + conversion(odds2) - 1
    return odds1 + odds2 - 1

if __name__=='__main__':
    # print(convert_odds(100, OddsConversion.AMERICAN2DECIMAL))
    print (calculate_vig(130, -150, OddsConversion.AMERICAN2IMPLIED))