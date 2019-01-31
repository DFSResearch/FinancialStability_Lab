#############################CONFIGURATIONS TO CONSTRUCT CURVES#################################
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser # ver. < 3.0
### Defining required parameters for constructing sample
#instantiate config
config = ConfigParser()
config.read('../CONFIG.ini')
#MOST IMPORTANT PARAMETERS
main_configs = {key.upper(): value for key, value in config['essential'].items()}

def convert_right_type(value):
    if value in ['True', 'False']:
        return value == 'True'
    try:
        value = float(value)
        return value
    except ValueError:
        print(value)
        return value

main_configs = dict(zip(main_configs.keys(), map(convert_right_type, main_configs.values())))
locals().update(main_configs)

###Parameters which decides whether to fix parameters for curve construction or not
FIX_TAU = True
FIX_ALL_CUTS = True

#Parameters which related to filtering sample dataset
INSTRUMENTS = ['MOM', 'MKM', 'NTK', 'MUM']
MAX_YIELD = None
MIN_YIELD = None
USE_OTC = True
NOTES_IN_OTC = True
MATURITY_FILTER = 1 #in days
DEAL_MARKET = None
USE_N_WIND = True
SPECIFIC_DEALS = None

###Adaptive optimizing parameters
#parameters for Adaptive contructing tenor cuts --- used only FIX_ALL_CUTS = False
N_CUTS = 3
LOOKBACK = 365
TIME_WINDOW = 30 #in days
DETECT_OUTLIERS = False
#parameters for simultaneous loss optimization --- used only FIX_TAU = False
TETA_MAX = 6
TETA_MIN = 0.5
RHO = 0.1
ALPHA = 3

### Renaming column names for more covenience usage
NAME_MASK = {'Coupon Interval': 'annual_freq',
             'Currency': 'currency',
             'Volume': 'volume_bonds',
             'Type': 'type_of_market',
             'Vol KZT': 'volume_kzt',
             'Dirty price': 'deal_price',
             'Coupon rate': 'coupon_rate',
             'Yield': 'ytm',
             'Symbol': 'symbol',
             'Date/Time': 'deal_date',
             'Clean price': 'clean_price',
             'Repayment start': 'end_date',
             'Market sector': 'deal_type'}

### Dictionaries of bonds charecteristics: Face value and Base time
MASK_FACE_VALUE = {'MUM': 1000,
                   'MOM': 1000,
                   'MKM': 100,
                   'NTK': 100}

MASK_BASE_TIME = {'MUM': 360,
                  'MOM': 360,
                  'MKM': 365,
                  'NTK': 365}
