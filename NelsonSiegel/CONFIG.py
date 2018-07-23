#############################CONFIGURATIONS TO CONSTRUCT CURVES#################################

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

### Defining required parameters for constructing sample
SETTLE_DATE = '2018-06-04'
INSTRUMENTS = ['MOM', 'MKM', 'NTK', 'MUM']
MAX_YIELD = 0.20
MIN_YIELD = 0.03
N_CUTS=3
USE_OTC = True
NOTES_IN_OTC = True
MATURITY_FILTER = None
DEAL_MARKET = None
USE_N_WIND = True
RHO = 0.5
ALPHA = 3
TIME_WINDOW = 30 #in days
MIN_N_DEAL = 10 
WEIGHT_SCHEME = 'no_weight'
EXPORT_TO_EXCEL = True
FILE_NAME = 'estimated_rates'
SPECIFIC_DEALS = None
#parameters for loss optimization
TETA_MAX = 6
TETA_MIN = 0.5