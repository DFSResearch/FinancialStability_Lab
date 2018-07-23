import os, sys
import pandas as pd
from .bond import coupon_rate_modifier

#### creating new essential features based on input data
def creating_new_columns(df, mask_face_value, mask_base_time):
    #Extracted symbol type of bond
    df['bond_symb'] = df.symbol.str.extract(r'([A-Z]+)', expand=False)
    #defining Face Value of bond
    df['face_value'] = df['bond_symb'].map(mask_face_value)
    #defining Base Time used for calculatting time structure of payments
    df['base_time'] = df['bond_symb'].map(mask_base_time)
    # correct time span of bond until matiruty in days
    # different tenor for different base times
    simple_tenor = (df['end_date'] - df['deal_date']).dt.days
    month_days = (df['end_date'].dt.month - df['deal_date'].dt.month) * 30 
    year_days = (df['end_date'].dt.year - df['deal_date'].dt.year) * 360 
    difference_days = (df['end_date'].dt.day - df['deal_date'].dt.day)

    df.loc[df['base_time'] == 360, 'span'] = year_days + month_days + difference_days
    df.loc[df['base_time'] == 365, 'span'] = simple_tenor
    #definig beg date for bonds
    beg_date = df.groupby('symbol').deal_date.min().rename('beg_date')
    df = pd.merge(df, pd.DataFrame(beg_date), left_on='symbol', right_index=True)
    return df

#### Cleaning data from non systematic errors
def cleaning_from_errors(df):
    #dropping out non-KZT deals or deals wo end date
    df = df[(df['currency'] == 'KZT') & ~(df['end_date'].isnull())]
    print('KZT and end date is not null', df.shape)
    #end date > beg date
    df = df[df['end_date'] > df['deal_date']]
    print('end date > beg date', df.shape)
    df = df[~((df['clean_price'] < 0.5) & (df['deal_price'] > 300))]
    print('strange deals', df.shape)
    return df

#### Defining filling function which will replace wrong values 
def filling_values(df, discount_bonds=['NTK', 'MKM']):
    #recoding type of deal in integers
    df['deal_type'] = df.deal_type.map({'1': 1, '2': 2, 'SR': 3})
    print('beggining of filling data', df.shape)
    #if clean price equal coupon rate than it is errors in dataset
    #so clean price is taken as 100 in this case
    df.loc[df['clean_price'] == df['coupon_rate'], 'clean_price'] = 100
    ##filling coupon rate and annual frequency for discount bonds
    df.loc[df.bond_symb.isin(discount_bonds), 'annual_freq'] = 1
    df.loc[df['deal_price'].isnull(), 'deal_price'] = (df.volume_kzt / df.volume_bonds)
    #cleaninig out too high or low values
    df.loc[(df['clean_price'] < 0.5) & (df['deal_price'] < 300), 'clean_price'] = df.deal_price
    #when yield is zero, yield in reality is coupon rate
    df.loc[(df.ytm == 0) | (df.ytm.isnull()), 'ytm'] = df['coupon_rate'] 
    #filling nan ytm
    df['ytm'] = df['ytm'].fillna(9)
    df = df[df.deal_price > 1]
    #throwing out deals with too small deal price relative to face value
    df = df[df['face_value'] / df['deal_price'] < 2]
    #standartize price
    df['stand_price'] = (df.deal_price / df.face_value) * 100
    print('end of filling data', df.shape)
    df = df.dropna(subset=['coupon_rate', 'deal_price'])
    print('cleaning from nan coupon rate and deal price', df.shape)
    return df 

#### Filtering data by type of transactions
def filtering(df, needed_bonds=None, use_otc=False, notes_in_otc=False, deal_market=None,
              maturity_filter=None, specific_deals=None):
    '''
    Parameters
    ----------
    df: Pandas Dataframe
        Dataframe of bonds' data
    needed_bonds: list-like
        Array of bond names which will be considered in construction of yield curve
    max_yield: float
        Transactions with yield higher than this number will not be used
    use_otc: bool
        Use transactions from OTC market or not. Default - False
    notes_in_otc: bool
        Use notes of NBRK in OTC market. This parameter is considered if only use_otc is False.
    maturity_filter: float
        Filtering transaction which remaining days to maturity less than maturity_filter
    specific_deals: list of strings
        Array of specific deals'codes to exclude from dataframe 
    '''
    #####---------#### do not forget
    if deal_market is not None:
        df = df[df.deal_type == deal_market]
    if needed_bonds is None:
        needed_bonds = ['MOM', 'MKM', 'NTK', 'MUM']
    if use_otc == False:
        if notes_in_otc == False:
            df = df[df.type_of_market != 'OTC']
        else:
            df = df[(df.type_of_market != 'OTC') | (df.symbol.str.match(pat='^NTK'))]
        print('OTC', df.shape)
    if specific_deals is not None:
        df = df[~df.symbol.isin(specific_deals)]
        print('specific symbols', df.shape)
        
    #dropping empty columns
    df = df.dropna(how='all', axis=1)
    if maturity_filter is not None:
        df = df[df['span'] > maturity_filter]
        print('filtering by maturity', df.shape)
    #choosing specific type of government bonds
    needed_bonds = '|'.join(['^' + code for code in needed_bonds])
    df = df[df.symbol.str.match(pat=needed_bonds)]
    print('filtering by bonds', df.shape)
    return df

#### Groupping of deals is needed for optimization of calculation
def groupping_transactions(df):
    clean_data = df
    ind_col = ['deal_date', 'symbol', 'deal_price']
    #aggragating data either by mean or median
    aggregated_by_median = df.groupby(ind_col)[['face_value', 'annual_freq', 'base_time', 'span']].median()
    print('median', aggregated_by_median.shape)
    aggregated_by_mean = df.groupby(ind_col)[['stand_price', 'clean_price', 'coupon_rate', 'ytm']].mean()
    print('mean', aggregated_by_mean.shape)
    #Why 9? --- Why not?
    aggregated_by_mean.ytm = aggregated_by_mean.ytm.fillna(9)
    #defining and filling clean_data with data
    clean_data = df.groupby(ind_col)[['volume_kzt']].sum()

    clean_data = pd.merge(clean_data, aggregated_by_median, left_index=True, right_index=True)
    clean_data = pd.merge(clean_data, aggregated_by_mean, left_index=True, right_index=True)
    clean_data['end_date'] = df.groupby(ind_col).end_date.first()
    clean_data['annual_freq'] = clean_data['annual_freq'].fillna(1)
    #adding span and changing coupon rate variable
    clean_data = clean_data.reset_index().set_index(ind_col)
    print(clean_data.shape)
    return clean_data

####
def processing_data(dataframe, mask_face_value, mask_base_time, 
              needed_bonds=None, use_otc=False, notes_in_otc=False, deal_market=None,
              maturity_filter=None, specific_deals=None, is_to_print=True):
    '''
    Transform raw dataset of bond's deal to clean dataframe, 
    on which algorithm is ready to be optimized
    
    Parameters
    ----------
    dataframe: Pandas Dataframe
        Dataframe of bonds' data
    needed_bonds: list-like
        Array of bond names which will be considered in construction of yield curve
    max_yield: float
        Transactions with yield higher than this number will not be used
    use_otc: bool
        Use transactions from OTC market or not. Default - False
    notes_in_otc: bool
        Use notes of NBRK in OTC market. This parameter is considered if only use_otc is False.
    maturity_filter: float
        Filtering transaction which remaining days to maturity less than maturity_filter
    specific_deals: list of strings
        Array of specific deals'codes to exclude from dataframe 
        
    Returns
    ------------
    Transformed and filtered Pandas DataFrame of bonds' data
    '''
    original_stdout = sys.stdout
    if not is_to_print:
        sys.stdout = None
    try:    
        clean_data = (dataframe.pipe(creating_new_columns, mask_face_value, mask_base_time)
                               .pipe(cleaning_from_errors)
                               .pipe(coupon_rate_modifier)
                               .pipe(filling_values)
                               .pipe(filtering, needed_bonds=needed_bonds, 
                                     use_otc=use_otc, deal_market=deal_market,
                                     notes_in_otc=notes_in_otc, maturity_filter=maturity_filter, 
                                     specific_deals=specific_deals)
                               .pipe(groupping_transactions)
                       )
    except Exception as e:
        print(e)
        sys.stdout = original_stdout
        raise 
    sys.stdout = original_stdout
    return clean_data

####download or save clean data to avoid non-needed calculation
def read_download_preprocessed_data(save_data, clean_data_path, clean_data=None):
    '''
    Parameters
    ------------
    save_data: bool
        Whether to load from HDF5 or save clean data into it
    clean_data_path: str
        Path to HDF5 which will contain/contains processed data
    clean_data: Pandas Dataframe
        Preprocessed pandas dataframe. Will be ridden to HDF5 file if only save_data is True
    '''
    with pd.HDFStore(clean_data_path, format='table') as hdf_clean_data:
        if os.path.isfile(clean_data_path):
            if not hdf_clean_data.keys():
                print('There is no clean data HDF5 file')
                iteration = 0
            else:
                #index latest version of clean data in hdf file
                iteration = max(map(lambda x: int(x[1:]), hdf_clean_data.keys())) 
        if save_data:
            iteration += 1
            print(iteration)
            assert 'clean_data' is not None, 'clean_data dataframe was not provided'
            print('writting new version of clean data in HDF5 file')
            hdf_clean_data['/' + str(iteration)] = clean_data
        else:
            print('old clean data will be downloaded')
            clean_data = hdf_clean_data['/' + str(iteration)]
    return clean_data