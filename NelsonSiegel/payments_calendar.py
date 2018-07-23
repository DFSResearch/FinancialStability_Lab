import os
import pandas as pd

def coupon_payments(face_value, coupon_rate, time_span, frequency=2, year_base=365):
    '''
    Calculates coupon payments of bond
    '''
    #days
    days_of_full_payments = year_base / frequency
    number_of_full_periods = int(time_span // days_of_full_payments)
    left_days = time_span % days_of_full_payments
    days = [days_of_full_payments] * number_of_full_periods
    
    if left_days != 0:
        days = [left_days] + days
    if coupon_rate == 0:
        coupons = [0] * len(days)
        coupons[-1] += face_value
    else:
        coupons = [face_value * coupon_rate / frequency] * len(days)
        coupons[-1] += face_value
    
    coupon_payments = pd.DataFrame()
    coupon_payments['cf'] = coupons
    coupon_payments['days_to_next_paym'] = days
    coupon_payments['cum_year'] = (coupon_payments['days_to_next_paym'].cumsum() / year_base)
    return coupon_payments

def creating_coupons(df):
    coupons_cf   = pd.DataFrame()
    streak_data  = pd.DataFrame()
    #going with loop over each bond and appending its cash flow graphic to dataframe
    for i, ind in enumerate(df.index):
        deal_row = df.iloc[i]
        try:
            coupons_df = coupon_payments(100, deal_row['coupon_rate'], deal_row['span'], 
                                         deal_row['annual_freq'], deal_row['base_time'])
        except Exception as e:
            print(i, ind, '\n')
            print(e) 
            break
        coupons_cf = pd.concat([coupons_cf, coupons_df.cf.rename(ind)], axis=1)
        streak_data = pd.concat([streak_data, coupons_df.cum_year.rename(ind)], axis=1)
    return coupons_cf, streak_data

def download_calendar(dataset, hdf_coupons_path=os.path.join('datasets', 'coupons_data.hdf'), 
                      delete_previous=False):
    '''
    Creating or downloading calendar payments of each deal in dataset
    Parameters
    ------------
    dataset: Pandas DataFrame
        Dataframe of deals' data
    hdf_coupons_path: str
        path to HDF5 file
    delte_previous: bool, default False
        If true deletes currently existing calendar data HDF5 file
    '''
    #Check for coupon data, if file already exists - just download it
    with pd.HDFStore(hdf_coupons_path, format='table') as hdf_handler:
        if hdf_handler.keys():
            print('Downloading calendar data for bonds')
            coupons_cf  = hdf_handler['coupons_cf']
            streak_data = hdf_handler['time_of_cf']

            absent_symbols = [i for i in range(dataset.shape[0])
                             if dataset.index[i] not in coupons_cf.columns]
            if absent_symbols:
                print('There are new bonds in clean data')
                print('Starting to estimate payment calendar for them')
                coupons_cf_abs, streak_data_abs = creating_coupons(dataset.iloc[absent_symbols])
                coupons_cf = pd.concat([coupons_cf, coupons_cf_abs], axis=1)
                streak_data = pd.concat([streak_data, streak_data_abs], axis=1)
                hdf_handler['coupons_cf'] = coupons_cf
                hdf_handler['time_of_cf'] = streak_data
        else:
            print('Completely new dataset for estimation payment calendar')
            coupons_cf, streak_data   = creating_coupons(dataset)
            hdf_handler['coupons_cf'] = coupons_cf
            hdf_handler['time_of_cf'] = streak_data
    return coupons_cf, streak_data