import pandas as pd
import numpy as np
from itertools import product
try:
    from sklearn.ensemble import IsolationForest
    possible_detect_outliers = True
except ImportError as e:
    possible_detect_outliers = False
import CONFIG

def adaptive_samples(df, time_window, min_n_deal=10, all_baskets_fixed=True):
    
    big_ind = []
    if not all_baskets_fixed:
        
        for mat_type in df.bond_maturity_type.unique():
            #filtering by time span of bond and reverse span
            deals = df[(df.bond_maturity_type == mat_type)]
            filtered_deals = deals[deals.reverse_span < time_window]
            #saving time_window before overwritting it in loop
            temp_time_wind = time_window
            #loop while there is too few deals
            while filtered_deals.shape[0] < min_n_deal:
                temp_time_wind += 5
                if temp_time_wind > 180:
                    break
                filtered_deals = deals[deals.reverse_span < temp_time_wind]
            big_ind.extend(filtered_deals.index.values)
        df_ = df.loc[big_ind]
        
        #Add medium term deals if they exempt from sample
        n_additional_deals = 5
        if df_.span.max() < 3 * 365:
            long_span_df = df[df.span > 3 * 365].sort_values(by='reverse_span').iloc[:n_additional_deals - 1]
            big_ind.extend(long_span_df.index.values)
        #Add long term deals if they exempt from sample
        if df_.span.max() < 8 * 365:
            long_span_df = df[df.span >= 8 * 365].sort_values(by='reverse_span').iloc[:n_additional_deals - 1]
            big_ind.extend(long_span_df.index.values)
        df = df.loc[big_ind]
    else:
        for mat_type in df.bond_maturity_type.unique():
            
            #filtering by time span of bond and reverse span
            deals = df[df.bond_maturity_type == mat_type].sort_values(by='reverse_span')
            deals['count_deals'] = range(1, deals.shape[0] + 1)
            rev_span_deals = deals.groupby('reverse_span').count_deals.last()
            
            #taking most Nth recent_deals
            needed_rev_span = rev_span_deals >= min_n_deal
            if  (~needed_rev_span).all():
                print(f'Too few deals for tenors in {mat_type}, # deals less than {min_n_deal}')
                rev_span_cut = rev_span_deals.index.max()
            else:
                rev_span_cut = rev_span_deals[needed_rev_span].index.min()                
            big_ind.extend(deals[deals.reverse_span <= rev_span_cut].index.values)
            
    df = df.loc[big_ind]
    return df

def choosing_time_frame(settle_date, clean_data, number_cuts=3, lookback=180,
                        max_days=180, time_window=30, all_baskets_fixed=True, 
                        min_n_deal=10, fix_first_cut=True, baskets = False):
    
    df = (clean_data.reset_index()
                     .assign(settle_date = pd.to_datetime(settle_date))
                     .query('settle_date > deal_date')
                     .assign(reverse_span = lambda x: (x.settle_date - x.deal_date).dt.days))
    
    if all_baskets_fixed:
        df = df.query('(reverse_span < @max_days)')
        
        if baskets == False:
            treshold = [0, 190, 370, 5 * 365, np.inf]
        else:
            treshold = baskets
    else:
        df_ = df.query('(settle_date < end_date)')
        treshold = [0]
        #if we want to fix cut at first year
        if fix_first_cut:
            df_ = df_[df_['span'] > 365]
            treshold.append(365)
        df_ = df_[df_['reverse_span'] < lookback]
        cum_span = df_.span.value_counts().sort_index().cumsum()
        base = cum_span.max() / number_cuts
        
        #creating treshold for finding range of spans which contain 
        #equal amount of deals
        cut = 0
        for i in range(number_cuts):
            cut += base
            cut_line = cum_span[cum_span <= cut].idxmax()
            #minimum length of interval is 60 -- some empirical consideration bases
            while (cut_line - treshold[-1]) < 60:
                cut += 1
                cut_line = cum_span[cum_span <= cut].idxmax()
                if cut_line == cum_span.idxmax():
                    print('Number of cuts is too high')
                    break
            treshold.append(cut_line)
    df.loc[:,'bond_maturity_type'] = pd.cut(df.span, bins=treshold)
    df = df[df.reverse_span < max_days]
        
    #filtering based on time window  
    filtered_data = adaptive_samples(df, time_window=time_window, min_n_deal=min_n_deal,
                                     all_baskets_fixed=all_baskets_fixed)
    return filtered_data.set_index(['deal_date', 'symbol', 'deal_price'])

def outlier_detection(data, contamination=0.015, n_jobs=1, **kwargs):
    isoforest = IsolationForest(contamination=contamination, n_jobs=n_jobs, 
                                random_state=123, **kwargs)
    data['span*ytm'] = (data['ytm']) * np.log(data['span'])
    X = data[['stand_price', 'ytm', 'span*ytm']]
    isoforest.fit(X)
    data = data[isoforest.predict(X) == 1]
    return data

def creating_sample(settle_date, data, time_window, min_n_deal, number_cuts=3, 
                    lookback=CONFIG.LOOKBACK, max_days=CONFIG.MAX_DAYS, adaptive=False, alpha=0.5, 
                    fix_first_cut=True, detect_outlier=CONFIG.DETECT_OUTLIERS, all_baskets_fixed=True, thresholds = False):
    '''
    Creates dataset for a given settle date
    Parameters
    -----------
    settle_date: str, datetime
        Settle date on which curve will be constructed
    data: Pandas DataFrame
        Cleaned from errors dataframe of bonds' deals
    time_window: int
        Initial number of days on which curve will be estimated.
    min_n_deal: int
        Minimum of number deals in each sub-sample
    number_cuts: int
        Number of sub-samples in sample data
    lookback: int
        Number of days
    '''
    thresholds = thresholds
    if adaptive:
       ##choosing right k
        ncuts_ser = pd.Series()
        k_range = range(1, 6) if fix_first_cut else range(2, 7)
        
        for k in k_range:
            try:
                k_data = choosing_time_frame(settle_date, data, number_cuts=k, max_days=max_days,
                                           lookback=lookback, min_n_deal=min_n_deal, 
                                           time_window=time_window)
                k_data['gr_weight'] = np.log1p(k_data.reverse_span)
                sizes = k_data.groupby('bond_maturity_type')['gr_weight'].sum().values
                n_lt_deals = k_data[k_data.span > 10 * 365].shape[0]
            except ValueError as e:
                print(f'error happend at {k} -- too high k', e)
                sizes = [1e9, 0]
                n_lt_deals = 0
            sizes = k_data.groupby('bond_maturity_type').size().values
            sum_of_diff = sum(set([abs(x - y) for x, y in 
                                        product(sizes, sizes) if x != y]))
            loss_met = alpha * sum_of_diff - (1 - alpha) * (n_lt_deals ** 2)
            ncuts_ser[str(k)] = loss_met
        #right k is the k that minimize loss
        number_cuts = int(ncuts_ser.argmin())
        
    if fix_first_cut:
        number_cuts = number_cuts - 1
        
    data = choosing_time_frame(settle_date, data, number_cuts=number_cuts, max_days=max_days,
                               lookback=lookback, min_n_deal=min_n_deal, all_baskets_fixed=all_baskets_fixed,
                               time_window=time_window, fix_first_cut=fix_first_cut, baskets = thresholds)
    #throwing out outliers
    if detect_outlier:
        if possible_detect_outliers:
            data = outlier_detection(data)
        else:
            print('For detecting outliers sklearn package should be installed')
    return data
