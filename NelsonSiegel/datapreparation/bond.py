import pandas as pd
import numpy as np

class Bond():
    '''
    Parameters
    ----------
    dataset: Pandas Dataframe
        Dataframe of bonds' data
    symbol: str
        KASE's unique symbol of bond
    '''
    
    def __init__(self, dataset, symbol):
        self.symbol = symbol
        self.dataset = dataset[dataset['symbol'] == symbol]
        self.setting_bond_features()
        
    def setting_bond_features(self):
        self.face_value = self.dataset['face_value'].unique()[0]
        self.frequency = self.dataset['annual_freq'].unique()[0]
        self.convention = self.dataset['base_time'].unique()[0]
        self.end_date = self.dataset['end_date'].unique()[0]
        #default values for unusual coupon rate 
        self.unusual_cr = False
        #trying to input coupon rate
        discount_bond_symbols = ['NTK', 'MKM']
        if self.symbol[:3] in discount_bond_symbols: 
            self.is_discount = True
            coupon_rate = 0
        else:
            self.is_discount = False
            first_date = (self.dataset.set_index('deal_date')
                               .sort_index(ascending=True)
                               .iloc[0])
            coupon_rate = first_date['coupon_rate']
            if np.isnan(coupon_rate):
                #first try to use ytm
                if ~np.isnan(first_date.ytm) and (first_date.ytm != 0):
                        coupon_rate = first_date.ytm
                        self.unusual_cr = True if coupon_rate > 25 else False
                #then if it fail, clean_price    
                else:
                    coupon_rate = first_date.clean_price
                    self.unusual_cr = True if (coupon_rate == 0 or 
                                               coupon_rate > 25) else False
        #properly scaling of coupon rate            
        self.coupon_rate = coupon_rate / 100
        assert ~np.isnan(self.coupon_rate), self.symbol
        return self
    
    def deals_calendar(self, deals_dates=None, deal_prices=None):
        if deals_dates is None:
            assert deal_prices is not None, 'Deal prices is provided wo deal dates!'
            deal_dates = self.dataset.deal_date.unique()
            deal_prices = self.dataset.deal_price.unique()
        self.streak_data = pd.DataFrame()
        self.coupons_cf = pd.DataFrame()
        for i, (deal_date, deal_price) in enumerate(zip(deal_dates, deal_prices)):
            deal_ = deal(deal_date, deal_price, 
                        dataset=self.dataset, symbol=self.symbol)
            coupons_cf, streak_data = deal_.payment_calendar()
            self.coupons_cf = pd.concat([coupons_cf, self.coupons_cf], axis=1)
            self.streak_data = pd.concat([streak_data, self.streak_data], axis=1)
            
        #setting correct column names    
        deal_indeces = pd.MultiIndex.from_arrays([deal_dates, 
                                                   [self.symbol] * deal_dates.shape[0], 
                                                   deal_prices])
        self.coupons_cf.columns = deal_indeces
        self.streak_data.columns = deal_indeces
        return self.coupons_cf, self.streak_data
    
class deal(Bond):
    def __init__(self, deal_date, deal_price, **kwargs):
        Bond.__init__(self, **kwargs)
        self.deal_date = pd.to_datetime(deal_date)
        self.deal_price = deal_price

    def payment_calendar(self):
        self.span = self.dataset.loc[(self.dataset.deal_date == self.deal_date) &
                                      (self.dataset.deal_price == self.deal_price), 
                                         'span']
        assert self.span.shape[0] == 1
        self.span = self.span.iloc[0]
        #100 - standartized face value of bond
        coupons_df = coupon_payments(100, self.coupon_rate, self.span, self.frequency, self.convention)
        coupons_cf = coupons_df.cf
        streak_data = coupons_df.cum_year
        return coupons_cf, streak_data
    
#method modify coupon rate of bond in input dataset
def coupon_rate_modifier(df):
    all_bonds_symbols = df['symbol'].unique()
    for bond_symbol in all_bonds_symbols:
        df.loc[df['symbol'] == bond_symbol, 
                              'coupon_rate'] = Bond(df, bond_symbol).coupon_rate
    return df