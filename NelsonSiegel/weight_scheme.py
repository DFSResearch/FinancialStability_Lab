import pandas as pd
import numpy as np
import CONFIG

#function of getting adaptive rho
#0.1 ~ 10% weight of the latest deal;
base_time_window = 30
weight_of_deal = 0.1

rho_func = lambda max_days: np.exp(np.log(weight_of_deal) * base_time_window / max_days)
#weight of deal based on rho
rev_span_weight = lambda x, rho: np.exp(x*rho)

#slicing tenor on equal, by number of deals, slices
def equal_slicing_maturity(df, n_cuts):
    cum_span = df.span.value_counts().sort_index().cumsum()
    base = np.ceil(cum_span.max() / n_cuts)
    cut = 0
    treshold = []
    #creating treshold for finding range of spans which contain
    #equal amount of deals
    for i in range(n_cuts):
        if not treshold:
            treshold.append(cut)
        cut += base
        while cum_span[cum_span <= cut].shape[0] == 0:
            cut += 1
        cut_line = cum_span[cum_span <= cut].idxmax()
        treshold.append(cut_line)
    mat_type = pd.cut(df.span, bins=treshold)
    return mat_type

class WeightScheme():
    def __init__(self, beta, dataset, time_window=CONFIG.TIME_WINDOW,
                            n_cuts=CONFIG.N_CUTS, rho=CONFIG.RHO):
        self.beta = beta
        self.df = dataset
        self.time_window = time_window
        self.n_cuts = n_cuts
        self.rho = rho
        
    rho_func = lambda self, wl ,max_days: np.log(wl) / max_days
    
    #different weights and their calculation:
    def complex_volume(self):
        self.df['vol_sector'] = equal_slicing_maturity(self.df, self.n_cuts)
        vol_sector = self.df.groupby('vol_sector').volume_kzt.sum()
        Vq = self.df['volume_kzt'] / self.df['vol_sector'].map(vol_sector)
        self.rho_w = np.exp(np.log(self.rho) / 100  * self.df['reverse_span'])
        Wq = (Vq * self.rho_w / self.rho_w.sum()).values
        return Wq
    
    def complex_new_av(self):
        
        #weight of each maturity basket: 0.25 for 4 baskets
        W_gr = 1 / self.df['bond_maturity_type'].nunique()
               
        for mat_type in self.df['bond_maturity_type'].unique():
			
			#deals in this basket
            deals = self.df[self.df['bond_maturity_type'] == mat_type]
            

            self.df.loc[self.df['bond_maturity_type'] == mat_type, 'rho'] = self.rho_func(self.last_weight, self.df.reverse_span.max())
            
            #weight by age
            Wq = rev_span_weight2(self.df.loc[self.df['bond_maturity_type'] == mat_type, 'reverse_span'],
                                 self.df.loc[self.df['bond_maturity_type'] == mat_type, 'rho'])
            Wq = 10**(-deals.reverse_span/self.df.reverse_span.max())
			
            #weight by volume
            Vq = np.log(deals['volume_kzt']) / np.log(deals['volume_kzt'].sum())
            W = (Wq * Vq).values
            
            #normalization
            if W.shape[0] > 1:
                W = W / W.sum()
            else:
                W = 1
            self.df.loc[self.df['bond_maturity_type'] == mat_type, 'weight'] =  W * W_gr

        return self.df['weight'].values
    #even weights between slices but uneven in the slice itself
    def complex_even(self):
        #weight of each slice
        W_gr = 1 / self.df['bond_maturity_type'].nunique()
         #weight deals depending on mean reverse span
        rev_span_mean = self.df.reverse_span.mean()
        self.df['rho'] = rho_func(rev_span_mean)
        for mat_type in self.df['bond_maturity_type'].unique():
            #deals in this slice
            deals = self.df[self.df['bond_maturity_type'] == mat_type]
            Wq = rev_span_weight(self.df.loc[self.df['bond_maturity_type'] == mat_type, 'reverse_span'],
                                 self.df.loc[self.df['bond_maturity_type'] == mat_type, 'rho'])
            #normalize by volume -- depending on subsample
            Vq = deals['volume_kzt'] / deals['volume_kzt'].sum()
            W = (Wq * Vq).values
            if W.shape[0] > 1:
                W = W / W.sum()
            else:
                W = 1
            self.df.loc[self.df['bond_maturity_type'] == mat_type, 'weight'] =  W * W_gr

        return self.df['weight'].values

    def even(self):
        mat_type = unequal_slice(self.df, shrink=True)
        w = (1 / self.n_cuts) / mat_type.value_counts() * self.df.shape[0]
        Wq = mat_type.map(w).values
        return Wq

    def volume_kzt(self):
        Vq = self.df.volume_kzt * np.exp((self.df['reverse_span'] / self.time_window) * np.log(self.rho))
        Wq = Vq.values
        return Wq

    def rev_span(self):
        #weight deals depending on reverse span
        rev_span_max = self.df.groupby('bond_maturity_type').reverse_span.max()
        self.df['rho'] = self.df.bond_maturity_type.map(rho_func(rev_span_max))
        W = rev_span_weight(self.df['reverse_span'], self.df['rho'])
        #normalize by volume -- depending on subsample
        vol_sector = self.df.groupby('bond_maturity_type').volume_kzt.sum()
        Vq = self.df['volume_kzt'] / self.df['bond_maturity_type'].map(vol_sector)
        Wq = (W * Vq).values
        return Wq

    def rev_span_only(self):
        rev_span_max = self.df.groupby('bond_maturity_type').reverse_span.max()
        self.df['rho'] = self.df.bond_maturity_type.map(rho_func(rev_span_max))
        Wq = rev_span_weight(self.df['reverse_span'], self.df.rho).values
        return Wq

    def rev_span_all(self):
        #weight deals depending on reverse span
        rev_span_max = self.df.groupby('bond_maturity_type')['reverse_span'].max()
        self.df['rho'] = self.df.bond_maturity_type.map(rho_func(rev_span_max))
        W = rev_span_weight(self.df.reverse_span, self.df.rho)
        #normalize by volume
        Wq = (self.df.volume_kzt * W / self.df.volume_kzt.sum()).values
        return Wq

    def no_weight(self):
        Wq = np.ones((self.df.shape[0],))
        return Wq

    def tenor_weight(self):
        Wq = self.df.span / self.df.span.sum()
        return Wq

def weight(beta, df, weight_scheme, **kwargs):
    '''
    Parameters
    ---------------
    beta: array-like
        Should be length of 4. Nelson-Siegel's vector of parameters
    df: Pandas Dataframe
        dataframe of bond's data
    weight_scheme: str
        Name of weight scheme used for weightning deals
        List of currently available weight schemes:
        ['complex_volume', 'even', , 'no_weight', 'volume_kzt',
        'rev_span', 'rev_span_all', 'rev_span_only', 'tenor_weight']
    '''
    def callMethod(ObjectInstance, method_name):
        return getattr(ObjectInstance, method_name)()
    #cheking on available weight schemes in class WeightScheme
    available_weight_schemes = [func for func in dir(WeightScheme)
                       if callable(getattr(WeightScheme, func)) and not func.startswith('__')]
    weight_calculation = WeightScheme(beta, df, **kwargs)
    if weight_scheme not in available_weight_schemes:
        raise ValueError(f"{weight_scheme} is not type of weight scheme. \n"
                         f"Use instead one of these schemes: \n{available_weight_schemes}")
    Wq = callMethod(weight_calculation, weight_scheme)
    return Wq
