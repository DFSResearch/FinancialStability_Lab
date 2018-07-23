import pandas as pd
import numpy as np
import CONFIG 

#function of getting adaptive rho
#0.1 ~ 10% weight of the latest deal; 
base_time_window = 30
rho_func = lambda max_days: np.exp(np.log(0.1) * base_time_window / max_days)
#weight of deal based on rho
rev_span_weight = lambda x, rho: np.exp((x / base_time_window) * np.log(rho)) 

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
        
    #different weights and their calculation:    
    def complex_volume(self):
        self.df['vol_sector'] = equal_slicing_maturity(self.df, self.n_cuts)
        vol_sector = self.df.groupby('vol_sector').volume_kzt.sum()
        Vq = self.df['volume_kzt'] / self.df['vol_sector'].map(vol_sector)
        self.rho_w = np.exp(np.log(self.rho) / 100  * self.df['reverse_span'])
        Wq = (Vq * self.rho_w / self.rho_w.sum()).values
        return Wq
    
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