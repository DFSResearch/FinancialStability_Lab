import pandas as pd
import numpy as np
from error_measures import area_boot, MAE_YTM
import CONFIG
from datapreparation.adaptive_sampling import creating_sample
from optimizator.simultaneous_min import iter_minimizer
from optimizator.gridsearch import grid_search

class stability_assession():
    
    def __init__(self, start_date, end_date, big_dataframe, beta_init, freq='1w', baskets = False):
        self.start_date = start_date
        self.end_date = end_date
        self.big_dataframe = big_dataframe
        self.freq = freq
        self.beta_init = beta_init
        self.settle_dates = pd.date_range(start=self.start_date, end=self.end_date, 
                                  normalize=True, freq=self.freq, closed='right')
        self.thresholds = baskets
    
    def __str__(self):
        s = f"""The curve optimization is initialized for parameters below:
		       Start date: {self.start_date}
			   End date: {self.end_date}
			   Thresholds: {self.thresholds}
			   Initial beta: {self.beta_init}
			   Remaining deal weight: {CONFIG.RHO}
			   Minimum number of deals: {CONFIG.MIN_N_DEAL}
			   Weight scheme: {CONFIG.WEIGHT_SCHEME}
			   
		   """
        return s
    
    def optimize(self, data, settle_date = None, **kwargs):
        '''
        Parameters
        -------------
        data: Pandas DataFrame
            Dataframe of bonds' data
        solver_type: str
            One of ['simult', 'grid']. 'simult' try to minimize 4-parameters at the same time. 
            Scheme 'simult' is not stable, but is much faster than grid. 
            Because scheme 'grid' loops through array of different tetas
        '''
        
        ##args and parameters of Loss function
        #Longest matuiry year of deals in data
        max_deal_span = (data['span'] / 365).round().max()
        
        
        #Maturity limit for Zero-curve plot
        longest_maturity_year = max([max_deal_span, 20])
        theor_maturities = np.linspace(0.001, longest_maturity_year, 10000)
        #Tuple of arguments for loss function 
        loss_args = (data, self.coupons_cf, self.streak_data, CONFIG.RHO, CONFIG.WEIGHT_SCHEME)
        options = {'maxiter': 500, 'eps': 9e-5, 'disp': True}
        #optimization
        if self.solver_type == 'simult':
            res_ = iter_minimizer(Loss=self.loss, beta_init=self.beta_init, method=self.method,
                           loss_args=loss_args, bounds=bounds, max_deal_span=max_deal_span, **kwargs)
            return res_.x, settle_date
        else:
            
            beta_init = self.beta_init[:-1]
            bounds = self.bounds[:-1]
            grid = grid_search(tau_grid=self.tau_grid, Loss=self.loss, 
                               loss_args=loss_args, beta_init=beta_init)
            beta_best = grid.fit(return_frame=False, method=self.method, 
                                 num_workers=1, bounds=bounds,  **kwargs) 
         #constraints=constr
        return beta_best, theor_maturities
            
    def compute_curves(self, coupons_cf, streak_data, loss, method='SLSQP', solver_type='simult',
                       teta_min=CONFIG.TETA_MIN, teta_max=CONFIG.TETA_MAX, tau_grid=None, **kwargs):
        assert solver_type in ['simult', 'grid']
        self.solver_type = solver_type
        self.params = pd.Series()
        self.tenors = []
        self.tasks = []
        self.results = []
        self.coupons_cf, self.streak_data = (coupons_cf, streak_data)
        self.method, self.teta_min, self.teta_max, self.loss = (method, teta_min, teta_max, loss)
        #Parameters bounds for constraint optimization
        self.bounds = ((0, 1), (None, None), (None, None), (self.teta_min, self.teta_max))
        if tau_grid is None:
            self.tau_grid = np.append(np.arange(self.teta_min, 2, 0.1), np.arange(2, self.teta_max, 0.5))
        else: 
            self.tau_grid = tau_grid
            
        for settle_date in self.settle_dates:
            #print settle date in loop if we set in options display = True
            if 'options' in kwargs.keys():
                if 'disp' in kwargs['options'].keys():
                    if kwargs['options']['disp']:
                        print(settle_date)
                    
            data = creating_sample(settle_date, self.big_dataframe, min_n_deal=CONFIG.MIN_N_DEAL, 
                                   time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds)
            data['weight'] = weight([1, 1, 1, 1], data, CONFIG.WEIGHT_SCHEME)
            
            #saving data in dictionary for potential further usage
            if not hasattr(self, 'data_different_dates'):
                self.data_different_dates = {}
            self.data_different_dates[settle_date] = data
            #optimization and saving results for work
            self.tasks.append(delayed(self.optimize)(data, settle_date))
        
        self.results = compute(*self.tasks, get=dask.multiprocessing.get, num_workers=8)
        for betas, date in self.results:
            self.params[date] = betas
        #to save space delete calendar data
        del self.coupons_cf, self.streak_data
        return self
    
    def gen_subsets(self,):
        self.tasks = []
        self.data_different_dates = {}
		
        for settle_date in self.settle_dates:
			
            self.tasks.append(delayed(creating_sample)(settle_date, self.big_dataframe, min_n_deal=CONFIG.MIN_N_DEAL, 
                                   time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds))
            self.data_different_dates[settle_date] = ''
		
        self.results = compute(*self.tasks, get=dask.multiprocessing.get, num_workers=8)
		
        for i, settle_date in enumerate(self.settle_dates):
            self.data_different_dates[settle_date] = self.results[i]

        self.results = []
        
    #area between envelope llines -- bigger area bigger volatility of curve
    def asses_area(self):
        '''
        area between envelope llines -- bigger area bigger volatility of curve
        Returns
        ------
        Total area between curves, Average area between cuves
        '''
        try:
            betas_array = self.params.apply(pd.Series).values
        except AttributeError as e:
            print("Model didn'f fitted to data use compute_curves() method to fit it")
            raise AttributeError
        position_of_max_tenor = pd.DataFrame(self.tenors).iloc[:, -1].values.argmax()
        area, av_area = area_boot(betas_array, self.tenors[position_of_max_tenor])
        return area, av_area
