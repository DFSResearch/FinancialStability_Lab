import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ns_func import Z
from datapreparation.adaptive_sampling import creating_sample
import CONFIG

#checking if dask is installed
try:
    import dask.multiprocessing
    from dask import compute, delayed
    use_one_worker = False
except ImportError as e:
    use_one_worker = True

##grid search over values of tau
class grid_search():
    def __init__(self, tau_grid, 
                 Loss, beta_init, 
                 loss_args, start_date, 
                 end_date, freq, 
                 toniaDF,
                 maturities=None, 
                 clean_data = None, 
                 thresholds = [0,190,370,5*365,np.inf],
                 several_dates = False,
                 num_workers = 16,):
        
        self.Loss = Loss
        self.beta_init = beta_init
        self.loss_args = loss_args
        self.tau_grid = tau_grid
        self.maturities = maturities
        self.results = []
        self.loss_res = {}
        self.several_dates = several_dates
        self.thresholds = thresholds
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.num_workers = num_workers
        self.tonia_df = toniaDF
        
        if self.maturities is None:
            self.maturities = np.arange(0.0001, 30, 1 / 12) 
        if clean_data is not None:
            self.raw_data = clean_data
            
            
    #actual minimizaiton
    def minimization_del(self, tau, Loss, loss_args, beta_init, **kwargs):
        print('now tau = ', tau)
        l_args = [arg for arg in loss_args]
        l_args.append(tau)
        l_args = tuple(l_args)
        res_ = minimize(Loss, beta_init, args=l_args, **kwargs, 
                        callback=lambda xk: print(xk, Loss(beta=xk, df=l_args[0], 
                                 coupons_cf=l_args[1], streak_data=l_args[2], rho=l_args[3], 
                                 weight_scheme=l_args[4], tau=tau))
                        ) 
                        
        return res_
    
    #generating multiple samples
    def gen_subsets(self,):
        self.tasks = []
        self.data_different_dates = {}
		
        for settle_date in self.settle_dates:
			
            self.tasks.append(delayed(creating_sample)(settle_date, self.raw_data, min_n_deal=CONFIG.MIN_N_DEAL, 
                                   time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds))
            self.data_different_dates[settle_date] = ''
		
        self.results = compute(*self.tasks, scheduler='processes', num_workers=self.num_workers)
		
        for i, settle_date in enumerate(self.settle_dates):
            if self.results[i].index.isin([(pd.to_datetime('2018-07-12 15:39:57'), 'MUM132_0002', 761.0591)]).any():
                self.results[i].drop([(pd.to_datetime('2018-07-12 15:39:57'), 'MUM132_0002', 761.0591)], inplace = True)
                print('Generating sample for {settle_date} - Done!')
            self.data_different_dates[settle_date] = self.results[i]

        self.results = []
        
    #creation of loss frame grid
    def loss_grid(self, **kwargs):
        #if num_worker == 1 dask will not be used at all to avoid overhead expenses
        if self.num_workers == 1:
            res_ = []
            for i, tau in enumerate(self.tau_grid):
                res = self.minimization_del(tau, self.Loss, 
                          self.loss_args, self.beta_init, **kwargs)
                res_.append(res)
        
        elif self.several_dates:
            
            
            self.settle_dates = pd.date_range(start=self.start_date, end=self.end_date, 
                                  normalize=True, freq=self.freq, closed='right')
            
            loss_args = self.loss_args
            
            if not hasattr(self, 'data_different_dates'):
                self.data_different_dates = {}
                
            self.gen_subsets()
            
            for date, dataset in self.data_different_dates.items():
                
                l_args = [arg for arg in loss_args]

                l_args[0] = dataset
                l_args = tuple(l_args)
                
                constr = ({'type':'eq',
                           'fun': lambda x: np.array(x[0] + x[1]- self.tonia_df.loc[date][0])},)
    
                #parallelization of loop via dask multiprocessing
                values = [delayed(self.minimization_del)(tau, self.Loss, 
                          l_args, self.beta_init, constraints = constr, **kwargs) for tau in self.tau_grid]
    
                res_ = compute(*values, scheduler='processes', num_workers=self.num_workers)
                
                #putting betas and Loss value in Pandas DataFrame
                loss_frame = pd.DataFrame([], columns=['b0', 'b1', 'b2', 'teta', 'loss'])
                loss_frame['b0'] = [res.x[0] for res in res_]
                loss_frame['b1'] = [res.x[1] for res in res_]
                loss_frame['b2'] = [res.x[2] for res in res_]
                loss_frame['teta'] = [t for t in self.tau_grid]
                loss_frame['loss'] = [res.fun for res in res_]
            
                self.loss_res[date] = loss_frame
                print(f'Done for {date}')
        
        else:
            #parallelization of loop via dask multiprocessing
            values = [delayed(self.minimization_del)(tau, self.Loss, 
                      self.loss_args, self.beta_init, **kwargs) for tau in self.tau_grid]
            res_ = compute(*values, get=dask.multiprocessing.get, num_workers=self.num_workers)
            
            #putting betas and Loss value in Pandas DataFrame
            loss_frame = pd.DataFrame([], columns=['b0', 'b1', 'b2', 'teta', 'loss'])
            loss_frame['b0'] = [res.x[0] for res in res_]
            loss_frame['b1'] = [res.x[1] for res in res_]
            loss_frame['b2'] = [res.x[2] for res in res_]
            loss_frame['teta'] = [t for t in self.tau_grid]
            loss_frame['loss'] = [res.fun for res in res_]
            
        return loss_frame
    
    #filtering frame from unacceptable data (spot rates < 0)
    def filter_frame(self, loss_frame):
        accepted_ind = []
        for ind in loss_frame.index:
            beta = loss_frame.loc[ind, loss_frame.columns[:-1]]
            spot_rate_curve = Z(self.maturities, beta) 
            if (spot_rate_curve >= 0).all():
                accepted_ind.append(ind)
        loss_frame_filtered = loss_frame.loc[accepted_ind, :]
        #printing info about № of dropped rows
        n_rows = loss_frame.shape[0]
        n_dropped_rows = n_rows - loss_frame_filtered.shape[0]
        print('{0} out of {1} of rows were dropped'.format(n_dropped_rows, n_rows))
        return loss_frame_filtered
    
    #actual fitting of data
    def fit(self, num_workers=8, return_frame=False, **kwargs):
        if use_one_worker:
            print('Multiprocessing is not enabled as dask is not installed\n'
                  'Install dask to enbale multiprocessing')
            self.num_workers = 1
        else:
            self.num_workers = num_workers
        self.loss_frame = self.loss_grid(**kwargs)
        #loss_frame = self.filter_frame(loss_frame)
        self.beta_best = self.loss_frame.loc[self.loss_frame['loss'].argmin(), :].values[:-1]
        self.beta_init = self.beta_best
        
        if return_frame:
            return self.beta_best, self.loss_frame
        else:
            return self.beta_best
