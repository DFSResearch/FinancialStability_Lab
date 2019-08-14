import pandas as pd
import numpy as np
import pickle
import time
import h5py
from scipy.optimize import minimize

from datapreparation.adaptive_sampling import creating_sample
import CONFIG
from ns_func import Z

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
                 thresholds = [0,180,365,5*365,np.inf],
                 several_dates = False,
                 inertia = False,
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
        
        if self.maturities is None:
            self.maturities = np.arange(0.0001, 30, 1 / 12) 
        
        if clean_data is not None:
            self.raw_data = clean_data
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.num_workers = num_workers
        self.tonia_df = toniaDF
        self.inertia = inertia
        self.dropped_deals = {}
        if self.several_dates:
            self.settle_dates = pd.date_range(start=self.start_date, end=self.end_date, 
                                              normalize=True, freq=self.freq, closed='right')
        self.previous_curve = []
        self.tasks = []
        self.data_different_dates = {}
        self.data_different_dates = {}
        self.beta_best = None
        self.update_date = None
        self.iter_dates = None
        self.best_betas
        
    #actual minimizaiton
    def minimization_del(self, tau, Loss, loss_args, beta_init, **kwargs):
        '''
        Returns an array of beta parameters that minimizes loss function given value of tau
    
        Parameters:
        -----------
            tau : value of fixed tau
            Loss: loss function, by default yield loss function
            loss_agrs : a tuple of additional arguments to loss function
            beta_init: initial  guess for beta parameters that are used 
                       as optimization starting point
            
    
        Returns:
        --------
            res_ : result of optimization - [b0, b1, b2]
    
       
        '''
        print('now tau = ', tau)
        l_args = [arg for arg in loss_args]
        l_args.append(tau)
        l_args = tuple(l_args)
        res_ = minimize(Loss, beta_init, args=l_args, **kwargs, 
                        callback=lambda xk: print(xk, Loss(beta=xk, df=l_args[0], 
                                                           coupons_cf=l_args[1], 
                                                           streak_data=l_args[2], 
                                                           rho=l_args[3], 
                                                           weight_scheme=l_args[4], tau=tau))
                        ) 
                        
        return res_
    
    def is_outlier(self, points, thresh=3.5, score_type ='mzscore'):
        '''
        Returns a boolean array with True if points are outliers and False
        otherwise.
    
        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.
    
        Returns:
        --------
            mask : A numobservations-length boolean array.
    
        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), 'Volume 16: How to Detect and
            Handle Outliers', The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        '''
        if score_type == 'zscore':
            thresh = 2
            
#            if len(points.shape) >= 1:
#                points = points.loc[:,'ytm'][:,None]
            
            if (self.inertia) & (len(self.previous_curve)!=0):
                print(f'diff to previous curve, Z-score threshold: {thresh}')
                diff = (points.loc[:,'ytm']*100 - (np.exp(Z(points.loc[:,'span']/365, self.previous_curve))-1)*100)
            else:
    #            print(points)
                print('first filtering')
                mean = np.mean(points.loc[:,'ytm'])
                diff = (points.loc[:,'ytm'] - mean)
                
            
            sstd = np.std(diff)
        
            z_score = np.abs(diff / sstd)
            
        elif score_type == 'mzscore':
#            if len(points.shape) >= 1:
#                points = points.loc[:,'ytm'][:,None]
            
            if (self.inertia) & (len(self.previous_curve)!=0):
                print(f'diff to previous curve, modified Z-score threshold: {thresh}')
                diff = np.abs(points.loc[:,'ytm']- (np.exp(Z(points.loc[:,'span']/365, self.previous_curve))-1))*100
            else:
                print('first filtering')
                median = np.median(points.loc[:,'ytm'])
#                diff = (points.loc[:,'ytm']*100- median*100)**2
                diff = np.abs(points.loc[:,'ytm'] - median)*100
                
#            sstd = np.sqrt(diff)
            sstd = np.median(diff) #med_abs_deviation
        
            z_score = 0.6745 * diff / sstd

        return (z_score, (z_score > thresh), sstd)
    
    #filtered data generation
    def gen_subsets(self,):
        '''
        Generate a dictionary of pandas DataFrames. DataFrames represents the
        sample used for optimization for each date
    
        Parameters:
        -----------
            tau : value of fixed tau
            Loss: loss function, by default yield loss function
            loss_agrs : a tuple of additional arguments to loss function
            beta_init: initial  guess for beta parameters that are used 
                       as optimization starting point
            
    
        Returns:
        --------
            Returns nothing, 
            updates class data field: self.data_different_dates
    
       
        '''        
        self.tasks = []
        self.data_different_dates = {}
        
        if not self.settle_dates.size:
            self.settle_dates = pd.date_range(start=self.start_date, end=self.end_date, 
                                              normalize=True, freq=self.freq, closed='right')
		
        for settle_date in self.settle_dates:
			
            self.tasks.append(delayed(creating_sample)(settle_date, self.raw_data, min_n_deal=CONFIG.MIN_N_DEAL, 
                                                       time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds))
            self.data_different_dates[settle_date] = ''
		
        self.results = compute(*self.tasks, scheduler='processes', num_workers=self.num_workers)
        
		
        for i, settle_date in enumerate(self.settle_dates):
            ind_out=[]
            for b in self.results[i].bond_maturity_type.unique():
                bsample = self.results[i].loc[self.results[i].loc[:,'bond_maturity_type']==b]
                zscores = self.is_outlier(bsample.loc[:, ['ytm']])
                print(f'Z-score: {zscores[2]}')
                self.results[i].loc[self.results[i].loc[:,'bond_maturity_type']==b, 'std']=zscores[2]
                
                bind_out = bsample.loc[(zscores[1])&(bsample.loc[:,'deal_type']!=1)].index.values
                if bind_out.size!= 0:
                    ind_out.append(bind_out)
            ind_out = [item for sublist in ind_out for item in sublist]
            
            print(f'DF shape: {self.results[i].shape} - original\n')
            print(f'Deals dropped:\n {ind_out}\n')
            self.dropped_deals[settle_date] = self.results[i].loc[ind_out,:]
            self.results[i].drop(ind_out, inplace = True)
            print(f'DF shape: {self.results[i].shape} - adjusted\n')
            print(f'Generating sample for {settle_date:%d.%m.%Y} - Done!\n')
            self.data_different_dates[settle_date] = self.results[i]

        self.results = []
        
    def gen_one_date(self, settle_date):
        '''
        Generates a dictionary of pandas DataFrames. DataFrames represents the
        sample used for optimization for each date
    
        Parameters:
        -----------
            tau : value of fixed tau
            Loss: loss function, by default yield loss function
            loss_agrs : a tuple of additional arguments to loss function
            beta_init: initial  guess for beta parameters that are used 
                       as optimization starting point
            
    
        Returns:
        --------
            Returns nothing, 
            updates class data field: self.data_different_dates
    
       
        '''   
        if not hasattr(self, 'data_different_dates'):
            self.data_different_dates = {}
            
        self.data_different_dates[settle_date] = creating_sample(settle_date, 
                                                                 self.raw_data, 
                                                                 min_n_deal=CONFIG.MIN_N_DEAL, 
                                                                 time_window=CONFIG.TIME_WINDOW, 
                                                                 thresholds = self.thresholds)
        

        ind_out=[]
        for b in self.data_different_dates[settle_date].bond_maturity_type.unique().sort_values():
            bsample = self.data_different_dates[settle_date].loc[self.data_different_dates[settle_date].loc[:,'bond_maturity_type']==b]
            zscores = self.is_outlier(bsample.loc[:, ['ytm', 'span']])
            print(f'Z-score: {zscores[0]},\n {zscores[1]},\n {bsample.loc[:,"ytm"]}')
            self.data_different_dates[settle_date].loc[self.data_different_dates[settle_date].loc[:,'bond_maturity_type']==b, 'std']=zscores[2]
            bind_out = bsample.loc[(zscores[1])&(bsample.loc[:,'deal_type']!=1)].index.values
            
            if bind_out.size!= 0:
                ind_out.append(bind_out)
        ind_out = [item for sublist in ind_out for item in sublist]
        
        print(f'DF shape: {self.data_different_dates[settle_date].shape} - original\n')
        print(f'Deals dropped:\n {ind_out}\n')
        self.dropped_deals[settle_date] = self.data_different_dates[settle_date].loc[ind_out,:]
        self.data_different_dates[settle_date].drop(ind_out, inplace = True)
        print(f'DF shape: {self.data_different_dates[settle_date].shape} - adjusted\n')
        
       
        print(f'Generating sample for {settle_date:%d.%m.%Y} - Done!\n')
    
    def new_dates(self, new_end_date = None):
        
        if new_end_date == None:
            self.update_date = [self.settle_dates[-1]+1]
            self.settle_dates = self.settle_dates.union([self.settle_dates[-1]+1])
            
    
    
    def dump(self):
        
        best_betas = {}
        for date in self.settle_dates:
            idx = self.loss_res[date].loc[:, 'loss'].idxmin()
            best_betas[date] = self.loss_res[pd.to_datetime(date)].loc[idx, ['b0','b1','b2','teta']].values
        best_betas = pd.DataFrame.from_dict(best_betas, orient='index', columns = ['b0','b1','b2','teta'])
        best_betas.sort_index(inplace=True)
        
        attributes = ['several_dates', 
                      'thresholds', 
                      'start_date', 
                      'end_date', 
                      'freq', 
                      'num_workers', 
                      'inertia', 
                      'settle_dates',
                      'loss_res']
        
        params = {k:self.__getattribute__(k) for k in attributes}
                  
       
        with h5py.File('grid_data.hdf5', 'w') as f:
            g = f.create_group('curveData')
            betas = g.create_dataset('betas', data = [pickle.dumps(best_betas)])
            samples = g.create_dataset('samples', data = [pickle.dumps(self.data_different_dates)])
            dropped_deals = g.create_dataset('dropped', data = [pickle.dumps(self.dropped_deals)])
            raw_data = g.create_dataset('raw_data', data = [pickle.dumps(self.raw_data)])
            params = g.create_dataset('params', data = [pickle.dumps(params)])
            
            meta = {'save date': f'{pd.datetime.now():%Y-%m-%d %H:%M:%S}',
                    'frequency': self.freq,
                    'start_date': self.start_date,
                    'end_date':self.end_date,
                   
                    }
            g.attrs.update(meta)
        
            print('saving data:\n')
            print('-'*10)
            for m in g.attrs.keys():
                print('{}: {}'.format(m, g.attrs[m]))
            print('-'*10, '\n')
                
    def load(self):
        
        with h5py.File('grid_data.hdf5', 'r') as f:
            g = f['curveData']
            print('loading stored data:')
            print('-'*10)
            for m in g.attrs.keys():
                print('{}: {}'.format(m, g.attrs[m]))
            print('-'*10, '\n')
            best_betas = pickle.loads(g['betas'][()])
            params = pickle.loads(g['params'][()])
            samples = pickle.loads(g['samples'][()])
            dropped = pickle.loads(g['dropped'][()])
            
            
        self.previous_curve = best_betas.iloc[-1].copy()
        self.beta_init = best_betas.iloc[-1].copy()
        self.data_different_dates = samples
        self.dropped_deals = dropped
        
        print('Following parameters were used:')
        print('-'*10)
        for k,v in params.items():
#            print(f'{k}: {v}') #uncomment for diagnostics
            self.__dict__[k] = v
    
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
            loss_args = self.loss_args
            
            if not hasattr(self, 'data_different_dates'):
                self.data_different_dates = {}
                self.gen_subsets()
            
            for date, dataset in self.data_different_dates.items():
                
                l_args = [arg for arg in loss_args]

                l_args[0] = dataset
                l_args = tuple(l_args)
                
                constr = ({'type':'eq',
                           'fun': lambda x: np.array(x[0] + x[1]- np.log(1 + self.tonia_df.loc[date][0]))},)
    
                #parallelization of loop via dask multiprocessing
                values = [delayed(self.minimization_del)(tau, self.Loss, 
                          l_args, self.beta_init, constraints = constr, **kwargs) for tau in self.tau_grid]
    
                res_ = compute(*values, scheduler='processes', num_workers=self.num_workers)
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
            
            self.loss_res[date] = loss_frame
            print(f'Optimization for {date:%d.%m.%Y} - Done!\n')
        
        elif self.inertia:
            print('start')
            loss_args = self.loss_args
            
            if self.update_date != None:
                self.iter_dates = self.update_date
            else:
                self.iter_dates = self.settle_dates
            
            for settle_date in self.iter_dates:
                
                self.gen_one_date(settle_date)
                
                l_args = [arg for arg in loss_args]
    
                l_args[0] = self.data_different_dates[settle_date]
                l_args = tuple(l_args)
                
                constr = ({'type':'eq',
                           'fun': lambda x: np.array(x[0] + x[1]- np.log(1 + self.tonia_df.loc[settle_date][0]))},)
                
                print('populating distributed tasks')
                #parallelization of loop via dask multiprocessing
                values = [delayed(self.minimization_del)(tau, self.Loss, 
                          l_args, self.beta_init, constraints = constr, **kwargs) for tau in self.tau_grid]
                
                print('start minimizing')
                res_ = compute(*values, scheduler='processes', num_workers=self.num_workers)
                
                #putting betas and Loss value in Pandas DataFrame
                loss_frame = pd.DataFrame([], columns=['b0', 'b1', 'b2', 'teta', 'loss'])
                loss_frame['b0'] = [res.x[0] for res in res_]
                loss_frame['b1'] = [res.x[1] for res in res_]
                loss_frame['b2'] = [res.x[2] for res in res_]
                loss_frame['teta'] = [t for t in self.tau_grid]
                loss_frame['loss'] = [res.fun for res in res_]
        
                self.loss_res[settle_date] = loss_frame
                self.beta_best = loss_frame.loc[loss_frame['loss'].idxmin(), :].values[:-1]
                self.beta_init = self.beta_best[:-1].copy()
                self.previous_curve = self.beta_best.copy()
                print(f'Optimization for {settle_date:%d.%m.%Y} - Done!\nBeta best: {self.beta_best}\nPrevious beta set to {self.previous_curve}\n')
                
            self.update_date = None          
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
    def fit(self, return_frame=False, **kwargs):
        if use_one_worker:
            print('Multiprocessing is not enabled as dask is not installed\n'
                  'Install dask to enbale multiprocessing')
            self.num_workers = 1
        else:
            self.num_workers = num_workers
        self.loss_frame = self.loss_grid(**kwargs)
        #loss_frame = self.filter_frame(loss_frame)
        self.beta_best = self.loss_frame.loc[self.loss_frame['loss'].argmin(), :].values[:-1]
	best_betas = {}
        for date in self.settle_dates:
            idx = self.loss_res[date].loc[:, 'loss'].idxmin()
            best_betas[date] = self.loss_res[pd.to_datetime(date)].loc[idx, ['b0','b1','b2','teta']].values
	self.best_betas = best_betas
		  
        if return_frame:
            return self.beta_best, self.loss_frame
        else:
            return self.beta_best
