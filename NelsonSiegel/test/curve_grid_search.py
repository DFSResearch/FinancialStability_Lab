from dask import compute, delayed
import pickle
import os, sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
   
sys.path.append(os.path.abspath('..'))
try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as c
    from dataextraction import draw
    
    plt.style.use('ggplot')
    to_draw_graphs = True
except ImportError as e:
    to_draw_graphs = False
    
import CONFIG
from optimizator.gridsearch import grid_search
from datapreparation.preparation_functions import processing_data, read_download_preprocessed_data
from datapreparation.adaptive_sampling import creating_sample
from estimation_ytm.estimation_ytm import new_ytm, newton_estimation, filtering_ytm
from error_measures import MAE_YTM, mean_absolute_error
from Loss import yield_Loss, price_Loss, naive_yield_Loss
from payments_calendar import download_calendar
from ns_func import D, Z, par_yield
from weight_scheme import weight


    
plt.style.use('seaborn')
PATH = os.path.join('..', 'extracted_data')
datasets_path = os.path.join('..', 'datasets')
clean_data_path = os.path.join(datasets_path, 'clean_data.hdf')
calendar_data_path = os.path.join(datasets_path, 'coupons_data.hdf')
save_data = False

### Initialization
#read initial raw data
path = os.path.join(datasets_path, 'bonds.xls')
df = pd.read_excel(path, skiprows=2).rename(columns=CONFIG.NAME_MASK)

#read TONIA data from kase.kz
tonia_path = os.path.join(datasets_path, 'TONIA.xls')
tonia = pd.read_excel(tonia_path, skiprows=1, usecols = [0,4])
tonia.iloc[:,1] = tonia.iloc[:,1]/100
tonia.iloc[:,0] = pd.to_datetime(tonia.iloc[:,0], format = '%d.%m.%y')
tonia.columns = ['Date', 'Close']
tonia.set_index('Date', inplace = True)
tonia_df = pd.DataFrame({'Date':pd.date_range(start='2010-07-15', end='2019-07-15', freq = 'D')})
tonia_df.set_index('Date', inplace = True)
tonia_df = tonia_df.join(tonia)
tonia_df.fillna(method ='ffill', inplace = True)

### data mungling
if save_data:
    clean_data = processing_data(df, 
                                 mask_face_value=CONFIG.MASK_FACE_VALUE, 
                                 mask_base_time=CONFIG.MASK_BASE_TIME,
                                 needed_bonds=CONFIG.INSTRUMENTS, 
                                 use_otc=CONFIG.USE_OTC, 
                                 deal_market=CONFIG.DEAL_MARKET,
                                 notes_in_otc=CONFIG.NOTES_IN_OTC, 
                                 maturity_filter=CONFIG.MATURITY_FILTER, 
                                 specific_deals=CONFIG.SPECIFIC_DEALS)
    
    #calendar payments data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, 
                                                hdf_coupons_path=calendar_data_path)
    #Estimating correct yield for data
    clean_data = (clean_data.pipe(new_ytm, coupons_cf, streak_data)
                            .pipe(filtering_ytm, 
                                  max_yield=CONFIG.MAX_YIELD, 
                                  min_yield=CONFIG.MIN_YIELD))
    
    
    clean_data['bond_symb'] = clean_data.index.get_level_values(1).str.extract(r'([A-Z]+)', expand=False)
    clean_data = read_download_preprocessed_data(save_data, clean_data_path=clean_data_path, clean_data=clean_data)
else:
    clean_data = read_download_preprocessed_data(save_data, 
                                                 clean_data_path=clean_data_path,)
    #Coupon Data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, 
                                                hdf_coupons_path=calendar_data_path)
    
print('starting to read fitered data for {} settle date'.format(CONFIG.SETTLE_DATE))  

###Creating sample
filtered_data = creating_sample('2017-08-14', 
                                clean_data, 
                                min_n_deal=CONFIG.MIN_N_DEAL, 
                                time_window=CONFIG.TIME_WINDOW)

print('filtered data shape {}'.format(filtered_data.shape[0]))

###Setting Loss arguments and optimization paramters 
#Initial guess vector(for optimization)
x0 = [0.09, -0.01, 0]

thresholds = [0, 190 ,370, 5*365, np.inf]

##define tau grid
tau_grid = np.append(np.arange(0.076, 2, 0.02), np.arange(2, CONFIG.TETA_MAX, 0.5))

#Parameters constraints
constr = ({'type':'ineq',
           'fun': lambda x: np.array(x[0] + x[1])})
           
#Longest mattuiry year of deals in data
max_deal_span = (filtered_data.span / 365).round().max()
#Parameters bounds for constraint optimization
bounds = ((0.05, 0.2), (-1, 1), (-1, 1))
#Maturity limit for Zero-curve plot
longest_maturity_year = max([max_deal_span, 20])
theor_maturities = np.linspace(0.001, longest_maturity_year, 10000)
options = {'maxiter': 150, 'ftol':1e-7,'eps': 1e-4, 'disp': True}
#Tuple of arguments for loss function            
loss_args = (filtered_data,
             coupons_cf, 
             streak_data, 
             CONFIG.RHO, 
             CONFIG.WEIGHT_SCHEME)


#defining loss -- Crucial --
loss = yield_Loss

START_DATE = '2018-01-01'
END_DATE = '2019-07-15'
FREQ = 'W-MON'#change for 'B' for business days


print('start optimization\n')
###### OPTIMIZATION
grid = grid_search(tau_grid=tau_grid, 
                   Loss=loss, 
                   loss_args=loss_args, 
                   beta_init=x0, 
                   toniaDF = tonia_df,
                   start_date = START_DATE, 
                   end_date = END_DATE,
                   thresholds = thresholds,
                   freq = FREQ, 
                   several_dates = False,
                   clean_data = clean_data,
                   inertia = True,
                   num_workers=16)

beta_best, loss_frame = grid.fit(options=options, 
                                 return_frame=True, 
                                 method='SLSQP', 
#                                 num_workers=16, 
                                 bounds=bounds,)
print('end optimization\n')
