import os
import sys
import warnings
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('..'))
try:
    import matplotlib.pyplot as plt
    from dataextraction import draw
    to_draw_graphs = True
    plt.style.use('ggplot')
except ImportError as e:
    to_draw_graphs = False
    
import CONFIG
from datapreparation.preparation_functions import processing_data, read_download_preprocessed_data
from datapreparation.adaptive_sampling import creating_sample
from optimizator.simultaneous_min import iter_minimizer
from estimation_ytm.estimation_ytm import new_ytm, newton_estimation, filtering_ytm
from error_measures import MAE_YTM, mean_absolute_error
from Loss import yield_Loss, price_Loss, naive_yield_Loss
from payments_calendar import download_calendar
from ns_func import D    
from weight_scheme import weight
    
PATH = os.path.join('..', 'extracted_data')
datasets_path = os.path.join('..', 'datasets')
clean_data_path = os.path.join(datasets_path, 'clean_data.hdf')
calendar_data_path = os.path.join(datasets_path, 'coupons_data.hdf')
save_data = False
warnings.simplefilter('ignore')

### Initialization
path = os.path.join(datasets_path, 'bonds.xls')
df = pd.read_excel(path, skiprows=2).rename(columns=CONFIG.NAME_MASK)


### data mungling
if save_data:
    clean_data = processing_data(df, 
                  mask_face_value=CONFIG.MASK_FACE_VALUE, mask_base_time=CONFIG.MASK_BASE_TIME,
                  needed_bonds=CONFIG.INSTRUMENTS, use_otc=CONFIG.USE_OTC, deal_market=CONFIG.DEAL_MARKET,
                  notes_in_otc=CONFIG.NOTES_IN_OTC, maturity_filter=CONFIG.MATURITY_FILTER, 
                  specific_deals=CONFIG.SPECIFIC_DEALS)
    
    #calendar payments data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)
    
    #Estimating correct yield for data
    clean_data = (clean_data.pipe(new_ytm, coupons_cf, streak_data)
                            .pipe(filtering_ytm, max_yield=CONFIG.MAX_YIELD, 
                                  min_yield=CONFIG.MIN_YIELD))
    
    clean_data['bond_symb'] = clean_data.index.get_level_values(1).str.extract(r'([A-Z]+)', expand=False)
    clean_data = read_download_preprocessed_data(save_data, clean_data_path=clean_data_path, clean_data=clean_data)
else:
    clean_data = read_download_preprocessed_data(save_data, clean_data_path=clean_data_path,)
    clean_data = clean_data.loc[clean_data.loc[:,'ytm']!=0]
    #Coupon Data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)
    



print('starting to read filtered data for {} settle date'.format(CONFIG.SETTLE_DATE)) 
###Creating sample
filtered_data = creating_sample(CONFIG.SETTLE_DATE, clean_data, min_n_deal=CONFIG.MIN_N_DEAL, 
                                time_window=CONFIG.TIME_WINDOW, thresholds = False)
print('filtered data shape {}'.format(filtered_data.shape[0]))

###Setting Loss arguments and optimization paramters 

#Initial guess vector(for optimization)
x0 = [0.09, -0.01, 0, 1.5]

#Parameters constraints
constr = ({'type':'ineq',
           'fun': lambda x: np.array(x[0] + x[1])})

#Longest maturity of deals in sample (in years)
max_deal_span = (filtered_data.span / 365).round().max()

#Parameters bounds for constraint optimization
bounds = ((0, 1), (None, None), (None, None), (CONFIG.TETA_MIN, CONFIG.TETA_MAX))

#Maturity limit for Zero-curve plot
longest_maturity_year = max([max_deal_span, 20])
theor_maturities = np.linspace(0.001, longest_maturity_year, 10000)
options = {'maxiter': 500, 'eps': 9e-5, 'disp': True}

#Tuple of arguments for loss function            
loss_args = (filtered_data, coupons_cf, streak_data, CONFIG.RHO, CONFIG.WEIGHT_SCHEME)

#defining loss
loss = yield_Loss

filtered_data['weight'] = weight([1, 1, 1, 1], filtered_data, CONFIG.WEIGHT_SCHEME)

###### OPTIMIZATION
print('start optimization\n')
res_ = iter_minimizer(Loss=loss, beta_init=x0, 
                loss_args=loss_args, method='SLSQP',  
                bounds=bounds,
                constraints=constr,
                max_deal_span=max_deal_span, options=options)
beta_best = res_.x
print('end optimization\n')

### Showing results of work

draw(beta_best, filtered_data, theor_maturities, f'{CONFIG.SETTLE_DATE:%d.%m.%Y}', 
     longest_maturity_year, draw_points=True, 
     weight_scheme=CONFIG.WEIGHT_SCHEME, label='Spot rate curve',
     alpha=0.8, shift=True)
     
plt.ylim(0, filtered_data.ytm.max() + 0.01)
plt.savefig(f'zero_curve_{date:%d.%m.%Y}.png', dpi=400);
plt.close()
