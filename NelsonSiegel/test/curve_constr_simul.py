import os, sys
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
    
PATH = os.path.join('..', 'extracted_data')
datasets_path = os.path.join('..', 'datasets')
clean_data_path = os.path.join(datasets_path, 'clean_data.hdf')
calendar_data_path = os.path.join(datasets_path, 'coupons_data.hdf')
save_data = True


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
    #Coupon Data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)
    
    
print('starting to read filtered data for {} settle date'.format(CONFIG.SETTLE_DATE)) 
###Creating sample
filtered_data = creating_sample(CONFIG.SETTLE_DATE, clean_data, min_n_deal=CONFIG.MIN_N_DEAL, 
                                time_window=CONFIG.TIME_WINDOW)
print('filtered data shape {}'.format(filtered_data.shape[0]))


###Setting Loss arguments and optimization paramters 
#Initial guess vector(for optimization)
x0 = [0.09, -0.01, 0, 1.5]
#Parameters constraints
constr = ({'type':'ineq',
           'fun': lambda x: np.array(x[0] + x[1])})
#Longest matuiry year of deals in data
max_deal_span = (filtered_data.span / 365).round().max()
#Parameters bounds for constraint optimization
bounds = ((0, 1), (None, None), (None, None), (CONFIG.TETA_MIN, CONFIG.TETA_MAX))
print(bounds)
#Maturity limit for Zero-curve plot
longest_maturity_year = max([max_deal_span, 20])
theor_maturities = np.linspace(0.001, longest_maturity_year, 10000)
options = {'maxiter': 150, 'eps': 9e-5, 'disp': True}
#Tuple of arguments for loss function            
loss_args = (filtered_data, coupons_cf, streak_data, CONFIG.RHO, CONFIG.WEIGHT_SCHEME)

#defining loss -- Crucial
loss = yield_Loss


print('start optimization\n')
###### OPTIMIZATION
res_ = iter_minimizer(Loss=loss, beta_init=x0, 
                loss_args=loss_args, method='SLSQP',  bounds=bounds,
                #constraints=constr,
                max_deal_span=max_deal_span, options=options)
beta_best = res_.x
print('end optimization\n')


### Showing results of work
print('final loss is {loss} and yield MAE is {mae}'.format(loss=res_.fun, 
      mae=MAE_YTM(beta_best, filtered_data, coupons_cf=coupons_cf, streak_data=streak_data)))
ind = filtered_data.index
price_hat = (D(streak_data[ind], beta_best) * coupons_cf[ind]).sum().values
ytm_hat = np.array([newton_estimation(filtered_data.iloc[i], price_hat[i], 
                    coupons_cf, streak_data, maxiter=200) for i in range(filtered_data.shape[0])])
print('price MAE is {mae}'.format(mae=mean_absolute_error(filtered_data.stand_price, price_hat)))

##distibution of prices
fig, ax = plt.subplots(1, 3, figsize=(18, 8))
filtered_data['stand_price'].plot(kind='kde', ax=ax[0], title='True Price', c='r')
filtered_data['price_hat'] = price_hat
filtered_data['price_hat'].plot(kind='kde', ax=ax[1], title='Estimated Price', c='g')
diagonal_price = np.linspace(filtered_data['stand_price'].min(), filtered_data['stand_price'].max(), 1000)
ax[2].plot(diagonal_price, diagonal_price)
ax[2].plot(filtered_data['stand_price'], price_hat, ls='', marker='o')
ax[2].set(ylabel='Estimated Price', xlabel='True Price')

##distibution of yields
fig, ax = plt.subplots(1, 3, figsize=(18, 8))
filtered_data['ytm'].plot(kind='kde', ax=ax[0], title='True Yield', c='r')
filtered_data['ytm_hat'] = ytm_hat
filtered_data['ytm_hat'].plot(kind='kde', ax=ax[1], title='Estimated Yield', c='g')
diagonal_ytm = np.linspace(filtered_data['ytm'].min(), filtered_data['ytm'].max(), 1000)
ax[2].plot(diagonal_ytm, diagonal_ytm)
ax[2].plot(filtered_data['ytm'], ytm_hat, ls='', marker='o')
ax[2].set(ylabel='Estimated Yield', xlabel='True Yield')

##plotting and saving Spot rate curve
draw(beta_best, filtered_data, theor_maturities, CONFIG.SETTLE_DATE, 
     longest_maturity_year, draw_points=True, 
     weight_scheme=CONFIG.WEIGHT_SCHEME, label='Spot rate curve',
     alpha=0.8, shift=True)
plt.ylim(0.05, filtered_data.ytm.max() + 0.01)
plt.savefig(os.path.join(PATH, f'zero_curve_simul_{CONFIG.SETTLE_DATE}_{loss.__name__}.png'), dpi=400);