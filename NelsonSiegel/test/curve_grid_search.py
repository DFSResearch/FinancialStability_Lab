import os, sys
import pandas as pd
import numpy as np
   
sys.path.append(os.path.abspath('..'))
try:
    import matplotlib.pyplot as plt
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
from ns_func import D


    
plt.style.use('seaborn')
PATH = os.path.join('..', 'extracted_data')
datasets_path = os.path.join('..', 'datasets')
clean_data_path = os.path.join(datasets_path, 'clean_data.hdf')
calendar_data_path = os.path.join(datasets_path, 'coupons_data.hdf')
save_data = False
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
    
print('starting to read fitered data for {} settle date'.format(CONFIG.SETTLE_DATE))  
###Creating sample
filtered_data = creating_sample(CONFIG.SETTLE_DATE, clean_data, min_n_deal=CONFIG.MIN_N_DEAL, 
                                time_window=CONFIG.TIME_WINDOW)
print('filtered data shape {}'.format(filtered_data.shape[0]))

###Setting Loss arguments and optimization paramters 
#Initial guess vector(for optimization)
x0 = [0.09, -0.01, 0]

##tau grid ---Essential--- algorithm loops trough this values ---
tau_grid = np.append(np.arange(0.2, 2, 0.25), np.arange(2, CONFIG.TETA_MAX, 0.5))
#Parameters constraints
constr = ({'type':'ineq',
           'fun': lambda x: np.array(x[0] + x[1])})
#Longest matuiry year of deals in data
max_deal_span = (filtered_data.span / 365).round().max()
#Parameters bounds for constraint optimization
bounds = ((0, 1), (None, None), (None, None))
#Maturity limit for Zero-curve plot
longest_maturity_year = max([max_deal_span, 20])
theor_maturities = np.linspace(0.001, longest_maturity_year, 10000)
options = {'maxiter': 150, 'disp': True}
#Tuple of arguments for loss function            
loss_args = (filtered_data, coupons_cf, streak_data, CONFIG.RHO, CONFIG.WEIGHT_SCHEME)


#defining loss -- Crucial --
loss = price_Loss

print('start optimization\n')
###### OPTIMIZATION
grid = grid_search(tau_grid=tau_grid, Loss=loss, 
                   loss_args=loss_args, beta_init=x0)
beta_best, loss_frame = grid.fit(options=options, return_frame=True, 
         method='SLSQP', num_workers=1, bounds=bounds, constraints=constr)
print('end optimization\n')


### Showing results of work
#progression of loss for different teta
loss_frame[['loss', 'teta']].plot(x='teta', y='loss', figsize=(15, 10))
plt.savefig('loss_teta_{}.png'.format(CONFIG.SETTLE_DATE))

#curves for all teta
fig, ax = plt.subplots(figsize=(15, 10))
linestyles = ['-', '--', '-.']
np.random.seed(111)
ls = np.random.choice(linestyles, size=loss_frame.shape[0])
for i, beta_ in enumerate(loss_frame.iloc[:, :-1].values):
    ls_ = ls[i]
    if (beta_ == beta_best).all():
        draw_points = True
        linewidth = 2
    else:
        draw_points = False
        linewidth = 1
    draw(beta_, filtered_data, theor_maturities,  title_date=CONFIG.SETTLE_DATE, 
         longest_maturity_year=longest_maturity_year, draw_points=draw_points,
         weight_scheme=CONFIG.WEIGHT_SCHEME, ax=ax, ls=ls_, linewidth=linewidth, shift=True)
ax.set_ylim(filtered_data.ytm.min() - 0.01, filtered_data['ytm'].max() + 0.01)
ax.legend(loc='upper right')
plt.savefig(f'spot_path_{CONFIG.WEIGHT_SCHEME}_{CONFIG.SETTLE_DATE}_{loss.__name__}.png', dpi=400);
 
#numerical loss
print('final loss is {loss} and yield MAE is {mae}'.format(loss=loss_frame['loss'].min(), 
      mae=MAE_YTM(beta_best, filtered_data, coupons_cf=coupons_cf, streak_data=streak_data)))
ind = filtered_data.index
price_hat = (D(streak_data[ind], beta_best) * coupons_cf[ind]).sum().values
ytm_hat = np.array([newton_estimation(filtered_data.iloc[i], price_hat[i], coupons_cf, streak_data, maxiter=200) 
                                                                          for i in range(filtered_data.shape[0])])
print('price MAE is {mae}'.format(mae=mean_absolute_error(filtered_data.stand_price, price_hat)))


##plotting and saving Spot rate curve
draw(beta_best, filtered_data, theor_maturities, CONFIG.SETTLE_DATE, 
     longest_maturity_year, draw_points=True, 
     weight_scheme=CONFIG.WEIGHT_SCHEME, label='Spot rate curve',
     alpha=0.8, shift=True)
plt.ylim(0, filtered_data.ytm.max() + 0.01)
plt.savefig(os.path.join(PATH, f'zero_curve_grid_{CONFIG.SETTLE_DATE}_{loss.__name__}.png'), dpi=400);
