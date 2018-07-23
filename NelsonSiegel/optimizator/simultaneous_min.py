import numpy as np
from scipy.optimize import minimize
from ns_func import Z

# Defining optimizators
def iter_minimizer(Loss, beta_init, loss_args, bounds, max_deal_span, **kwargs):
    i = 0.1
    def callback_print(xk):
        loss = Loss(xk, df=loss_args[0], coupons_cf=loss_args[1], 
                    streak_data=loss_args[2], weight_scheme=loss_args[4])
        print(xk, loss)
        
    while (Z(np.linspace(0.001, 30, 1000), beta_init)).min() < 0 or i == 0.1:
        res = minimize(fun=Loss, x0=beta_init, args=loss_args, bounds=bounds, 
                       callback=callback_print, **kwargs) 
        beta_init = res.x
        bounds = ((0, 1), (None, None), (None, None), 
                  (beta_init[-1] + i, 0.05 * max_deal_span))
        i += 0.1
    return res

