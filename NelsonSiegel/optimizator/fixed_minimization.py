import numpy as np
from scipy.optimize import minimize
from ns_func import Z

# Defining optimizators
def fixed_tau_minimizer(Loss, beta_init, loss_args, bounds, max_deal_span, **kwargs):
    i = 0.1
    beta = beta_init.copy()
    beta.append(loss_args[5])
    def callback_print(xk):
        loss = Loss(xk, df=loss_args[0], coupons_cf=loss_args[1],
                    streak_data=loss_args[2], weight_scheme=loss_args[4],
                    tau=loss_args[5])
        print(xk, loss)
    #optimization goes in loop until spot rate function will not be bigger
    #than 0 at each point;
    #Z constructs from 0.001 year to 30 year (no bonds with tenor > 30 year)

    while (Z(np.linspace(0.001, 30, 1000), beta)).min() < 0 or i == 0.1:
        res = minimize(fun=Loss, x0=beta_init, args=loss_args, bounds=bounds,
                       callback=callback_print, **kwargs)
        beta_init = res.x
        #if Z < 0 than tau will incremeantly increase until Z >= 0
        #from tuple to list and back
        i += 1
        loss_args = list(loss_args)
        loss_args[5] += i
        loss_args = tuple(loss_args)
        #beta change in order to asses new curve
        beta = beta_init.copy()
        beta = np.append(beta, loss_args[5])
    return res