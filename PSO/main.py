import tvb_simulation
import numpy as np
import multiprocessing
import time as t
import random
# from optimization_strategy import stepwise_fit
from PSO import PSO_model
import sys

def step_wise_function():
    G_bounds = [-100.0,100.0]
    w_p_bounds = [-100.0,100.0]
    lamda_bounds = [-100.0,100.0]

    G_init = -2.0
    w_p_init = 1.0
    lamda_init = 1.0
    
    _G = np.linspace(G_bounds[0], G_bounds[1],5)
    _w_p = np.linspace(w_p_bounds[0],w_p_bounds[1],5)
    _lamda = np.linspace(lamda_bounds[0],lamda_bounds[1],5)

    param_ranges = {'G': _G, 'w_p': _w_p, 'lamda': _lamda}
    order = ['G', 'w_p', 'lamda']
    inits = {'G': G_init, 'w_p': w_p_init, 'lamda': lamda_init}
    
    # sf = stepwise_fit(param_ranges, order, inits, sim_len=10.0)
    # sf.run()

def PSO_function():
    # w,c1,c2,r1,r2,N,D,M参数初始化
    w=[0.4, 2]
    c1=c2=2#一般设置为2
    N=150
    D=4
    M=40
    tip = f"test N={N}, M={M}, psd_range=(1, 100), channel=All\n" + \
            "add I_o to invert the post-scaling of electrode data to the scaling during fitting.\n" + \
            "and add the limitation of velocity and parameters, referring to \\tvb\\simulator\\models\\wong_wang_exc_inh.py\n" + \
            "also remove the r1 and r2, and set them as random value at the beginning of each iteration.\n"
    pso_object=PSO_model(w,c1,c2,N,D,M,tip)#设置初始权值
    pso_object.init_pop()
    pso_object.update()
    
if __name__ == "__main__":
    print("begin")
    PSO_function()
    print("finish")
