import tvb_simulation
import time as t
import copy
import numpy as np

class stepwise_fit(object):
    
  # Params is a dictionary of lists
  # Order is an ordered list of keys
  # initvals is a dict of initial values
    
  def __init__(self, param_ranges, step_order, initvals, sim_len=500):
    
    # Make some empty attributes to fill
    
    for v in ['param_ranges', 'step_order', 'initvals']:
      exec('self.%s = %s' %(v,v))
        
    self.start_time = t.time()
    self.end_time = 0
    self.run_time = 0
    
    self.finalvals = {}
    self.step_history = []
    self.completed = False
   
    self.initargs = copy.copy(initvals)
    self.initargs['sim_len'] = sim_len
    
    print("stepwise_fit _init_ over\n")
    
  def run(self):
    for step in self.step_order:
      print(f"stepwise_fit run {step}\n")
      # Define parameter set to run for this step
      params = self.set_step_params(step)
      print(params)
        
      # Run sims
      print(f"stepwise_fit run {step} multiprocess\n")
      res = tvb_simulation.multiprocess(params)
      print(f"{step}'s res = {res}\n")
        
      # Identify best fit and log
      idx = np.nanargmax(res)
      self.finalvals[step] = params[idx][step]
    
      # Keep some other info
      self.step_history.append({'varying': step, 'params': params, 'res': res})
      print(self.step_history)

    print("optimization finish\n")
    self.finalfit = res[idx]
    
    # Re-run the final parameter set
    print(params[idx]['G'], params[idx]['w_p'], params[idx]['lamda'], params[idx]['sim_len'])
    result = tvb_simulation.RWWsimulation(params[idx]['G'], 
                  params[idx]['w_p'], params[idx]['lamda'], params[idx]['sim_len'])
    
    self.end_time = t.time()
    self.run_time = self.end_time - self.start_time
    
    self.completed = True
    
    print(self._repr_html_())
    
    print("total finish")
    
  def set_step_params(self, step):
    # Specify a list of arguments for corrSCFC_4params for the current step
    params = copy.copy(self.initargs)   # Start with the initial values
    params.update(self.finalvals)  # Add in final values obtained so far
    
    paramslist = []
    for varval in self.param_ranges[step]:
      ps = copy.copy(params)
      ps[step] = varval
      paramslist.append(ps)
      
    return paramslist

  def _repr_html_(self):
    if self.completed:
      print(self.run_time, self.finalfit, self.finalvals)
      msg = 'TVB Stepwise Model Fit: \n'
      msg += f'completed in: {self.run_time} \n'
      msg += f'best fit:     {self.finalfit} \n'
      msg += f'best params:   {self.finalvals} \n'
    else:
      msg =  'not yet run'
          
    return msg    
    
    
  def print_summary(self):
    if self.completed: 
      print('completed in %.5f minutes' %self.run_time)
      print('best fit: %s' %self.finalfit)
    else:
      print('not yet run')
        
if __name__ == '__main__':
    G_bounds = [-5.0,5.0]
    w_p_bounds = [-5.0,5.0]
    lamda_bounds = [-5.0,5.0]

    G_init = -2.0
    w_p_init = 1.0
    lamda_init = 1.0
    _G = np.linspace(G_bounds[0], G_bounds[1],11)
    _w_p = np.linspace(w_p_bounds[0],w_p_bounds[1],11)
    _lamda = np.linspace(lamda_bounds[0],lamda_bounds[1],11)

    param_ranges = {'G': _G, 'w_p': _w_p, 'lamda': _lamda}
    order = ['G', 'w_p', 'lamda']
    inits = {'G': G_init, 'w_p': w_p_init, 'lamda': lamda_init}
    
    sf = stepwise_fit(param_ranges, order, inits, sim_len=10.0)
    sf.run()
    d(sf)