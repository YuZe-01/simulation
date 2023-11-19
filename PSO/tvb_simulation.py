from tvb.simulator.lab import *
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time as t
import multiprocessing
from sklearn.metrics import r2_score
from r2_ import r2_cal
import random
import logging
import subprocess

def RWWsimulation(path, process_id=None, result_queue=None):
    logging.disable(logging.CRITICAL)  # 禁用所有日志输出
    data = np.loadtxt(path)
    start_time = t.time()

    WWM = models.ReducedWongWangExcInh(G = data[0], w_p = data[1], lamda = data[2])

    white_matter = connectivity.Connectivity.from_file('connectivity_192.zip')
    white_matter.speed = np.array([4.0])
    white_matter_coupling = coupling.Difference(a=np.array([0.014]))

    rm_f_name = 'regionMapping_16k_76.txt'
    rm = RegionMapping.from_file(rm_f_name)
    sensorsEEG = SensorsEEG.from_file('eeg_unitvector_62.txt.bz2')
    prEEG = ProjectionSurfaceEEG.from_file('projection_eeg_62_surface_16k.mat', matlab_data_name="ProjectionMatrix")

    heunint = integrators.HeunStochastic(
        dt=2**-4,
        noise=noise.Additive(nsig=np.array([2 ** -5, ]))
    )

    fsamp = 1e3/500 # 500 Hz

    monitor_MEG=monitors.MEG.from_file(rm_f_name=rm_f_name)
    monitor_MEG.period=fsamp
    mons = (
        monitors.EEG(sensors=sensorsEEG, projection=prEEG, region_mapping=rm, period=fsamp),
        monitors.ProgressLogger(period=100.0),
    )

    local_coupling_strength = np.array([2 ** -10])
    default_cortex = Cortex.from_file(region_mapping_file='regionMapping_16k_76.txt')
    default_cortex.region_mapping_data.connectivity = white_matter
    default_cortex.coupling_strength = local_coupling_strength

    sim = simulator.Simulator(
        model=WWM,
        connectivity=white_matter,
        coupling=white_matter_coupling,
        integrator=heunint,
        monitors=mons,
        surface=default_cortex,
        simulation_length=4.
        # simulation_length=list['sim_len'] # unit is ms
    )
    sim.configure()

    sim_file = open(f"./log/test{random.randint(1, 1000)}.txt", "a")
    sim_file.write(str(random.randint(1, 1000))+"\n")
    sim_file.close()

    eeg, _ = sim.run()

    time, data = eeg

    r2 = - 2.2

    '''
    try:
        default_stdout = sys.stdout
        sys.stdout = open('./log/log.txt', 'a')  # 将标准输出重定向到文件
        # r2 = r2_cal(data)
        r2 = -2.3
        sys.stdout = default_stdout
    except Exception as e:
        print(f"写入文件出错：{str(e)}")
    '''
    end_time = t.time()

    duration = end_time - start_time

#     if process_id != None:
#         pass
#     else:
#         pass
    
    if result_queue != None:
        # result_queue.put(r2)
        return r2
    else:
        final_result = []
        final_result.append(r2)
        #print(r2)
        return final_result
    
def func(i):
    print("func\n")
    path = open(f"./bash/{i}.sh",'w')
    path.write(f"#!/bin/bash\n")
    path.write("#SBATCH --job-name=lyz_subprocess\n")
    path.write("#SBATCH --ntasks=1\n")
    path.write("#SBATCH --cpus-per-task=1\n")
    path.write(f"#SBATCH --output=./slurm/out_{i}.out\n")

    python_path = "\n/public/home/ynhang/yuze/TVB_Distribution/tvb_data/bin/python"
    script_path = " /public/home/ynhang/yuze/code/multitask/simulation.py"
    order = python_path + script_path + f" ./test/test_{i}.txt" + f' {i}'
    path.write(order)
    path.close()

    subprocess.run(f"sbatch ./bash/{i}.sh", shell=True)    

def multiprocess(param_list):
    process_num_max = 200
    r2_result = []
    process_num = len(param_list)
    
    if process_num > process_num_max:
        for j in range(0, process_num//process_num_max + 1, 1):
            for i in range(process_num_max if j != process_num//process_num_max else process_num % process_num_max):
                np.savetxt(f'./test/test_{i}.txt', (param_list[i+j*process_num_max]['G'],
                                                    param_list[i+j*process_num_max]['w_p'],
                                                    param_list[i+j*process_num_max]['lamda'],
                                                    param_list[i+j*process_num_max]['I_o']))
                func(i)
                print(f"Started process {i+j*process_num_max}")
            
            while(1):
                t.sleep(5)
                num = subprocess.run("squeue -o '%T' --name=lyz_subprocess | grep RUNNING | wc -l", shell=True)
                if num == 0:
                    break
    else:
        for i in range(process_num):
            np.savetxt(f'./test/test_{i}.txt', (param_list[i]['G'],param_list[i]['w_p'],param_list[i]['lamda'],param_list[i]['I_o']))
            func(i)
            print(f"Started process {i}")
        
        while(1):
            t.sleep(5)
            res = subprocess.run("squeue -o '%T' --name=lyz_subprocess | grep RUNNING | wc -l", shell=True, stdout=subprocess.PIPE)
            num = int(res.stdout.strip())
            if num == 0:
                break               
    
    for i in range(process_num):
        path = open(f'./return/{i}.txt', 'r')
        r2_result.append(float(path.readline()))
    
    return r2_result

if __name__ == "__main__":
    pass
