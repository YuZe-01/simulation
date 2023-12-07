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

def func(i):
    print("func\n")
    path = open(f"../bash/{i}.sh",'w')
    path.write(f"#!/bin/bash\n")
    path.write("#SBATCH --job-name=lyz_subprocess\n")
    path.write("#SBATCH --ntasks=1\n")
    path.write("#SBATCH --cpus-per-task=1\n")
    path.write(f"#SBATCH --output=../slurm/out_{i}.out\n")

    python_path = "\n/public/home/ynhang/yuze/TVB_Distribution/tvb_data/bin/python"
    script_path = " /public/home/ynhang/yuze/code/multitask/PSO/simulation.py"
    order = python_path + script_path + f" ../test/test_{i}.txt" + f' {i}'
    path.write(order)
    path.close()

    subprocess.run(f"sbatch ../bash/{i}.sh", shell=True)    

def multiprocess(param_list):
    process_num_max = 200
    r2_result = []
    process_num = len(param_list)
    
    if process_num > process_num_max:
        for j in range(0, process_num//process_num_max + 1, 1):
            for i in range(process_num_max if j != process_num//process_num_max else process_num % process_num_max):
                np.savetxt(f'../test/test_{i}.txt', (param_list[i+j*process_num_max]['G'],
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
            np.savetxt(f'../test/test_{i}.txt', (param_list[i]['G'],param_list[i]['w_p'],param_list[i]['lamda'],param_list[i]['I_o']))
            func(i)
            print(f"Started process {i}")
        
        while(1):
            t.sleep(5)
            res = subprocess.run("squeue -o '%T' --name=lyz_subprocess | grep RUNNING | wc -l", shell=True, stdout=subprocess.PIPE)
            num = int(res.stdout.strip())
            if num == 0:
                break               
    
    for i in range(process_num):
        path = open(f'../return/{i}.txt', 'r')
        r2_result.append(float(path.readline()))
    
    return r2_result

if __name__ == "__main__":
    pass
