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

def RWWsimulation(G, w_p, lamda, simulation_time, process_id=None, result_queue=None):

#         print('simulation begin')

    start_time = t.time()

    WWM = models.ReducedWongWangExcInh(G = np.array(G), w_p = np.array(w_p), lamda = np.array(lamda))

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
        simulation_length=simulation_time # unit is ms
    )
    sim.configure()
    eeg, _ = sim.run()

    time, data = eeg
    
    try:
        default_stdout = sys.stdout
        sys.stdout = sys.stdout = open('./log/log.txt', 'a')  # 将标准输出重定向到文件
        r2 = r2_cal(data)
        sys.stdout = default_stdout
    except Exception as e:
        print(f"写入文件出错：{str(e)}")

    end_time = t.time()

    duration = end_time - start_time

#     if process_id != None:
#         pass
#     else:
#         pass

    if result_queue != None:
        result_queue.put(r2)
        sys.stdout = default_stdout
        return r2
    else:
        final_result = []
        final_result.append(r2)
        sys.stdout = default_stdout
        return final_result
    
def multiprocess(param_list):
    # 创建一个队列，用于存放结果
    result_queue = multiprocessing.Queue()
    
    G = np.ones(192)
    w_p = np.ones(192)
    lamda = np.zeros(192)
    simulation_time = 1000.0
    process_num_max = 1
    
    process_num = len(param_list)
    
    start_time = t.time()
    
    if process_num > process_num_max:
        for j in range(0, process_num//process_num_max + 1, 1):
            processes = []
            for i in range(process_num_max if j != process_num//process_num_max else process_num % process_num_max):
                p = multiprocessing.Process(target=RWWsimulation, args=(param_list[i+j*process_num_max]['G'],
                         param_list[i+j*process_num_max]['w_p'], param_list[i+j*process_num_max]['lamda'], 
                                        param_list[i+j*process_num_max]['sim_len'], i, result_queue))
                processes.append(p)
                p.start()
                print(f"Started process {i+j*process_num_max}")

            for i, p in zip(range(len(processes)), processes):
                p.join()
                p.close()
                print(f"Ended process {i+j*process_num_max}")
    else:
        processes = []
        for i in range(process_num):
            p = multiprocessing.Process(target=RWWsimulation, args=(param_list[i]['G'],
                     param_list[i]['w_p'], param_list[i]['lamda'], param_list[i]['sim_len'], i, result_queue))
            processes.append(p)
            p.start()
            print(f"Started process {i}")

        for i, p in zip(range(len(processes)), processes):
            p.join()
            p.close()
            print(f"Ended process {i}")
    
    r2_result = []
    while not result_queue.empty():
        result = result_queue.get()
        r2_result.append(result)
    
    end_time = t.time()

    duration = end_time - start_time
    
    print(r2_result)

    print(f"单参数优化总执行时间：{duration}秒")
    
    return r2_result
    
    
    