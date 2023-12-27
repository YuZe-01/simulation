import multiprocessing

import matplotlib.pyplot as plt
import scipy
from tvb.simulator.lab import *
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG
import numpy as np
import time as t
import logging

def multi(x):
    # x (N, D)
    start_time = t.time()

    process_num_max = 10
    process_num = len(x)

    if process_num > process_num_max:
        result = []
        result_queue = multiprocessing.Queue()
        for j in range(0, (process_num // process_num_max) + 1, 1):
            processes = []
            for i in range(process_num_max if j != process_num//process_num_max else process_num % process_num_max):
                # path = f'./plot/test_{i + j * process_num_max}.txt'
                # np.savetxt(path, (x[i+j*process_num_max, :]))

                p = multiprocessing.Process(target=simulation, args=(x[i+j*process_num_max, :], i+j*process_num_max, result_queue))
                processes.append(p)
                p.start()

            for i, p in zip(range(len(processes)), processes):
                p.join()
                p.close()

            while not result_queue.empty():
                temp_result = result_queue.get()
                result.append(temp_result)

            print(f'{j} batch over.')
    else:
        processes = []
        result_queue = multiprocessing.Queue()
        for i in range(process_num):
            p = multiprocessing.Process(target=simulation, args=(x[i, :], i, result_queue))
            processes.append(p)
            p.start()

        for i, p in zip(range(len(processes)), processes):
            p.join()
            p.close()

        result = []
        while not result_queue.empty():
            temp_result = result_queue.get()
            result.append(temp_result)

    end_time = t.time()

    duration = end_time - start_time

    result = sorted(result, key=lambda x: x['index'])

    list = []
    for i in range(len(result)):
        list.append(result[i]['value'])

    # print('list:', list)

    print(f"单参数优化总执行时间：{duration}秒")

    return list

def simulation(x, index, result_queue):
    start = t.time()

    logging.disable(logging.CRITICAL)  # 禁用所有日志输出

    # print('x:', x, index)

    WWM = models.ReducedWongWangExcInh(G=np.array([0]*66), w_p=np.array(x[0]*66), W_e=np.array(x[1]*66), W_i=np.array(x[2]*66))

    white_matter = connectivity.Connectivity.from_file('connectivity_66.zip')
    white_matter.speed = np.array([4.0])
    white_matter_coupling = coupling.Difference(a=np.array([0.014]))

    heunint = integrators.HeunStochastic(
        dt=2**-4,
        noise=noise.Additive(nsig=np.array([2 ** -5, ]))
    )

    # local_coupling_strength = np.array([2 ** -10])
    # default_cortex = Cortex.from_file(region_mapping_file='regionMapping_16k_76.txt')
    # default_cortex.region_mapping_data.connectivity = white_matter
    # default_cortex.coupling_strength = local_coupling_strength

    sim = simulator.Simulator(
        model=WWM,
        connectivity=white_matter,
        coupling=white_matter_coupling,
        integrator=heunint,
        # surface=default_cortex,
        monitors=(monitors.Raw(),),
        simulation_length=1e3,
    )

    sim.configure()

    temp = sim.run()

    time, data = temp[0]

    end = t.time()

    # print(end - start)

    temp = data[:, 0, :, 0] + data[:, 1, :, 0]

    data = data[:, 0, 0, 0] + data[:, 1, 0, 0]

    data = data.reshape(-1, 1)

    # 计算平均值和标准差
    mean_x = np.mean(data)
    std_x = np.std(data)

    # 标准化数据
    data = (data - mean_x) / std_x

    (f1, S1) = scipy.signal.welch(data.T, 16000, nperseg=16000)

    keys = ['index', 'value']
    temp_dict = dict.fromkeys(keys)
    temp_dict['index'] = index
    temp_dict['value'] = -(sum(S1[0, 8:13]))

    result_queue.put(temp_dict)

    # list = []
    # list1 = []
    # for i in range(66):
    #     data = temp[:, i]
    #     data = data.reshape(-1, 1)
    #
    #     # 计算平均值和标准差
    #     mean_x = np.mean(data)
    #     std_x = np.std(data)
    #
    #     # 标准化数据
    #     data = (data - mean_x) / std_x
    #
    #     (f1, S1) = scipy.signal.welch(data.T, 16000, nperseg=16000)
    #
    #     print((sum(S1[0, 8:13])/sum(S1[0, :100]))*100)
    #
    #     list1.append(sum(S1[0, 8:13]))
    #
    #     list.append((sum(S1[0, 8:13])/sum(S1[0, :100]))*100)
    #
    # plt.bar(range(0, 66), list)
    # plt.title("percentage(8-12)/(0,100)*100%")
    # plt.show()

    # plt.bar(range(0, 66), list1)
    # plt.title("sum(8-12)")
    # plt.show()

    # print(S1[i, 8]+S1[i, 9]+S1[i, 10]+S1[i, 11]+S1[i, 12])
    # return S1[0, 8]+S1[0, 9]+S1[0, 10]+S1[0, 11]+S1[0, 12]

def draw_picture(Y_history, x):
    simulation(x)
    i = len(Y_history)
    plt.plot(range(0, i), -np.min(Y_history, 1))
    # plt.title(f"genetic N={N} D={D} iteration={i}")
    # plt.legend()
    plt.show()

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    x = np.array([1.130919072953435078e+00, 1.408633512515828468e+00, 8.751690909367258048e-03])
    simulation(x, 1, queue)