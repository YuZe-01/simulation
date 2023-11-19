from tvb.simulator.lab import *
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG
import numpy as np
import time as t
from r2_ import r2_cal
import logging
import sys

if __name__ == '__main__':
    start = t.time()
    path = sys.argv[1]
    i = sys.argv[2]

    logging.disable(logging.CRITICAL)  # 禁用所有日志输出
    data = np.loadtxt(path)
    start_time = t.time()

    WWM = models.ReducedWongWangExcInh(G = data[0], w_p = data[1], lamda = data[2], I_o = data[3])

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
        simulation_length=1000.
    )
    sim.configure()

    eeg, _ = sim.run()

    time, data = eeg
    end = t.time()
    # r2 = -2.2
    r2 = r2_cal(data)
    path = open(f'./return/{i}.txt','w')
    path.write(str(r2))
    path.write('\n')
    path.write(str(end - start))
