import numpy as np
from sko.GA import GA
import matplotlib.pyplot as plt
import pandas as pd
from sko.PSO import PSO

from simulation import multi

D = 3
N = 100
i = 15
round = 5

if __name__ == "__main__":
    # pso = PSO(func=multi, n_dim=D, pop=N, max_iter=i, lb=[0, 0, 0], ub=[2, 2, 1], w=0.8, c1=0.5, c2=0.5, verbose=True)
    # pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    #
    # record = pso.record_value
    # np.savetxt(f'./single/pso_node{round}_best_x_best_y.txt', (pso.gbest_x, pso.gbest_y))
    # np.savetxt(f'./single/pso_node{round}_x.txt', np.array(record['X']).reshape(-1, D))
    # np.savetxt(f'./single/pso_node{round}_best_x.txt', np.array(record['Y']).reshape(-1, 1))
    #
    # # print(record, np.array(record).shape)
    # y_history = pso.gbest_y_hist
    # print(-np.array(y_history)[:, 0])
    # plt.plot(range(0, i), -np.array(y_history)[:, 0])
    # plt.title(f"pso N={N} D={D} iteration={i}")
    # plt.show()

    ga = GA(func=multi, n_dim=D, size_pop=N, max_iter=i, prob_mut=0.001, lb=[0, 0, 0], ub=[2, 2, 1], precision=1e-7)
    best_x, best_y = ga.run()
    np.savetxt(f'./single/node{round}.txt', (best_x))
    print('best_x:', best_x, '\n', 'best_y:', best_y)

    Y_history = ga.all_history_Y
    X_history = ga.all_history_X[0]
    best_X_history = ga.best_history_X

    np.savetxt(f'./single/node{round}_x.txt', X_history)
    np.savetxt(f'./single/node{round}_best_x.txt', best_X_history)

    print(np.min(Y_history, 1))
    plt.plot(range(0, i), -np.min(Y_history, 1))
    plt.title(f"genetic N={N} D={D} iteration={i}")
    # plt.legend()
    plt.show()
