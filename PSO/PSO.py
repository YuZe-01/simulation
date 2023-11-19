import numpy as np
from tvb_simulation import multiprocess
import random
import sys
import datetime
from r2_ import true_data
import time

class PSO_model:
    def __init__(self,w,c1,c2,N,D,M,description):
        self.node_num = 192
        self.initargs = []
        self.w = w # 惯性权值
        self.c1=c1
        self.c2=c2
        self.threshold=[[0.0,10.0],[0.0,2.0],[0.0,1.0],[0.0,1.0]]
        self.v2p=0.1
        self.N=N # 初始化种群数量个数
        self.D=D # 搜索空间维度
        self.M=M # 迭代的最大次数
        self.x=np.zeros((self.N,self.D,self.node_num))  #粒子的初始位置
        self.v=np.zeros((self.N,self.D,self.node_num))  #粒子的初始速度
        self.pbest=np.zeros((self.N,self.D,self.node_num))  #个体最优值初始化
        self.gbest=np.zeros((1,self.D,self.node_num))  #种群最优值
        self.p_fit=np.zeros(self.N)
        self.fit=-1e8 #初始化全局最优适应度

        today = datetime.date.today()
        now = datetime.datetime.now()
        current_time = now.strftime("%H-%M-%S")
        
        self.log_file_path = f'./log/log_{today}_{current_time}.txt'
        psofile = open(self.log_file_path, 'w') 
        psofile.write(description) 
        psofile.close()     
 
    def print_parameter(self, file):
#         print("w: ", self.w,
#             "c1: ", self.c1,
#             "c2: ", self.c2,
#             "r1: ", self.r1,
#             "r2: ", self.r2,
#             "N: ", self.N,
#             "D: ", self.D,
#             "M: ", self.M,
#             "x: ", self.x[:,:,0:5],
#             "pbest: ", self.pbest[:,:,0:5],
#             "gbest: ", self.gbest[0,:,0:5],
#             "p_fit: ", self.p_fit,
#             "fit: ", self.fit)
        file.write(f"v:{self.v[0,:,0:5]}\n")
        file.write(f"x: {self.x[0,:,0:5]}\n")
        file.write(f"gbest: {self.gbest[0,:,0:5]}\n")
        # file.write(f"p_fit: {self.p_fit}\n")
        file.write(f"fit: {self.fit}\n")
    
    def alter(self, x):
        result = []
        keys = ['G', 'w_p', 'lamda', 'I_o']
        temp_dict = dict.fromkeys(keys)
        for i in range(self.N):
            for j in range(len(keys)):
                temp_dict[keys[j]] = x[i][j]
            
            result.append(temp_dict)
        
        return result

     # 初始化种群
    def init_pop(self):
        psofile = open(self.log_file_path, 'a')
        psofile.write("init_pop\n") 
        for i in range(self.N):
            for j in range(self.D):
                if j != 3:
                    self.x[i][j] = np.random.permutation(np.linspace(self.threshold[j][0], self.threshold[j][1], self.node_num))
                    self.v[i][j] = np.random.permutation(np.linspace(-self.threshold[j][1]*self.v2p, high = self.threshold[j][1]*self.v2p, self.node_num))
                else:
                    self.x[i][j] = np.random.permutation(np.linspace(self.threshold[j][0], self.threshold[j][1], 1))
                    self.v[i][j] = np.random.permutation(np.linspace(-self.threshold[j][1]*self.v2p, self.threshold[j][1]*self.v2p, 1))
                self.pbest[i] = self.x[i] # 初始化个体的最优值
        
        raw = self.alter(self.x)
        start = time.time()
        aim = multiprocess(raw) # 计算个体的适应度值 直接把参数空间内的所有可能性投入Multiprocessing中，多进程分批次跑完得到
                               #  参考值，r2或者MSE，
        end = time.time()
        self.p_fit = aim # 初始化个体的最优位置
        duration = end - start
        psofile.write(f"单参数优化总执行时间：{duration}秒\n")
        # psofile.write(f"PSO: {aim}\n") 
        for i in range(len(self.p_fit)):
            if self.p_fit[i] > self.fit:  # 对个体适应度进行比较，计算出最优的种群适应度
                self.fit = self.p_fit[i]
                self.gbest[0] = self.x[i]    
        
        self.print_parameter(psofile)
        psofile.write("init_pop finish\n")
        psofile.close()

        today = datetime.date.today()
        now = datetime.datetime.now()
        current_time = now.strftime("%H-%M-%S")

        np.savetxt(f'./data/initdata_{today}_{current_time}.txt', (self.gbest[0,0,:],self.gbest[0,1,:],self.gbest[0,2,:],self.gbest[0,3,:]))

    # 更新粒子的位置与速度
    def update(self):
        psofile = open(self.log_file_path, 'a')
        for t in range(self.M): # 在迭代次数M内进行循环
            r1 = random.random()
            r2 = random.random()
            psofile.write(f"r1: {r1} r2: {r2}\n")
            psofile.write(f"开始第{t+1}次迭代\n")
            for i in range(self.N): # 更新粒子的速度和位置
                weight = self.w[1] - (self.w[1] - self.w[0])*(t/self.M)
                self.v[i]=weight*self.v[i]+self.c1*r1*(self.pbest[i]-self.x[i])+ self.c2*r2*(self.gbest-self.x[i])
                for j in range(self.D):
                    self.v[i][j]=[-self.threshold[j][1]*self.v2p if x <= -self.threshold[j][1]*self.v2p 
                               else self.threshold[j][1]*self.v2p if x >= self.threshold[j][1]*self.v2p else x for x in self.v[i][j]]
                
                self.x[i]=self.x[i]+self.v[i]
                for j in range(self.D):
                    self.x[i][j]=[self.threshold[j][0] if x <= self.threshold[j][0]
                               else self.threshold[j][1] if x >= self.threshold[j][1] else x for x in self.x[i][j]]
            
            raw = self.alter(self.x)
            start = time.time()
            aim = multiprocess(raw)# 计算所有目标函数的适应度
            end = time.time()
            duration = end - start
            psofile.write(f"单参数优化总执行时间：{duration}秒\n")
            # psofile.write(f"PSO: {aim}\n")

            for i in range(len(aim)):
                if aim[i] >= self.p_fit[i]: # 比较适应度大小，将大的赋值给个体最优
                    self.p_fit[i]=aim[i]
                    self.pbest[i]=self.x[i]
                    if self.p_fit[i] >= self.fit: # 如果是个体最优再将和全体最优进行对比
                        self.gbest[0]=self.x[i]
                        self.fit = self.p_fit[i]
    
            self.print_parameter(psofile)
            psofile.write(f"第{t+1}次迭代完成\n")
            
        psofile.write(f"最优值：{self.fit}\n")
        psofile.write(f"位置为：{self.gbest[0,:,:]}\n")
        psofile.close()

        today = datetime.date.today()
        now = datetime.datetime.now()
        current_time = now.strftime("%H-%M-%S")

        np.savetxt(f'./data/finaldata_{today}_{current_time}.txt', (self.gbest[0,0,:],self.gbest[0,1,:],self.gbest[0,2,:],self.gbest[0,3,:]))

if __name__ == '__main__':
    # w,c1,c2,r1,r2,N,D,M参数初始化
    w=[0.4, 2]
    c1=c2=2#一般设置为2
    r1=0.7
    r2=0.5
    N=5
    D=3
    M=3
    pso_object=PSO_model(w,c1,c2,r1,r2,N,D,M)#设置初始权值
    pso_object.init_pop()
    pso_object.update()
