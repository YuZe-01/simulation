import numpy as np
from tvb_simulation import multiprocess
import random
import sys

class PSO_model:
    def __init__(self,w,c1,c2,r1,r2,N,D,M):
        self.node_num = 192
        self.initargs = []
        self.w = w # 惯性权值
        self.c1=c1
        self.c2=c2
        self.r1=r1
        self.r2=r2
        self.N=N # 初始化种群数量个数
        self.D=D # 搜索空间维度
        self.M=M # 迭代的最大次数
        self.x=np.zeros((self.N,self.D,self.node_num))  #粒子的初始位置
        self.v=np.zeros((self.N,self.D,self.node_num))  #粒子的初始速度
        self.pbest=np.zeros((self.N,self.D,self.node_num))  #个体最优值初始化
        self.gbest=np.zeros((1,self.D,self.node_num))  #种群最优值
        self.p_fit=np.zeros(self.N)
        self.fit=-1e8 #初始化全局最优适应度
        self.stdout = sys.stdout
        
        try:
           sys.stdout = open('./log/PSO_log.txt', 'a')  # 将标准输出重定向到文件

        except Exception as e:
           print(f"写入文件出错：{str(e)}")
        
    def print_parameter(self):
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
        print("gbest: ", self.gbest[0,:,0:5],
            "p_fit: ", self.p_fit,
            "fit: ", self.fit)
    
    def alter(self, x):
        result = []
        keys = ['G', 'w_p', 'lamda', 'sim_len']
        for i in range(self.N):
            temp_dict = dict.fromkeys(keys)
            
            for j in range(len(keys)):
                if j != 3:
                    temp_dict[keys[j]] = x[i][j]
                else: 
                    temp_dict[keys[j]] = 1000.0
            
            result.append(temp_dict)
        
#         print(result)
            
        return result

     # 初始化种群
    def init_pop(self):
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = np.random.uniform(low = -5, high = 5, size=self.node_num)
                self.v[i][j] = np.random.uniform(low = -5, high = 5, size=self.node_num)
                
                self.pbest[i] = self.x[i] # 初始化个体的最优值
        
        raw = self.alter(self.x)

        aim = multiprocess(raw) # 计算个体的适应度值 直接把参数空间内的所有可能性投入Multiprocessing中，多进程分批次跑完得到
                               #  参考值，r2或者MSE，
        self.p_fit = aim # 初始化个体的最优位置
        
        for i,num in zip(self.p_fit, range(len(self.p_fit))):
            if i > self.fit:  # 对个体适应度进行比较，计算出最优的种群适应度
                self.fit = i
                self.gbest[0] = self.x[num]    
        
        self.print_parameter()
        print("init_pop finish")
        
        sys.stdout = self.stdout # 将标准输出重定向到文件
        print("init_pop finish")
        sys.stdout = open('./log/PSO_log.txt', 'a')  # 将标准输出重定向到文件
        
    # 更新粒子的位置与速度
    def update(self):
        for t in range(self.M): # 在迭代次数M内进行循环
            raw = self.alter(self.x)
            aim = multiprocess(raw)# 计算所有目标函数的适应度
            for i in range(len(aim)):
                if aim[i] >= self.p_fit[i]: # 比较适应度大小，将大的赋值给个体最优
                    self.p_fit[i]=aim[i]
                    self.pbest[i]=self.x[i]
                    if self.p_fit[i] >= self.fit: # 如果是个体最优再将和全体最优进行对比
                        self.gbest[0]=self.x[i]
                        self.fit = self.p_fit[i]
            for i in range(self.N): # 更新粒子的速度和位置
                weight = self.w[1] - (self.w[1] - self.w[0])*(t/self.M)
                self.v[i]=weight*self.v[i]+self.c1*self.r1*(self.pbest[i]-self.x[i])+ self.c2*self.r2*(self.gbest-self.x[i])
                self.x[i]=self.x[i]+self.v[i]
                
            self.print_parameter()
            print(f"迭代次数{t}完成")
            
            sys.stdout = self.stdout # 将标准输出重定向到文件
            print(f"迭代次数{t}完成")
            sys.stdout = open('./log/PSO_log.txt', 'a')  # 将标准输出重定向到文件
                
        print("最优值：",self.fit,"位置为：",self.gbest[0,:,:])


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
