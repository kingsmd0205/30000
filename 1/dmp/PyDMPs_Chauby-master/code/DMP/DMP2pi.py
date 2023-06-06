# -*- coding: utf-8 -*-
'''
This code is implemented by Chauby, it is free for everyone.
Email: chaubyZou@163.com
'''

#%% import package
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

from cs import CanonicalSystem

#%% define discrete dmp
class dmp_discrete():
    def __init__(self, n_dmps=1, n_bfs=100, dt=0, alpha_y=None, beta_y=None, **kwargs):
        self.n_dmps = n_dmps # number of data dimensions, one dmp for one degree
        self.n_bfs = n_bfs # number of basis functions
        self.dt = dt

        self.y0 = np.zeros(n_dmps)  # for multiple dimensions
        self.goal = np.ones(n_dmps) # for multiple dimensions

        alpha_y_tmp = 60 if alpha_y is None else alpha_y
        beta_y_tmp = alpha_y_tmp / 4.0  if beta_y is None else beta_y
        self.alpha_y = np.ones(n_dmps) * alpha_y_tmp
        self.beta_y = np.ones(n_dmps) * beta_y_tmp
        self.tau = 1.0
        self.delta = np.ones((n_dmps, 1))
        self.delta_2 = np.ones((n_dmps, 1))

        self.w = np.zeros((n_dmps, n_bfs)) # weights for forcing term
        self.psi_centers = np.zeros(self.n_bfs) # centers over canonical system for Gaussian basis functions
        self.psi_h = np.zeros(self.n_bfs) # variance over canonical system for Gaussian basis functions

        # canonical system
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = round(self.cs.run_time / self.dt)

        # generate centers for Gaussian basis functions
        self.generate_centers()

        # self.h = np.ones(self.n_bfs) * self.n_bfs / self.psi_centers # original
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.psi_centers / self.cs.alpha_x # chose from trail and error

        # reset state
        self.reset_state()

    # Reset the system state
    def reset_state(self):
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def generate_centers(self):
        t_centers = np.linspace(0, self.cs.run_time, self.n_bfs) # centers over time

        cs = self.cs
        x_track = cs.run() # get all x over run time
        t_track = np.linspace(0, cs.run_time, cs.timesteps) # get all time ticks over run time

        for n in range(len(t_centers)):
            for i, t in enumerate(t_track):
                if abs(t_centers[n] - t) <= cs.dt: # find the x center corresponding to the time center
                    self.psi_centers[n] = x_track[i]
        
        return self.psi_centers

    def generate_psi(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None]

        self.psi = np.exp(-self.h * (x - self.psi_centers)**2)

        return self.psi
    
    def generate_weights(self, f_target):
        x_track = self.cs.run()
        psi_track = self.generate_psi(x_track)

        for d in range(self.n_dmps):
            # ------------ Original DMP in Schaal 2002
            # delta = self.goal[d] - self.y0[d]
            
            # ------------ Modified DMP in Schaal 2008
            delta = 1.0
            self.delta[d] = self.goal[d] - self.y0[d]

            for b in range(self.n_bfs):
                # as both number and denom has x(g-y_0) term, thus we can simplify the calculation process
                numer = np.sum(x_track * psi_track[:,b] * f_target[:,d])
                denom = np.sum(x_track**2 * psi_track[:,b])
                # numer = np.sum(psi_track[:,b] * f_target[:,d]) # the simpler calculation
                # denom = np.sum(x_track * psi_track[:,b])
                # self.w[d, b] = numer / (denom*delta)


                self.w[d, b] = numer / denom
                if abs(delta) > 1e-6:
                    self.w[d, b] = self.w[d, b] / delta
                
        self.w = np.nan_to_num(self.w)

        return self.w

    def learning(self, y_demo, plot=False):
        if y_demo.ndim == 1: # data is with only one dimension
            y_demo = y_demo.reshape(1, len(y_demo))

        self.y0 = y_demo[:,0].copy()
        self.goal = y_demo[:,-1].copy()
        self.y_demo = y_demo.copy()

        # interpolate the demonstrated trajectory to be the same length with timesteps
        x = np.linspace(0, self.cs.run_time, y_demo.shape[1])
        y = np.zeros((self.n_dmps, self.timesteps))
        for d in range(self.n_dmps):
            y_tmp = interp1d(x, y_demo[d])
            for t in range(self.timesteps):
                y[d, t] = y_tmp(t*self.dt)
        
        # calculate velocity and acceleration of y_demo

        # method 1: using gradient
        dy_demo = np.gradient(y, axis=1) / self.dt
        ddy_demo = np.gradient(dy_demo, axis=1) / self.dt

        # method 2: using diff
        # dy_demo = np.diff(y) / self.dt
        # # let the first gradient same as the second gradient
        # dy_demo = np.hstack((np.zeros((self.n_dmps, 1)), dy_demo)) # Not sure if is it a bug?
        # # dy_demo = np.hstack((dy_demo[:,0].reshape(self.n_dmps, 1), dy_demo))

        # ddy_demo = np.diff(dy_demo) / self.dt
        # # let the first gradient same as the second gradient
        # ddy_demo = np.hstack((np.zeros((self.n_dmps, 1)), ddy_demo))
        # # ddy_demo = np.hstack((ddy_demo[:,0].reshape(self.n_dmps, 1), ddy_demo))

        x_track = self.cs.run()
        f_target = np.zeros((y_demo.shape[1], self.n_dmps))
        for d in range(self.n_dmps):
            # ---------- Original DMP in Schaal 2002
            # f_target[:,d] = ddy_demo[d] - self.alpha_y[d]*(self.beta_y[d]*(self.goal[d] - y_demo[d]) - dy_demo[d])

            # ---------- Modified DMP in Schaal 2008, fixed the problem of g-y_0 -> 0
            k = self.alpha_y[d]
            f_target[:,d] = (ddy_demo[d] - self.alpha_y[d]*(self.beta_y[d]*(self.goal[d] - y_demo[d]) - dy_demo[d]))/k + x_track*(self.goal[d] - self.y0[d])
        
        self.generate_weights(f_target)

        if plot is True:
            # plot the basis function activations
            plt.figure()
            plt.subplot(211)
            psi_track = self.generate_psi(self.cs.run())
            plt.plot(psi_track)
            plt.title('basis functions')

            # plot the desired forcing function vs approx
            plt.subplot(212)
            plt.plot(f_target[:,0])
            plt.plot(np.sum(psi_track * self.w[0], axis=1) * self.dt)
            plt.legend(['f_target', 'w*psi'])
            plt.title('DMP forcing function')
            plt.tight_layout()
            plt.show()

        # reset state
        self.reset_state()

    def reproduce(self, tau=None, initial=None, goal=None):
        # set temporal scaling
        if tau == None:
            timesteps = self.timesteps
        else:
            timesteps = round(self.timesteps/tau)

        # set initial state
        if initial != None:
            self.y0 = initial
        
        # set goal state
        if goal != None:
            self.goal = goal
        
        # reset state
        self.reset_state()

        y_reproduce = np.zeros((timesteps, self.n_dmps))
        dy_reproduce = np.zeros((timesteps, self.n_dmps))
        ddy_reproduce = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            y_reproduce[t], dy_reproduce[t], ddy_reproduce[t] = self.step(tau=tau)
        
        return y_reproduce, dy_reproduce, ddy_reproduce

    def step(self, tau=None):
        # run canonical system
        if tau == None:
            tau = self.tau
        x = self.cs.step_discrete(tau)

        # generate basis function activation
        psi = self.generate_psi(x)

        for d in range(self.n_dmps):
            # generate forcing term
            # ------------ Original DMP in Schaal 2002
            # f = np.dot(psi, self.w[d])*x*(self.goal[d] - self.y0[d]) / np.sum(psi)

            # ---------- Modified DMP in Schaal 2008, fixed the problem of g-y_0 -> 0
            # k = self.alpha_y[d]
            # f = k*(np.dot(psi, self.w[d])*x / np.sum(psi)) - k*(self.goal[d] - self.y0[d])*x # Modified DMP
            
            # ---------- Modified DMP with a simple solution to overcome the drawbacks of trajectory reproduction
            k = self.alpha_y[d]

            self.delta_2[d] = self.goal[d] - self.y0[d] # Modified DMP extended
            if abs(self.delta[d]) > 1e-5:
                k2 = self.delta_2[d]/self.delta[d]
            else:
                k2 = 1.0

            f = k*(np.dot(psi, self.w[d])*x*k2 / np.sum(psi)) - k*(self.goal[d] - self.y0[d])*x


            # generate reproduced trajectory
            self.ddy[d] = self.alpha_y[d]*(self.beta_y[d]*(self.goal[d] - self.y[d]) - self.dy[d]) + f
            self.dy[d] += tau*self.ddy[d]*self.dt
            self.y[d] += tau*self.dy[d]*self.dt
        
        return self.y, self.dy, self.ddy


# %% test code
if __name__ == "__main__":
    csv_data = pd.read_csv(r"C:\Users\swife\Desktop\DMP\data\1recording1.csv", header=None)
    data_len = len(csv_data)

    # ----------------- For different initial and goal positions
    t = np.linspace(0, 1.5 * np.pi, data_len)
    y_demo = np.zeros((7, data_len))
    y_demo[0, :] = csv_data.iloc[:, 0].values
    y_demo[1, :] = csv_data.iloc[:, 1].values
    y_demo[2, :] = csv_data.iloc[:, 2].values
    y_demo[3, :] = csv_data.iloc[:, 3].values
    y_demo[4, :] = csv_data.iloc[:, 4].values
    y_demo[5, :] = csv_data.iloc[:, 5].values
    y_demo[6, :] = csv_data.iloc[:, 6].values
    print(y_demo)
    # 检查 y_demo[3, :] 是否有数据小于 0
    less_than_zero_indices = y_demo[4, :] < 0

    # 如果找到小于 0 的数据，将其加上 6.8
    y_demo[4, less_than_zero_indices] +=  2 * np.pi


    # 空间轨迹 DMP learning
    dmp = dmp_discrete(n_dmps=y_demo.shape[0], n_bfs=100, dt=1.0 / data_len)
    dmp.learning(y_demo, plot=False)

    # reproduce learned trajectory
    y_reproduce, dy_reproduce, ddy_reproduce = dmp.reproduce()

    # 检查 y_reproduce[:, 3] 中是否有大于 pi 的值
    greater_than_pi_indices = np.where(y_reproduce[:, 4] > np.pi)

    # 如果有大于 pi 的值，从这些值中减去 2π
    y_reproduce[greater_than_pi_indices, 4] -= 2 * np.pi

    # start = y_demo[:, 0].tolist()
    # end = y_demo[:, -1].tolist()
    # delta = [0,0.2, -0.3, 0.01,0,0,0,]
    # start = [float(x) + d for x, d in zip(start, delta)]
    # end   = [float(x) + d for x, d in zip(end, delta)]
    start = [1.002050161,0.731300477933831,0.508035022565628,-0.0211859268236256,3.12922822408803,-0.00766777798069065,-2.93673605883269]
    end = [14.56303811,0.728458124200021,0.270786436035884,-0.0202265331216688,3.1281607629809,-0.0545719031354862,-2.96679272683969]

    # set new initial and goal poisitions     initial 的第一个元素（0.2）表示第一个维度（正弦曲线）的起始位置，第二个元素（0.8）表示第二个维度（余弦曲线）的起始位置。
    y_reproduce_2, dy_reproduce_2, ddy_reproduce_2 = dmp.reproduce(tau=1, initial=start, goal=end)

    # 检查 y_reproduce[:, 3] 中是否有大于 pi 的值
    greater_than_pi_indices = np.where(y_reproduce_2[:, 4] > np.pi)

    # 如果有大于 pi 的值，从这些值中减去 2π
    y_reproduce_2[greater_than_pi_indices, 4] -= 2 * np.pi




    for i in range(7):
        plt.figure(figsize=(10, 5))
        plt.plot(csv_data.iloc[:, i].values, 'g', label='DMP')
        plt.plot(y_reproduce[:, i], 'r--', label='reproduce')
        plt.plot(y_reproduce_2[:, i], 'b-.', label='generalization')
        plt.legend(loc="upper right")
        # 更新坐标轴范围以适应数据
        plt.gca().relim()
        plt.gca().autoscale_view()

        plt.grid()
        plt.xlabel('time')
        plt.ylabel('y')
        plt.title('2008---dmp')  # 添加标题
        plt.savefig(f'picture/impove/dmp_{i}.png')
        plt.show()


y_reproduce_df0 = pd.DataFrame(y_reproduce)

y_reproduce_df0.to_csv(r"C:\Users\swife\Desktop\75555261recording1outputDMPreproduce.csv", index=False, header=None)



y_reproduce_df = pd.DataFrame(y_reproduce_2)

# 将 DataFrame 写入 CSV 文件
y_reproduce_df.to_csv(r"C:\Users\swife\Desktop\75555261recording1generalization.csv", index=False, header=None)














































