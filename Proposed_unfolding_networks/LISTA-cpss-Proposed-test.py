#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to 
a) select a problem to be solved 
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf
import scipy.io as sio

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train
A=np.load('A_250_500.npy')
# Create the basic problem structure.
# prob = problems.bernoulli_gaussian_trial(A=A,M=20,N=100,L=1000) #a Bernoulli-Gaussian x, noisily observed through a random matrix
#prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO


# tao=100
# eps=[0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.028,0.024,0.02,0.018,0.014,0.01,0.009,0.008,0.007,0.006,0.0055,0.005,0.0045,0.004,0.0035,0.003,0.0025,0.002,0.0015,0.001]
# tao=1
# eps=[0.7,0.6,0.5,0.4,0.3,0.2,0.18,0.16,0.14,0.12,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.018,0.016,0.014,0.012,0.01]
tao=10
p=0.1
eps=[0.3,0.12,0.1,0.09,0.08,0.07,0.06,0.05,0.045,0.04,0.038,0.036,0.034,0.032,0.03,0.028,0.026,0.024,0.022,0.02,0.018,0.016,0.014,0.012,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003]#,0.095,0.09,0.085,0.07,0.065,0.06,0.055,0.05,0.01]#,0.095,0.09,0.085,0.07,0.065,0.06,0.055,0.05,0.01
# p=0.01
# eps=[1,0.6,0.05,0.04,0.035,0.03,0.025,0.02,0.015,0.01]
# eps=[0.0001]
xs=[]
ys_mean=[]
ys_var=[]
ys_std=[]
T=16
count=1
M=250
N=500

loss_total=np.zeros([len(eps),10000])
for i in range(len(eps)):
    print(eps[i])
    prob = problems.bernoulli_gaussian_trial(A=A,M=M,N=N,L=1000,is_train=False)
    layers = networks.build_LISTA_cpss_act(prob,tao,T=T,eps=eps[i],initial_lambda=.1,p=1.2, maxp=13,untied=False)
    savefile = 'LISTA-cpss_T_'+str(T)+'_tao_' + str(tao) + '_count_'+str(count)+'.npz'
    num,num_min,num_max,loss_mean,loss_var,loss_std,loss=train.do_testing_another(layers,prob,tao,'T_10_tao_'+str(tao)+'_thresh_'+str(eps[i])+'_test.txt',savefile)
    print(num_min)
    print(num_max)
    xs.append(num)
    ys_mean.append(loss_mean)
    ys_var.append(loss_var)
    ys_std.append(loss_std)
    loss_total[i,:]=loss
    tf.reset_default_graph()
xs=np.asarray(xs)
ys_mean=np.asarray(ys_mean)
ys_var=np.asarray(ys_var)
ys_std=np.asarray(ys_std)
sio.savemat('LISTA-cpss_'+str(T)+'_noise_free_tao_'+str(tao)+'_mse_test',{'layer':xs,'loss':ys_mean,'loss_var':ys_var,'loss_std':ys_std,'loss_total':loss_total})


