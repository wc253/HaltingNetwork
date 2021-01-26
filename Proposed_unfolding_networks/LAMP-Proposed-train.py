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
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train
A=np.load('A_250_500.npy')


tao=10
eps=0.01 #this value will not influence the traning results, it's used to see the layers that different signals use, the average layer is reduced if eps increase
for i in range(16,17):
    prob = problems.bernoulli_gaussian_trial(A=A,M=250,N=500,L=1000,is_train=True)
    layers = networks.build_LAMP_act(prob,tao,T=i,eps=eps,shrink='soft',untied=False)
    training_stages = train.setup_training(layers,i, prob, tao=tao, trinit=1e-4,type='LAMP')
    sess = train.do_training(training_stages,prob,'T_'+str(i)+'_tao_'+str(tao)+'.txt',restorefile='LAMP-soft T=16 trainrate=0.001.npz',maxit=400000,better_wait=10000)
    sess.close()
    tf.reset_default_graph()
