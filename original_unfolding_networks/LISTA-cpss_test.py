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
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train
A=np.load('A_250_500.npy')
prob = problems.bernoulli_gaussian_trial(A=A,M=250,N=500,L=1000,is_train=False)
T=14

layers = networks.build_LISTA_cpss(prob,T=T,initial_lambda=.1,p=1.2, maxp=13,untied=False)
train.do_testing_another(layers,prob,T,'LISTA-cpss T='+str(T)+' trainrate=0.001.npz')
