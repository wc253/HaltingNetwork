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
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train
A=np.load('A_250_500.npy')

for i in range(17):
  prob = problems.bernoulli_gaussian_trial(A=A,M=250,N=500,L=1000,is_train=True)
  if i>=1:
    layers = networks.build_LISTA_cpss(prob,T=i,initial_lambda=.1,p=1.2,maxp=13,untied=False)
  else:
    layers = networks.build_LISTA_cpss(prob,T=1,initial_lambda=.1,p=1.2,maxp=13,untied=False)

# plan the learning
  training_stages = train.setup_training(layers,prob,i,trinit=1e-4,refinements=(.1,.01,.001),start=i)


  if i>=2:
    sess = train.do_training(training_stages,prob,'LISTA-cpss T='+str(i-1)+' trainrate=0.001.npz',printfile='layer_'+str(i)+'.txt') #'LISTA T='+str(i-1)+' trainrate=0.001.npz'
  else:
    sess = train.do_training(training_stages,prob,'Linear trainrate=0.001.npz')  
  tf.reset_default_graph()

