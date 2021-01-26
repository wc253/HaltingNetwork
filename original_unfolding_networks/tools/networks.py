#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la

import tensorflow as tf
import tools.shrinkage as shrinkage

def build_LISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.simple_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2+1e-4)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_)
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ )
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) )
    return layers



def build_LAMP(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2+1e-10)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    # if getattr(prob,'iid',True) == False:
    #     # set up individual parameters for every coordinate
    #     theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LISTA_cpss(prob,T,initial_lambda=.1,p=1.2, maxp=13,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.shrink_ss
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2+1e-4)
    ps=[(t+1) * p for t in range (T)]
    ps=np.clip(ps,0.0,maxp)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    # S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')

    xhat_ = eta( By_, lam0_,ps[0])
    layers.append( ('LISTA-cpss T=1',xhat_, (lam0_,) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        vt_ = prob.y_ - tf.matmul(prob.A_, xhat_)

        xhat_ = eta(xhat_+tf.matmul(B_,vt_), lam_,ps[t])
        layers.append( ('LISTA-cpss T='+str(t+1),xhat_,(lam_,)) )
    return layers