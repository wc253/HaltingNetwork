#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la

import tensorflow as tf
import tools.shrinkage as shrinkage



def adaptive_computation_time(halting_probas,eps=.06):
    sh=halting_probas.get_shape().as_list()
    batch=sh[0]
    max_units=sh[1]

    zero_col = tf.ones((batch, 1)) * 1e-8  #LISTA and LAMP use 1e-6 LISTA-CPSS use 1e-8
    halting_padded = tf.concat([halting_probas[:, :-1], zero_col], axis=1)

    halt_flag_final=( halting_padded<=eps) #[batchsize,T] 最后一层h为1
    decay=1./(10.+tf.to_float(tf.range(max_units)))
    halt_flag_final_with_decay=tf.to_float(halt_flag_final)+decay[None,:]
    N=tf.to_int32(tf.argmax(halt_flag_final_with_decay,dimension=1))

    N=tf.stop_gradient(N)

    num_units=N+1 #有多少层起作用了 [batchsize,1]

    unit_index = tf.range(max_units)

    p = tf.where(tf.less_equal(unit_index, (max_units-1)*tf.ones((max_units),dtype=tf.int32)), tf.ones((max_units)), tf.zeros((max_units)))

    return num_units,halting_padded,tf.to_float(p),tf.reduce_max(num_units)

def build_LISTA_act(prob,tao,T,eps=0.01,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.simple_soft_threshold
    xhats = []
    halting_distribs=[]
    layers=[]
    batchsize=prob.L
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A, 2) ** 2 + 1e-10)
    # B = A.T
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    # if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        # initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_)
    xhats.append(xhat_)  #tf.reduce_sum(tf.squared_difference(prob.y_,tf.matmul(A,xhat_)),axis=0,keep_dims=True)
    # halting_distrib=tf.nn.sigmoid(fully_connect(tf.transpose(tf.concat([tf.reduce_sum(tf.squared_difference(prob.y_,tf.matmul(A,tf.stop_gradient(xhat_))),axis=0,keep_dims=True),tf.norm(tf.stop_gradient(xhat_),ord=1,axis=0,keepdims=True)],axis=0)),output_size=1,scope='LISTA_T_1'))
    w1=tf.Variable(1,name='h_w1_0',dtype=tf.float32)
    C = tf.Variable(np.eye(M), name='h_w2_0', dtype=tf.float32)
    b=tf.Variable(0,name='h_b_0',dtype=tf.float32)
    halting_distrib = tf.nn.sigmoid(w1*tf.transpose(
        tf.reduce_sum(tf.squared_difference(tf.matmul(C,prob.y_), tf.matmul(C,tf.matmul(prob.A_, tf.stop_gradient(xhat_)))), axis=0, keep_dims=True)
         )+b)
    halting_distribs.append(halting_distrib)
    for t in range(1,T):
        lam_ = tf.Variable(initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ ) #[N,batchsize]
        xhats.append(xhat_)
        # if t<T-1:
        w1 = tf.Variable(1, name='h_w1_{0}'.format(t),dtype=tf.float32)
            # C = tf.Variable(np.eye(M), name='h_w2_{0}'.format(t), dtype=tf.float32)
        b = tf.Variable(0, name='h_b_{0}'.format(t),dtype=tf.float32)
        halting_distrib = tf.nn.sigmoid(w1*tf.transpose(tf.reduce_sum(
                    tf.squared_difference(tf.matmul(C,prob.y_), tf.matmul(C,tf.matmul(prob.A_, tf.stop_gradient(xhat_)))), axis=0, keep_dims=True) )
                    +b)
        halting_distribs.append(halting_distrib)
    halting_distribs=tf.concat(halting_distribs,1)
    num_units, halting_distribution,p,max_num=adaptive_computation_time(halting_distribs,eps=eps)

    xhat_final1 = tf.zeros((N, batchsize))
    xhat_final3= []
    for i in range(T):

        xhat_final1 = xhat_final1 + tf.to_float(
                tf.equal(tf.squeeze(tf.reshape([i] * batchsize, shape=(batchsize, 1))),
                         num_units - 1)) * xhats[i]

        xhat_final3.append(tf.reduce_sum(tf.squared_difference(xhats[i], prob.x_), axis=0,keep_dims=True))
    xhat_final3=tf.transpose(tf.concat(xhat_final3,axis=0))
    xhat_final2=tf.reduce_sum(p*(xhat_final3/(halting_distribution+1e-6)+tao*halting_distribution),axis=1)
    layers.append((xhat_final1,xhat_final2,xhat_final3,tf.transpose(num_units),xhats,halting_distribution,max_num,p))
    return layers

def build_LAMP_act(prob,tao,T,eps,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    xhats = []
    halting_distribs = []
    layers = []
    batchsize = prob.L
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )


    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    xhats.append(xhat_)
    w1 = tf.Variable(1, name='h_w1_0', dtype=tf.float32)
    C = tf.Variable(np.eye(M), name='h_w2_0', dtype=tf.float32)
    b = tf.Variable(0, name='h_b_0', dtype=tf.float32)
    halting_distrib = tf.nn.sigmoid(w1 * tf.transpose(
        tf.reduce_sum(tf.square(tf.matmul(C, prob.y_-tf.matmul(prob.A_, tf.stop_gradient(xhat_)))),
                      axis=0, keep_dims=True)
    ) + b)
    halting_distribs.append(halting_distrib)

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

        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        xhats.append(xhat_)
        w1 = tf.Variable(1, name='h_w1_{0}'.format(t), dtype=tf.float32)
        b = tf.Variable(0, name='h_b_{0}'.format(t), dtype=tf.float32)
        halting_distrib = tf.nn.sigmoid(w1 * tf.transpose(tf.reduce_sum(
            tf.square(tf.matmul(C, prob.y_-tf.matmul(prob.A_, tf.stop_gradient(xhat_)))), axis=0,keep_dims=True))
                                        + b)
        halting_distribs.append(halting_distrib)
    halting_distribs = tf.concat(halting_distribs, 1)
    num_units, halting_distribution, p, max_num = adaptive_computation_time(halting_distribs, eps=eps)
    # xhat_final2 = tf.zeros((batchsize))
    xhat_final1 = tf.zeros((N, batchsize))
    xhat_final3 = []
    for i in range(T):
        xhat_final1 = xhat_final1 + tf.to_float(
            tf.equal(tf.squeeze(tf.reshape([i] * batchsize, shape=(batchsize, 1))),
                     num_units - 1)) * xhats[i]

        # xhat_final2 = xhat_final2 + tf.transpose(halting_distribution[:, i])
        xhat_final3.append(tf.reduce_sum(tf.squared_difference(xhats[i], prob.x_), axis=0, keep_dims=True))
    xhat_final3 = tf.transpose(tf.concat(xhat_final3, axis=0))
    xhat_final2 = tf.reduce_sum(p * (xhat_final3 / (halting_distribution + 1e-6) + tao * halting_distribution), axis=1)
    layers.append(
        (xhat_final1, xhat_final2, xhat_final3, tf.transpose(num_units), xhats, halting_distribution, max_num, p))

    return layers

def build_LISTA_cpss_act(prob,tao,T,eps=0.01,initial_lambda=.1,p=1.2, maxp=13,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.shrink_ss

    xhats = []
    halting_distribs = []
    layers = []
    A = prob.A
    M,N = A.shape
    batchsize = prob.L
    B = A.T / (1.01 * la.norm(A,2)**2+1e-4)
    ps=[(t+1) * p for t in range (T)]
    ps=np.clip(ps,0.0,maxp)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )


    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')

    xhat_ = eta( By_, lam0_,ps[0])
    xhats.append(xhat_)
    w1 = tf.Variable(1, name='h_w1_0', dtype=tf.float32)
    C = tf.Variable(np.eye(M), name='h_w2_0', dtype=tf.float32)
    b = tf.Variable(0, name='h_b_0', dtype=tf.float32)
    halting_distrib = tf.nn.sigmoid(w1 * tf.transpose(
        tf.reduce_sum(tf.square(tf.matmul(C, prob.y_-tf.matmul(prob.A_, tf.stop_gradient(xhat_)))),axis=0, keep_dims=True)
    ) + b)
    halting_distribs.append(halting_distrib)


    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        vt_ = prob.y_ - tf.matmul(prob.A_, xhat_)

        xhat_ = eta(xhat_+tf.matmul(B_,vt_), lam_,ps[t])
        xhats.append(xhat_)
        # if t<T-1:
        w1 = tf.Variable(1, name='h_w1_{0}'.format(t), dtype=tf.float32)
        b = tf.Variable(0, name='h_b_{0}'.format(t), dtype=tf.float32)
        halting_distrib = tf.nn.sigmoid(w1 * tf.transpose(tf.reduce_sum(
            tf.square(tf.matmul(C, prob.y_-tf.matmul(prob.A_, tf.stop_gradient(xhat_)))), axis=0,keep_dims=True))+ b)
        halting_distribs.append(halting_distrib)
    halting_distribs = tf.concat(halting_distribs, 1)
    num_units, halting_distribution, p, max_num = adaptive_computation_time(halting_distribs, eps=eps)
    # xhat_final2 = tf.zeros((batchsize))
    xhat_final1 = tf.zeros((N, batchsize))
    xhat_final3 = []
    for i in range(T):
        xhat_final1 = xhat_final1 + tf.to_float(
            tf.equal(tf.squeeze(tf.reshape([i] * batchsize, shape=(batchsize, 1))),
                     num_units - 1)) * xhats[i]

        # xhat_final2 = xhat_final2 + tf.transpose(halting_distribution[:, i])
        xhat_final3.append(tf.reduce_sum(tf.squared_difference(xhats[i], prob.x_), axis=0, keep_dims=True))
    xhat_final3 = tf.transpose(tf.concat(xhat_final3, axis=0))
    xhat_final2 = tf.reduce_sum(p * (xhat_final3 / (halting_distribution + 1e-8) + tao * halting_distribution), axis=1)
    layers.append(
        (xhat_final1, xhat_final2, xhat_final3, tf.transpose(num_units), xhats, halting_distribution, max_num, p))

    return layers
