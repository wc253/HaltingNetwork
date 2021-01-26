#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
import scipy.io as sio
from tools import problems
def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:  ##记得该回�?
                # print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
                # if 'w2' in k:
                #     U,sigma,V=np.linalg.svd(d)
                #     print(sigma)
                # if 'h' in k:
                #     print(k)
                #     print('\n')
                #     print(d)
            else:
                other[k] = d
    except IOError:
        pass
    return other

def load_LISTA_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() if 'LISTA_T_' not in v.name])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other

def setup_training(layer_info,T,prob,tao=0.22, trinit=1e-3,type='LISTA'):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]

    for xhat1_,xhat2_,xhat3_,num_units_,xhats_,halting_distribution_,max_num_,p_ in layer_info:
        loss_care=tf.reduce_mean(tf.reduce_sum(tf.squared_difference(xhat1_, prob.x_),axis=0)/tf.reduce_sum(tf.square(prob.x_),axis=0))
        t_vars=tf.trainable_variables()

        h_vars = [var for var in t_vars if 'h_w1'  in var.name or 'h_w2' in var.name or 'h_b' in var.name]
        count=0
        for i in range(1,2):
            loss = (tf.reduce_mean(xhat2_)*1e-4)
            if i==0:
                train_ = tf.train.AdamOptimizer(tr_ ).minimize(loss, var_list=h_vars)
            else:
                train_ = tf.train.AdamOptimizer(1e-6).minimize(loss)
            training_stages.append(('state'+str(i), xhat1_, xhat2_,xhat3_, loss, loss_care, num_units_,train_,
                                        xhats_, halting_distribution_,max_num_,p_,type+'_T_'+str(T)+'_tao_'+str(tao)+'_count_'+str(i)+'.npz'))
            count = count + 1

    return training_stages



def do_training(training_stages,prob,printfile,restorefile='LISTA_T_10.npz',ivl=100,maxit=100000,better_wait=10000):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    f1=open(printfile,'w')
    f1.close()
    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state=load_trainable_vars(sess,restorefile)

    done=state.get('done',[])

    xval = np.load('xval.npy')
    yval = np.load('yval.npy')
    count = -1

    data_batch, data_initializer = prob.get_batch()
    sess.run(data_initializer)

    lossbest = np.inf
    for  name,xhat1_,xhat2_,xhat3_,loss_,loss_care_,num_units_,train_,xhats_,halting_distribution_,max_num_,p_,savefile in training_stages:
        # if name in done:
        #    print('Already did ' + name + '. Skipping.')
        #    continue
        print('\n')
        print(name)
        f1 = open(printfile, 'a+')
        f1.write(name)
        f1.write('\n')
        f1.close()

        loss_history = []

        # loss_history = np.append(loss_history, lossbest)
        for i in range(maxit+1):
            if i%ivl == 0:

                loss,losscare,num,n_,h__,error__,max_num__,p__= sess.run([loss_,loss_care_,tf.reduce_sum(num_units_),num_units_,halting_distribution_,xhat3_,max_num_,p_],feed_dict={prob.y_:yval,prob.x_:xval})
                # sio.savemat('h1.mat', {'h': h__, 'x': xval,'error':error__,'p':p__,'num':n_})

                num1,loss1=sess.run([tf.reduce_sum(num_units_),loss_care_], feed_dict={prob.y_:prob.yval1,prob.x_:prob.xval1})
                num2,loss2=sess.run([tf.reduce_sum(num_units_),loss_care_], feed_dict={prob.y_:prob.yval2,prob.x_:prob.xval2})
                num3,loss3=sess.run([tf.reduce_sum(num_units_),loss_care_], feed_dict={prob.y_:prob.yval3,prob.x_:prob.xval3})
                num4,loss4=sess.run([tf.reduce_sum(num_units_),loss_care_], feed_dict={prob.y_:prob.yval4,prob.x_:prob.xval4})

                if np.isnan(loss):
                    raise RuntimeError('loss is NaN')
                loss_history = np.append(loss_history,loss)
                lossbest = loss_history.min()
                f1=open(printfile,'a+')
                f1.write('\ri={i:<6d} loss={loss:.6f} (best={best:.6f}) losscare={losscare:.6f} num={num:.6f} num1={num1:.6f} num2={num2:.6f} num3={num3:.6f} num4={num4:.6f} loss1={loss1:.6f} loss2={loss2:.6f} loss3={loss3:.6f} loss4={loss4:.6f}\n'.format(i=i,loss=loss,best=lossbest,losscare=losscare,num=num/prob.L,num1=num1/prob.L,num2=num2/prob.L,num3=num3/prob.L,num4=num4/prob.L,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4))
                f1.close()
                sys.stdout.write('\ri={i:<6d} loss={loss:.6f} (best={best:.6f}) losscare={losscare:.6f} num={num:.6f} max_num={max_num:.6f} num1={num1:.6f} num2={num2:.6f} num3={num3:.6f} num4={num4:.6f} loss1={loss1:.6f} loss2={loss2:.6f} loss3={loss3:.6f} loss4={loss4:.6f}'.format(i=i,loss=loss,best=lossbest,losscare=losscare,num=num/prob.L,max_num=max_num__,num1=num1/prob.L,num2=num2/prob.L,num3=num3/prob.L,num4=num4/prob.L,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4))
                sys.stdout.flush()
                if i%(50*ivl) == 0:

                    age_of_best = len(loss_history) - loss_history.argmin()-1 # how long ago was the best nmse
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
                    save_trainable_vars(sess, savefile)
                    print('')
                    # tf.train.Saver(tf.global_variables(), max_to_keep=10).save(sess,'./model.ckpt'+name,global_step=i)
            data_batch_ = sess.run(data_batch)
            y = data_batch_['y']
            x = data_batch_['x']
            sess.run(train_, feed_dict={prob.y_: y[0,:,:], prob.x_: x[0,:,:]})
        done = np.append(done,name)

    

        state['done'] = done

        save_trainable_vars(sess,savefile,**state)
    return sess


def do_testing_another(layer_info,prob,tao,printfile,savefile,last_stage='0.001'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('norms xtest:{xtest:.7f} ytest:{ytest:.7f}'.format(xtest=la.norm(prob.xval), ytest=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    for xhat1_, xhat2_, ponder_cost_, num_units_, xhats_, halting_distribution_,max_num_,p_ in layer_info:
        loss_care_ = tf.reduce_sum(tf.squared_difference(xhat1_, prob.x_), axis=0)
        # loss_care_ =tf.reduce_sum(tf.squared_difference(xhat1_, prob.x_), axis=0)/ tf.reduce_sum(tf.square(prob.x_), axis=0)
        x = np.load('xtest_uni.npy')
        y = np.load('ytest_uni.npy')

        num_units = []
        loss = []

        for i in range((10000 // prob.L)):
            num_units__, loss__ = sess.run([num_units_, loss_care_],
                                           feed_dict={prob.y_: y[:, i * prob.L:(i + 1) * prob.L],
                                                      prob.x_: x[:, i * prob.L:(i + 1) * prob.L]})

            if i == 0:
                loss = loss__
                num_units = num_units__
            else:
                loss = np.concatenate([loss, loss__])
                num_units = np.concatenate([num_units, num_units__])


        print('loss is %.5f' % np.mean(loss))
        print('std is %.5f' % np.std(loss))
        print('N is %.5f' % np.mean(num_units))

    sess.close()
    return np.mean(num_units),np.min(num_units),np.max(num_units),np.mean(loss),np.var(loss),np.std(loss),loss

