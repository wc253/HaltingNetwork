#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
import scipy.io as sio
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
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
                print(d)
            else:
                other[k] = d
    except IOError:
        pass
    return other

def setup_training(layer_info,prob,T, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None,start=1):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    count=-1
    flag=-1
    for name,xhat_,var_list in layer_info:
        count=count+1
        if count<start or flag>0:
            continue
        flag=1
        loss_  = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(xhat_,prob.x_),axis=0))
        loss_original_ = tf.reduce_sum(tf.squared_difference(xhat_, prob.x_), axis=0)

        if var_list is not None:
            train_=tf.train.AdamOptimizer(tr_).minimize(loss_,var_list=var_list)
            training_stages.append((name ,xhat_,loss_,loss_original_,train_,var_list))
        #train2_ = tf.train.AdamOptimizer(tr_).minimize(loss_)
        #training_stages.append((name, xhat_, loss_, loss_original_, train2_, ()))

        # if name != 'LVAMP-bg T='+str(T):
        #     print(name)
        #     continue
        for fm in refinements:
        # fm=refinements
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,loss_original_,train2_,()) )
            print(name+' trainrate=' + str(fm))


    return training_stages


def do_training(training_stages,prob,savefile,ivl=100,printfile='run.txt'):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))
    f1=open(printfile,'w')
    f1.close()

    xval = np.load('xval.npy')
    yval = np.load('yval.npy')

    data_batch, data_initializer = prob.get_batch()
    sess.run(data_initializer)

    for name,xhat_,loss_,loss_original_,train_,var_list in training_stages:
        maxiter=300000
        wait_time=10000
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        f1=open(printfile,'a+')
        f1.write(name)
        f1.close()
        print(name + ' ' + describe_var_list)
        loss_history=[]

       # print(total_count)
        for i in range(maxiter+1):
            if i%ivl == 0:
                # xval = np.load('xval.npy')
                # yval = np.load('yval.npy')
                loss= sess.run(loss_,feed_dict={prob.y_:yval,prob.x_:xval})
                loss1 = sess.run(loss_,
                                       feed_dict={prob.y_: prob.yval1, prob.x_: prob.xval1})
                loss2 = sess.run(loss_,
                                 feed_dict={prob.y_: prob.yval2, prob.x_: prob.xval2})
                loss3 = sess.run(loss_,
                                 feed_dict={prob.y_: prob.yval3, prob.x_: prob.xval3})
                loss4 = sess.run(loss_,
                                 feed_dict={prob.y_: prob.yval4, prob.x_: prob.xval4})
                if np.isnan(loss):
                    raise RuntimeError('loss is NaN')
                loss_history = np.append(loss_history,loss)
                lossbest = loss_history.min()

                f1 = open(printfile, 'a+')
                f1.write('\ri={i:<6d} loss={loss:.6f} (best={best:.6f} loss1={loss1:.6f} loss2={loss2:.6f} loss3={loss3:.6f} loss4={loss4:.6f})\n'.format(i=i,loss=loss,best=lossbest,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4))
                f1.close()
                sys.stdout.write('\ri={i:<6d} loss={loss:.6f} (best={best:.6f} loss1={loss1:.6f} loss2={loss2:.6f} loss3={loss3:.6f} loss4={loss4:.6f}) \n'.format(i=i,loss=loss,best=lossbest,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4))
                sys.stdout.flush()
                if i%(50*ivl) == 0:
                    save_trainable_vars(sess,name)
                    age_of_best = len(loss_history) - loss_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > wait_time:
                        break # if it has not improved on the best answer for quite some time, then move along
            data_batch_ = sess.run(data_batch)
            y = data_batch_['y']
            x = data_batch_['x']
            sess.run(train_, feed_dict={prob.y_: y[0,:,:], prob.x_: x[0,:,:]})
        done = np.append(done,name)

        # log =  log+'\n{name} loss={loss:.6f} dB in {i} iterations'.format(name=name,loss=loss,i=i)

        # state['done'] = done
        # state['log'] = log
        save_trainable_vars(sess,name,**state)
    return sess


def do_testing_another(layer_info,prob,T,savefile):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('norms xtest:{xtest:.7f} ytest:{ytest:.7f}'.format(xtest=la.norm(prob.xval), ytest=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)
    x = np.load('xtest_uni.npy')
    y = np.load('ytest_uni.npy')

    A = prob.A
    M, N = A.shape



    layer_num=-1
    flag=0
    loss_total = np.zeros((len(layer_info), 10000))
    for name,xhat_,var_list in layer_info:
        if layer_num<0:
            if 'Linear' in name:
                flag=1
                continue
        layer_num = layer_num + 1

        nmse_ = tf.reduce_sum(tf.squared_difference(xhat_, prob.x_), axis=0)/tf.reduce_sum(tf.square(prob.x_), axis=0)

        print(name)

        loss=[]

        for i in range((10000 // prob.L)):
            loss__ = sess.run(nmse_, feed_dict={prob.y_: y[:, i * prob.L:(i + 1) * prob.L],
                                                      prob.x_: x[:, i * prob.L:(i + 1) * prob.L]})

            if loss == []:
                loss = loss__

            else:
                loss = np.concatenate([loss, loss__])

        loss_total[layer_num,:]=loss
        print(np.mean(np.asarray(loss)))
    if flag==1:
        loss_total=loss_total[:-1,:]


    xs=np.arange(loss_total.shape[0])+1
    ys_mean=np.asarray(np.mean(loss_total,axis=1))
    ys_var=np.asarray(np.var(loss_total,axis=1))
    ys_std = np.asarray(np.std(loss_total, axis=1))
    # sio.savemat('LVAMP_T_'+str(T)+'_noise_free.mat',{'layer':xs,'loss':ys_mean,'loss_var':ys_var})
    # sio.savemat('LAMP_T_' + str(T) + '_noise_free.mat', {'layer': xs, 'loss': ys_mean, 'loss_var': ys_var})
    if 'LISTA' in savefile and 'cpss' not in savefile:
        sio.savemat('LISTA_T_' + str(T) + '_noise_free_mse_uni.mat', {'layer': xs, 'loss': ys_mean,'loss_var': ys_var,'std':ys_std,'loss_total':loss_total})
    if 'LAMP-soft' in savefile:
        sio.savemat('LAMP-soft_T_' + str(T) + '_noise_free_mse_uni.mat',{'layer': xs, 'loss': ys_mean, 'loss_var': ys_var, 'std': ys_std, 'loss_total': loss_total})
    if 'LISTA-cpss' in savefile:
        sio.savemat('LISTA-cpss_T_' + str(T) + '_noise_free_mse_uni.mat',{'layer': xs, 'loss': ys_mean, 'loss_var': ys_var, 'std': ys_std, 'loss_total': loss_total})

    sess.close()


