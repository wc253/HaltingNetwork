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
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
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

def setup_training(layer_info,prob,tao=0.22, trinit=1e-3,refinements=(.5,.1,.01)):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    init_beta=32
    for xhat1_,xhat2_,ponder_cost_,num_units_,xhats_,halting_distribution_ in layer_info:
        loss_care=tf.reduce_mean(tf.reduce_sum(tf.squared_difference(xhat1_, prob.x_),axis=0))
        t_vars=tf.trainable_variables()

        h_vars = [var for var in t_vars if 'h'  in var.name]
        net_vars = [var for var in t_vars if 'h' not in var.name]



        decay_rate=np.power(1/init_beta,1/15)
        # trinit = 1e-3
        # lr_decay_rate = np.power(1e-2,1/6)
        count = 0
        for i in range(6):

            # beta=init_beta*np.power(decay_rate,i)
            if i<2:
                loss = 0.1*(tf.reduce_mean(xhat2_+tao*tf.to_float(num_units_)))+loss_care
            elif i<4:
                loss = 0.05 * (tf.reduce_mean(xhat2_ + tao * tf.to_float(num_units_))) + loss_care
            else:
                loss = 0.01 * (tf.reduce_mean(xhat2_ + tao * tf.to_float(num_units_))) + loss_care
            # for j in range(2):

            train1_ = tf.train.AdamOptimizer(1e-5).minimize(loss,var_list=net_vars)

            if i<1:
                train2_ = tf.train.AdamOptimizer(tr_ ).minimize(loss, var_list=h_vars)
                train_=train2_
            else:
                train2_ = tf.train.AdamOptimizer(tr_*0.1).minimize(loss, var_list=h_vars)
                train_=tf.group(train2_,train1_)
            training_stages.append(('state'+str(i), xhat1_, xhat2_, loss, loss_care, num_units_,train_,
                                        xhats_, halting_distribution_,'LISTA_T_10_tao_'+str(tao)+'_count_'+str(i)+'_try3.npz'))
            count = count + 1
            # for j in range(2):
            #     train_ = tf.train.AdamOptimizer(1e-5*(0.1**(j))).minimize(loss_care,var_list=net_vars)
            #     # train_ = tf.train.AdamOptimizer(tr_).minimize(loss)
            #
            #     training_stages.append(('beta' + str(beta) + 'lr_rate' + str(1e-5*(0.1**(j)))+'net_only' , xhat1_, xhat2_, loss_care, loss_care, num_units_,train_,
            #                             xhats_, halting_distribution_,'LISTA_T_10_tao_' + str(tao) + '_beta_' + str(beta) + '.npz'))




        # except Exception:
        #     fm=refinements
        #     train_ = tf.train.AdamOptimizer(tr_ * fm).minimize(loss,var_list=h_vars)
        #     training_stages.append((
        #                            str(fm), xhat1_, xhat2_, loss, loss_care,train_, num_units_, xhats_,
        #                            halting_distribution_))
    return training_stages



def do_training(training_stages,prob,printfile,restorefile='LISTA_T_10.npz',ivl=100,maxit=40000,better_wait=10000):
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
    # t_vars = tf.global_variables()
    # h_vars = [var for var in t_vars if 'beta' not in var.name]
    # h_vars = [var for var in h_vars if 'Adam' not in var.name]
    # tf.train.Saver(var_list=h_vars).restore(sess, 'model.ckpt0.1-85000')
    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)
    #
    done=state.get('done',[])


    lossbest = np.inf
    for  name,xhat1_,xhat2_,loss_,loss_care_,num_units_,train_,xhats_,halting_distribution_,savefile in training_stages:
        # if name in done:
        #    print('Already did ' + name + '. Skipping.')
        #    continue
        print('\n')
        print(name)
        f1 = open(printfile, 'a+')
        f1.write(name)
        f1.write('\n')
        f1.close()
        xval = np.load('xval.npy')
        yval = np.load('yval.npy')
        loss_history = []
        # loss_history = np.append(loss_history, lossbest)
        for i in range(maxit+1):
            if i%ivl == 0:

                loss,losscare,num= sess.run([loss_,loss_care_,tf.reduce_sum(num_units_)],feed_dict={prob.y_:yval,prob.x_:xval})

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
                sys.stdout.write('\ri={i:<6d} loss={loss:.6f} (best={best:.6f}) losscare={losscare:.6f} num={num:.6f} num1={num1:.6f} num2={num2:.6f} num3={num3:.6f} num4={num4:.6f} loss1={loss1:.6f} loss2={loss2:.6f} loss3={loss3:.6f} loss4={loss4:.6f}'.format(i=i,loss=loss,best=lossbest,losscare=losscare,num=num/prob.L,num1=num1/prob.L,num2=num2/prob.L,num3=num3/prob.L,num4=num4/prob.L,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4))
                sys.stdout.flush()
                if i%(50*ivl) == 0:

                    age_of_best = len(loss_history) - loss_history.argmin()-1 # how long ago was the best nmse
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
                    save_trainable_vars(sess, savefile)
                    print('')
                    # tf.train.Saver(tf.global_variables(), max_to_keep=10).save(sess,'./model.ckpt'+name,global_step=i)
            y, x = prob(sess)
            loss__ = sess.run(loss_, feed_dict={prob.y_: y, prob.x_: x})
            if loss__ < lossbest or abs((loss__-lossbest)/lossbest)<0.5:
                lossbest = loss__
                sess.run(train_, feed_dict={prob.y_: y, prob.x_: x})
        done = np.append(done,name)

    

        state['done'] = done

        save_trainable_vars(sess,savefile,**state)
    return sess

def do_testing(training_stages,prob,printfile,savefile,last_stage='0.001'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('norms xtest:{xtest:.7f} ytest:{ytest:.7f}'.format(xtest=la.norm(prob.xval), ytest=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    for name,xhat1_,xhat2_,loss_,loss_care_,train_,num_units_,xhats_,halting_distribution_  in training_stages:
        if name != last_stage:
            print( name )
            continue
        x=prob.xval
        y=prob.yval
        xtilde,num_units,tmp1,loss=sess.run([xhat1_,tf.reduce_sum(num_units_),num_units_,loss_care_], feed_dict={prob.y_:y,prob.x_:x})
        num_units1=sess.run(tf.reduce_sum(num_units_), feed_dict={prob.y_:prob.yval1,prob.x_:prob.xval1})
        num_units2=sess.run(tf.reduce_sum(num_units_), feed_dict={prob.y_:prob.yval2,prob.x_:prob.xval2})
        num_units3=sess.run(tf.reduce_sum(num_units_), feed_dict={prob.y_:prob.yval3,prob.x_:prob.xval3})
        num_units4=sess.run(tf.reduce_sum(num_units_), feed_dict={prob.y_:prob.yval4,prob.x_:prob.xval4})
        strict,loose=Accuracy(xtilde,x)

        f1=open(printfile,'w')
        f1.write('strict accuracy is %.5f\n'%(strict))
        f1.write('loose accuracy is %.5f\n' % (loose))
        f1.write('loss is %.5f\n' % (loss))
        f1.write('N is %.5f\n' % (num_units/prob.L))
        f1.write('N1 is %.5f\n' % (num_units1/prob.L))
        f1.write('N2 is %.5f\n' % (num_units2/prob.L))
        f1.write('N3 is %.5f\n' % (num_units3/prob.L))
        f1.write('N4 is %.5f\n' % (num_units4/prob.L))
        f1.close()
        print('strict accuracy is %.5f'%(strict))
        print('loose accuracy is %.5f' % (loose))
        print('loss is %.5f' % (loss))
        print('N is %.5f' % (num_units/prob.L))
        print('N1 is %.5f' % (num_units1/prob.L))
        print('N1 is %.5f' % (num_units2/prob.L))
        print('N1 is %.5f' % (num_units3/prob.L))
        print('N4 is %.5f' % (num_units4/prob.L))
    sess.close()

def do_testing_another(layer_info,prob,tao,printfile,savefile,last_stage='0.001'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('norms xtest:{xtest:.7f} ytest:{ytest:.7f}'.format(xtest=la.norm(prob.xval), ytest=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    for xhat1_, xhat2_, ponder_cost_, num_units_, xhats_, halting_distribution_ in layer_info:
        loss_care_ = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(xhat1_, prob.x_), axis=0))
        loss_care_1=tf.reduce_sum(tf.squared_difference(xhat1_, prob.x_), axis=0)
        # if name != last_stage:
        #     print( name )
        #     continue
        x=np.load('xval.npy')
        y=np.load('yval.npy')
        x1 = np.load('xval1.npy')
        y1 = np.load('yval1.npy')
        x2 = np.load('xval2.npy')
        y2 = np.load('yval2.npy')
        x3 = np.load('xval3.npy')
        y3 = np.load('yval3.npy')
        x4 = np.load('xval4.npy')
        y4 = np.load('yval4.npy')
        xtilde,num_units,tmp1,loss=sess.run([xhat1_,tf.reduce_sum(num_units_),num_units_,loss_care_], feed_dict={prob.y_:y,prob.x_:x})
        num_units1,loss1=sess.run([tf.reduce_sum(num_units_),loss_care_1], feed_dict={prob.y_:y1,prob.x_:x1})
        num_units2,loss2=sess.run([tf.reduce_sum(num_units_),loss_care_1], feed_dict={prob.y_:y2,prob.x_:x2})
        num_units3,loss3=sess.run([tf.reduce_sum(num_units_),loss_care_1], feed_dict={prob.y_:y3,prob.x_:x3})
        num_units4,loss4=sess.run([tf.reduce_sum(num_units_),loss_care_1], feed_dict={prob.y_:y4,prob.x_:x4})
        sio.savemat('h', {'n1': num_units1 / prob.L, 'l1': loss1, 'n2': num_units2 / prob.L, 'l2': loss2,
                            'n3': num_units3 / prob.L, 'l3': loss3, 'n4': num_units4 / prob.L, 'l4': loss4})
        # strict,loose=Accuracy(xtilde,x)
        error=np.sum(np.where(loss>tao,np.ones_like(loss),np.zeros_like(loss)))/prob.L
        error1 = np.sum(np.where(loss1 > tao, np.ones_like(loss1), np.zeros_like(loss1)))/prob.L
        error2 = np.sum(np.where(loss2 > tao, np.ones_like(loss2), np.zeros_like(loss2))) / prob.L
        error3 = np.sum(np.where(loss3 > tao, np.ones_like(loss3), np.zeros_like(loss3))) / prob.L
        error4 = np.sum(np.where(loss4 > tao, np.ones_like(loss4), np.zeros_like(loss4))) / prob.L
        print('%.5f  %.5f  %.5f  %.5f  %.5f  '%(error,error1,error2,error3,error4))

        f1=open(printfile,'w')
        # f1.write('strict accuracy is %.5f\n'%(strict))
        # f1.write('loose accuracy is %.5f\n' % (loose))
        f1.write('loss is %.5f\n' % (loss))
        f1.write('N is %.5f\n' % (num_units/prob.L))
        f1.write('N1 is %.5f\n' % (num_units1/prob.L))
        f1.write('N2 is %.5f\n' % (num_units2/prob.L))
        f1.write('N3 is %.5f\n' % (num_units3/prob.L))
        f1.write('N4 is %.5f\n' % (num_units4/prob.L))
        f1.close()

        # print('strict accuracy is %.5f'%(strict))
        # print('loose accuracy is %.5f' % (loose))
        print('loss is %.5f' % (loss))
        print('N is %.5f' % (num_units/prob.L))
        print('N1 is %.5f' % (num_units1/prob.L))
        print('N2 is %.5f' % (num_units2/prob.L))
        print('N3 is %.5f' % (num_units3/prob.L))
        print('N4 is %.5f' % (num_units4/prob.L))
    sess.close()
def Accuracy(xtilde,x):
    l=xtilde.shape[1]
    x_t=np.where(x!=0,np.ones_like(x),np.zeros_like(x))
    k=np.sum(x_t,axis=0).astype(int)
    score_total_for_strict=0
    score_total_for_loose=0
    for i in range(l):
        score=0
        # label=np.argpartition(np.abs(x[:,i]),-k[i])[-k[i]:]
        # pred=np.argpartition(np.abs(xtilde[:,i]),-k[i])[-k[i]:]
        label=np.argsort(np.abs(x[:,i]))[-k[i]:]
        pred=np.argsort(np.abs(xtilde[:,i]))[-k[i]:]
        for m in range(k[i]):
            for n in range(k[i]):
                score=score+(label[m]==pred[n])
        if score==k[i]:
            score_total_for_strict=score_total_for_strict+1
        score_total_for_loose=score_total_for_loose+score/k[i]
    return score_total_for_strict/l,score_total_for_loose/l

def do_testing1(training_stages,prob,savefile,tao=None,beta=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('norms xtest:{xtest:.7f} ytest:{ytest:.7f}'.format(xtest=la.norm(prob.xval), ytest=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    for name,xhat1_,xhat2_,loss1_,loss2_,loss1_original_,train_,num_units_,xhats_,halting_distribution_  in training_stages:
        if name != '0.001':
            print( name )
            continue
        A = np.load('A.npy')
        x = problems.generate(100, 1000).astype(np.float32)
        y = np.matmul(A, x)
        xtilde, loss, loss_original,num_units= sess.run([xhat1_, loss1_, loss1_original_,tf.reduce_sum(num_units_)], feed_dict={prob.y_: y, prob.x_: x})
        # loss_original = np.concatenate(loss_original, axis=0)
        x1 = problems.generate1(100, 1000).astype(np.float32)
        y1 = np.matmul(A, x1)
        loss_original1, num_units1 = sess.run([ loss1_original_, tf.reduce_sum(num_units_)],
                                                          feed_dict={prob.y_: y1, prob.x_: x1})
        x2 = problems.generate2(100, 1000).astype(np.float32)
        y2 = np.matmul(A, x2)
        loss_original2, num_units2 = sess.run([loss1_original_, tf.reduce_sum(num_units_)],
                                              feed_dict={prob.y_: y2, prob.x_: x2})
        x3 = problems.generate3(100, 1000).astype(np.float32)
        y3 = np.matmul(A, x3)
        loss_original3, num_units3 = sess.run([loss1_original_, tf.reduce_sum(num_units_)],
                                              feed_dict={prob.y_: y3, prob.x_: x3})
        x4 = problems.generate4(100, 1000).astype(np.float32)
        y4 = np.matmul(A, x4)
        loss_original4, num_units4 = sess.run([loss1_original_, tf.reduce_sum(num_units_)],
                                              feed_dict={prob.y_: y4, prob.x_: x4})


        if tao is not None and beta is not None:
            sio.savemat('T_10_act_tao_' + str(tao) +'_beta_'+str(beta)+ '.mat',
                    {'loss': loss_original, 'num': (num_units / prob.L),'loss1': loss_original1, 'num1': (num_units1 / prob.L),'loss2': loss_original2, 'num2': (num_units2 / prob.L),'loss3': loss_original3, 'num3': (num_units3 / prob.L),'loss4': loss_original4, 'num4': (num_units4 / prob.L)})

        print('loss is %.5f' % (loss))
        print('N is %.5f' % (num_units/prob.L))
