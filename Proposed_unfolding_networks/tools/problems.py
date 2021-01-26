#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf
import os
from collections import OrderedDict
class Generator(object):
    def __init__(self,A,L,**kwargs):
        self.A = A
        self.L=L
        M,N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.float32,(N,L),name='x' )
        self.y_ = tf.placeholder( tf.float32,(M,L),name='y' )

class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_ ) )
    def get_batch(self):
        dataset_info_path='train_info.txt'#train_info.txt
        with open(dataset_info_path,'r') as dataset_info:
            input_info=OrderedDict()
            for line in dataset_info.readlines():
                items=line.split(',')
                try:
                    input_info[items[0]]=[int(dim) for dim in items[1:]]
                except:
                    input_info[items[0]]=[]
        def _parse_tf_example(example_proto):
            features=dict([(key,tf.FixedLenFeature([],tf.string)) for key,_ in input_info.items()])
            parsed_features=tf.parse_single_example(example_proto,features=features)
            return [tf.reshape(tf.decode_raw(parsed_features[key],tf.float32),value) for key,value in input_info.items()]

        dataset_path='train.tfrecords'
        dataset=tf.data.TFRecordDataset(dataset_path)
        dataset=dataset.map(_parse_tf_example)
        dataset=dataset.repeat()
        dataset=dataset.batch(1)
        iterator=dataset.make_initializable_iterator()
        data_batch=iterator.get_next()
        keys=list(input_info.keys())
        data_batch=dict([(keys[i],data_batch[i]) for i in range(len(keys))])
        return data_batch,iterator.initializer

class NumpyGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)

    def __call__(self,sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches,self.nsubprocs)
        
    def get_batch(self):
        dataset_info_path='train_info.txt'#train_info.txt
        with open(dataset_info_path,'r') as dataset_info:
            input_info=OrderedDict()
            for line in dataset_info.readlines():
                items=line.split(',')
                try:
                    input_info[items[0]]=[int(dim) for dim in items[1:]]
                except:
                    input_info[items[0]]=[]
        def _parse_tf_example(example_proto):
            features=dict([(key,tf.FixedLenFeature([],tf.string)) for key,_ in input_info.items()])
            parsed_features=tf.parse_single_example(example_proto,features=features)
            return [tf.reshape(tf.decode_raw(parsed_features[key],tf.float32),value) for key,value in input_info.items()]

        dataset_path='train.tfrecords'
        dataset=tf.data.TFRecordDataset(dataset_path)
        dataset=dataset.map(_parse_tf_example)
        dataset=dataset.repeat()
        dataset=dataset.batch(1)
        iterator=dataset.make_initializable_iterator()
        data_batch=iterator.get_next()
        keys=list(input_info.keys())
        data_batch=dict([(keys[i],data_batch[i]) for i in range(len(keys))])
        return data_batch,iterator.initializer


def generate(N,L,pnz):
    return ((np.random.uniform(0, 1, (N, L)) < pnz) * np.random.normal(0, 1, (N, L))).astype(np.float32)

def generate_k(N,L,k):
    bernoulli_ = np.zeros([N, L])
    for i in range(L):
        d1=np.zeros(N-k)
        d2=np.ones(k)
        d=np.concatenate([d1,d2])
        np.random.shuffle(d)
        bernoulli_[:,i]=d
    return (bernoulli_*np.random.normal(0, 1, (N, L))).astype(np.float32)

def generate_uni(N,L):
    bernoulli_ = np.zeros([N, L])
    k=np.random.randint(1,100,1)[0]
    for i in range(L):
        d1=np.zeros(N-k)
        d2=np.ones(k)
        d=np.concatenate([d1,d2])
        np.random.shuffle(d)
        bernoulli_[:,i]=d
    return (bernoulli_*np.random.normal(0, 1, (N, L))).astype(np.float32)


def cond(tmp):
    return tf.less(tf.squeeze(tf.abs(tmp)),0.1)

def body(tmp):
    tmp=tf.random_normal([1])
    return tmp

def bernoulli_gaussian_trial(A,M=250,N=500,L=1000,is_train=False):
  
    A_ = tf.constant(A,name='A')
    prob = TFGenerator(A=A,L=L,A_=A_)
    prob.name = 'Bernoulli-Gaussian, random A'

    if is_train:
       if not os.path.exists(os.path.join(os.getcwd(), 'train.tfrecords')):
           print('preparing training datasets\n')
           f1 = open('prepare_data.txt', 'w')
           f1.close()
           def bytes_feature(value):
             return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
           writer=tf.python_io.TFRecordWriter(os.path.join(os.getcwd(),'train.tfrecords'))
           for i in range(30000):
             feature={}
             if i%100==0:
                 print(i)
                 f1 = open('prepare_data.txt', 'a+')
                 f1.write('%d\n'%(i))
                 f1.close()
             np.random.seed()
             prob.xval = generate(N, L,0.1).astype(np.float32)
             prob.xval = prob.xval / (np.sqrt(np.sum(np.square(prob.xval), axis=0, keepdims=True)))
             prob.yval = np.matmul(A, prob.xval)

             feature['y']=bytes_feature(prob.yval.tostring())
             feature['x'] = bytes_feature(prob.xval.tostring())
    
             example=tf.train.Example(features=tf.train.Features(feature=feature))
             writer.write(example.SerializeToString())
           writer.close()
           with open(os.path.join(os.getcwd(),'train_info.txt'),'w') as dataset_info:
             dataset_info.write('y'+','+str(M)+','+str(L)+'\n')
             dataset_info.write('x' + ',' + str(N) + ',' + str(L) + '\n')
       if not os.path.exists(os.path.join(os.getcwd(), 'xval.npy')):
           prob.xval = generate(N, L,0.1).astype(np.float32)
           prob.xval = prob.xval / (np.sqrt(np.sum(np.square(prob.xval), axis=0, keepdims=True)))
           prob.yval = np.matmul(A, prob.xval)
           np.save('xval.npy', prob.xval)
           np.save('yval.npy', prob.yval)

       prob.xval1 = generate_k(N,L,20).astype(np.float32)
       prob.xval1 = prob.xval1 / (np.sqrt(np.sum(np.square(prob.xval1), axis=0, keepdims=True)))
       prob.yval1 = np.matmul(A,prob.xval1)
       prob.xval2 = generate_k(N, L,40).astype(np.float32)
       prob.xval2 = prob.xval2 / (np.sqrt(np.sum(np.square(prob.xval2), axis=0, keepdims=True)))
       prob.yval2 = np.matmul(A, prob.xval2)
       prob.xval3 = generate_k(N, L,60).astype(np.float32)
       prob.xval3 = prob.xval3 / (np.sqrt(np.sum(np.square(prob.xval3), axis=0, keepdims=True)))
       prob.yval3 = np.matmul(A, prob.xval3)
       prob.xval4 = generate_k(N, L,80).astype(np.float32)
       prob.xval4 = prob.xval4 / (np.sqrt(np.sum(np.square(prob.xval4), axis=0, keepdims=True)))
       prob.yval4 = np.matmul(A, prob.xval4)

    else:
        if not os.path.exists(os.path.join(os.getcwd(), 'xtest_uni.npy')):
            print('preparing testing datasets\n')
            prob.xval = generate_uni(N, 10000).astype(np.float32)
            prob.xval=prob.xval/(np.sqrt (np.sum (np.square (prob.xval), axis=0, keepdims=True)))
            prob.yval = np.matmul(A, prob.xval)
            np.save('xtest_uni.npy', prob.xval)
            np.save('ytest_uni.npy', prob.yval)


    prob.xinit = generate(N,L,0.1).astype(np.float32)
    prob.xinit = prob.xinit / (np.sqrt(np.sum(np.square(prob.xinit), axis=0, keepdims=True)))
    prob.yinit = np.matmul(A,prob.xinit)
    prob.xval = generate(N, L,0.1).astype(np.float32)
    prob.xval = prob.xval / (np.sqrt(np.sum(np.square(prob.xval), axis=0, keepdims=True)))
    prob.yval = np.matmul(A, prob.xval)

    return prob

