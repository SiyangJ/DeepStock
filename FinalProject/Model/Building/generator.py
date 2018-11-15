import os
from random import shuffle
import random
import sys
import numpy as np
import pandas as pd
#import h5py
from config import FLAGS

def _PrepareData():
    data_X = pd.read_hdf(FLAGS.data_path,FLAGS.X_ID).values
    data_Y = pd.read_hdf(FLAGS.data_path,FLAGS.Y_ID).values
    
    X,y = [],[]
    total_batch = int(data_X.shape[0] / FLAGS.NUM_PER_DAY)
    for i in range(total_batch):
        X.append(data_X[i*FLAGS.NUM_PER_DAY:(i+1)*FLAGS.NUM_PER_DAY,:])
        y.append(data_Y[i*FLAGS.NUM_PER_DAY:(i+1)*FLAGS.NUM_PER_DAY,:])
    
    train_size=int(FLAGS.sep*len(X))
    split_index=[1]*train_size
    split_index.extend([0] * (len(X) - train_size))
    np.random.shuffle(split_index)

    #division all_data into train and test data
    train_X,train_y,test_X,test_y=[],[],[],[]
    for i,v in enumerate(split_index):
        if v==0:
            test_X.append(X[i])
            test_y.append(y[i])
        else:
            train_X.append(X[i])
            train_y.append(y[i])
    train_X=np.array(train_X).astype('float32')
    train_y=np.array(train_y).astype('float32')
    test_X=np.array(test_X).astype('float32')
    test_y=np.array(test_y).astype('float32')
    return train_X,train_y,test_X,test_y

_train_X,_train_Y,_test_X,_test_Y = _PrepareData()

def get_training_and_testing_generators():
    batch_size = FLAGS.batch_size
    training_generator = data_generator(istrain=True, batch_size=batch_size)
    validation_generator = data_generator(istrain=False, batch_size=batch_size)
    
    return training_generator, validation_generator

def data_generator(istrain=True,batch_size=1):
    X = _train_X if istrain else _test_X
    Y = _train_Y if istrain else _test_Y
    L = X.shape[0]
    S = FLAGS.NUM_PER_DAY
    d = FLAGS.seq_length
    # assert L>=batch_size, "Can't generate batch!"
    while True:
        
        i = np.random.randint(0,L,batch_size)
        j = np.random.randint(0,S-d+1,batch_size)
        
        ret_X,ret_Y=[],[]
        
        for b in range(batch_size):
            ret_X.append(X[i[b]][j[b]:j[b]+d])
            ret_Y.append(Y[i[b]][j[b]:j[b]+d])
            
        ret_X=np.array(ret_X).astype('float32')
        ret_Y=np.array(ret_Y).astype('float32')
        yield ret_X,ret_Y

def main():
    pass

if __name__ == '__main__':
    main()
