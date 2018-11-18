from config import FLAGS
import os.path
import numpy as np
import math
import h5py
import time
import tensorflow as tf
from generator import get_training_and_testing_generators

## A very crude version.
## Just for experiment.
def predict(td):
    preds = []
    X_test = []
    Y_test = []
    start_time  = time.time()
    _, test_generator = get_training_and_testing_generators()
    ## TODO
    ## Just for the purpose of testing.
    patch_num = 20
    print('>> begin predicting for each patch')
    for _i in  range(patch_num):
        _test_X, _test_Y = next(test_generator)
        feed_dict = { td.X_variable : _test_X, 
                     td.Y_variable : _test_Y}
        ops = [td.pred,]
        [pred,] = td.sess.run(ops, feed_dict=feed_dict)
        preds.append(pred)
        X_test.append(_test_X)
        Y_test.append(_test_Y)
        
    array_pred = np.asarray(preds)
    #print '>> begin vote in overlapped patch..'
    #seg_res, possibilty_map = vote_overlapped_patch(patches_pred, index, d,h,w)
    # seconds
    elapsed = int(time.time() - start_time)

    print('Predict complete, cost [%3d] seconds' % (elapsed))

    return X_test, Y_test, array_pred

def main():
    predict(None)

if __name__ == '__main__':
    main()
    