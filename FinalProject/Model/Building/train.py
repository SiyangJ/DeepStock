import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
from generator import get_training_and_testing_generators
from copy import deepcopy
from config import FLAGS

def _save_checkpoint(train_data,batch):
    td = train_data
    saver = tf.train.Saver()
    model_path = os.path.join(FLAGS.save_path,'snapshot_'+str(batch))
    
    save_path = saver.save(td.sess, model_path) #, global_step=batch)
    print("Model saved in file: %s" % save_path)   
    print ('Model saved in file: %s' % saver.last_checkpoints)

def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()

    # if the sess is restored from last checkpoint, do not need to 
    if FLAGS.restore_from_last:
        saver = tf.train.Saver()
        model_path = tf.train.latest_checkpoint(FLAGS.last_trained_checkpoint)
        print('training: restore last checkpoint from:%s' % model_path)
        saver.restore(td.sess, model_path)
    else:
        init_op = tf.global_variables_initializer()
        print('training: global variable initialization...')
        td.sess.run(init_op)

    lrval       = FLAGS.lr
    start_time  = time.time()
    done  = False
    batch = 0
    
    training_generator, testing_generator = get_training_and_testing_generators()

    while not done:
        batch += 1
        
        train_X, train_Y = next(training_generator)

        feed_dict = {td.X_variable : train_X, 
                     td.Y_variable : train_Y, 
                     td.learning_rate : lrval}

        cur_time = time.ctime()
        
        if batch % 10 == 0:
            ops = [td.train_minimize, td.loss, summaries,] 
            [_, loss, summary_vis] = td.sess.run(ops, feed_dict=feed_dict)

            print('[%25s], iter [%4d], Lr[%1.8f] ,loss[%3.10f]'
                            % (cur_time, batch, lrval, loss) )
            
            # Update learning rate
            if batch % FLAGS.learning_rate_reduce_life == 0:
                lrval *= FLAGS.learning_rate_percentage

            td.summary_writer.add_summary(summary_vis, batch)
            
            ############## Editted Nov 04 by Siyang Jing
            ############## Try to add validation loss
            
            val_X, val_Y = next(testing_generator)
            
            ############## Nov 14 working notes
            ## Severe bug
            ## How to enable different batch size???
            ## and different seq_length???
            
            val_feed_dict = {td.X_variable : val_X, 
                             td.Y_variable : val_Y}
            
            val_ops = [td.loss, summaries,] 
            [val_loss, val_summary] = td.sess.run(val_ops, feed_dict=val_feed_dict)

            print('[%25s], validation: iter [%4d], loss[%3.10f]'
                            % (cur_time, batch, val_loss) )
            
            td.val_sum_writer.add_summary(val_summary, batch)
            
        else:
            ops = [td.train_minimize, td.loss] 
            [_, loss] = td.sess.run(ops, feed_dict=feed_dict)
            
        if batch % FLAGS.checkpoint_period == 0:
            _save_checkpoint(td, batch)

        if batch  >= FLAGS.epoch_size:
            done = True

    _save_checkpoint(td, batch)
    print('Finished training!')

if __name__ == '__main__':
	import time
	time.ctime() # 'Mon Oct 18 13:35:29 2010'
	time.strftime('%l:%M%p %Z on %b %d, %Y') # ' 1:36PM EDT on Oct 18, 2010'
	time.strftime('%l:%M%p %z on %b %d, %Y') # ' 1:36PM EST on Oct 18, 2010'