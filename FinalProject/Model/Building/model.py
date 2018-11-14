import tensorflow as tf
import numpy as np

from config import FLAGS

DTYPE = tf.float32

def _weight_variable( scope_name, name, shape, stddev=0.01):
    if FLAGS.xavier_init:
        return tf.get_variable(name, shape, DTYPE, initializer=tf.contrib.layers.xavier_initializer())
    else:
        return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=stddev))



def _bias_variable( scope_name, name, shape, constant_value=0.01):
    bias = tf.get_variable(name, shape, DTYPE, tf.constant_initializer(constant_value))
        return bias
'''
#general W
def W_var(in_dim,out_dim):
    return tf.Variable(tf.random_normal([in_dim,out_dim]),tf.float32)

#general b
def b_var(out_dim):
    return tf.Variable(tf.random_normal([out_dim,]),tf.float32)
'''

#lstm : 64 lstm_size, 2 lstm_layer

def lstm_cell(config,keep_prob):
    temp=tf.contrib.rnn.BasicLSTMCell(config.lstm_size,reuse=False)
    drop = tf.nn.rnn_cell.DropoutWrapper(temp, output_keep_prob=keep_prob)
    return drop
        
def output_layers(config,output_lstm):
    in_size=output_lstm.get_shape()[-1].value
    #output_lstm=output_lstm[:,-1,:]
    output_lstm=tf.reshape(output_lstm,[-1,in_size])
    W=W_var(in_size,config.y_size)
    b=b_var(config.y_size)
    ###################################
    ## TODO
    ## Should change to tf.nn.xw_plus_b
    #output_final=tf.add(tf.matmul(output_lstm,W),b)
    output_final=tf.nn.xw_plus_b(output_lstm,W,b)
    output_final = tf.reshape(output_final,[config.batch_size,config.seq_length,config.y_size])
    return output_final

def _lstm_cell(layer_size,keep_prob):
    temp=tf.contrib.rnn.BasicLSTMCell(layer_size,reuse=False)
    drop=tf.nn.rnn_cell.DropoutWrapper(temp, output_keep_prob=keep_prob)
    return drop

def add_lstm_layers(scope_name,prev_layer,num_layers,layer_size,keep_prob):
    with tf.variable_scope(scope_name) as scope:
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [_lstm_cell(layer_size,keep_prod) for _ in range(config.lstm_layers)])
        
        ## Need to think
        batch_size = prev_layer.shape[0].value
        initial_state = stacked_lstm.zero_state(batch_size,DTYPE)
        outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, prev_layer, 
              initial_state=initial_state)
        return outputs,final_state

def add_output_layer(scope_name,prev_layer,output_size):
    ##### Possible bug here
    '''
    prev_layer should be output of dynamic_rnn
    shape should be:
    [batch_size, seq_length, layer_size]
    First transform to [batch_size * seq_length, layer_size]
    Then use xw_plus_b to layer_size -> output_size
    Therefore W should be [layer_size, output_size]
              b should be [output_size]    
    '''
    
    with tf.variable_scope(scope_name) as scope:
        in_size=prev_layer.get_shape()[-1].value
        prev_layer=tf.reshape(prev_layer,[-1,in_size])
        W=W_var(in_size,config.y_size)
        b=b_var(config.y_size)
        ###################################
        ## TODO
        ## Should change to tf.nn.xw_plus_b
        #output_final=tf.add(tf.matmul(output_lstm,W),b)
        output_final=tf.nn.xw_plus_b(prev_layer,W,b)
        output_final = tf.reshape(output_final,
                                  [config.batch_size,config.seq_length,config.y_size])
        return output_final
    
    
def cal_loss(pred,truth):
    loss=tf.reduce_mean(tf.square(pred-truth))
    return loss

def create_model(train_phase=True):
    
    
def create_optimizers(train_loss):
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    train_opti = tf.train.AdamOptimizer(learning_rate)
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    # Disable var_list, train all variables
    # var_list = _get_var_list()
    train_minimize = train_opti.minimize(train_loss, name='loss_minimize', global_step=global_step)
    return train_minimize, learning_rate, global_step