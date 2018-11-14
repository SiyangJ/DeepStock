import os
import sys
import tensorflow as tf
import configparser
FLAGS = tf.app.flags.FLAGS

## Configuration File Parse
CONFIG_DIR = './config.ini'
if len(sys.argv)>1 and sys.argv[1][-4:]=='.ini':
    CONFIG_DIR = sys.argv[1]
CFP = configparser.ConfigParser()
CFP.read(CONFIG_DIR)

ARGS = CFP['Default']

'''
[Default]
seq_length=388      #seq length
batch_size=4      #batch_size
feature_num=1940      #dim of a seq
y_size=50
lstm_size=64   #hidden layer units
lstm_layers=6
keep_prob=0.5
lr=0.0001        #initial learn rate
sep=0.8          #train and test sep
epoch_size=10000 #train number
save_path=/pine/scr/s/i/siyangj/DeepStock/FinalProject/Model/Building/ckpt/
data_path=/pine/scr/s/i/siyangj/DeepStock/FinalProject/Data/XY_sequence.h5
X_ID = X
Y_ID = Y
'''

################### Train Data################
tf.app.flags.DEFINE_string('data_path', ARGS['data_path'],
                           "Where to read data.")
tf.app.flags.DEFINE_integer('X_ID', ARGS.getint('X_ID'),
                           "ID of X in data.")
tf.app.flags.DEFINE_integer('Y_ID', ARGS.getint('Y_ID'),
                           "ID of Y in data.")
tf.app.flags.DEFINE_float('sep', ARGS.getfloat('sep'), 
                          "Data split ratio")

####################################################

tf.app.flags.DEFINE_integer('seq_length', ARGS.getint('seq_length'), 
                            "Sequence length of one sample")
tf.app.flags.DEFINE_integer('batch_size', ARGS.getint('batch_size'), 
                            "Number of samples per batch.")
tf.app.flags.DEFINE_integer('feature_num', ARGS.getint('feature_num'), 
                            "Number of features in one time step of one sample.")
tf.app.flags.DEFINE_integer('y_size', ARGS.getint('y_size'), 
                            "Output size in one time step of one sample.")

####################################################

tf.app.flags.DEFINE_integer('lstm_size', ARGS.getint('lstm_size'), 
                            "Hidden layer size.")
tf.app.flags.DEFINE_integer('lstm_layers', ARGS.getint('lstm_layers'), 
                            "Number of lstm hidden layers.")

############### Training and Learning rate decay ##################################
tf.app.flags.DEFINE_float('lr', ARGS.getfloat('lr'), 
                          "Start learning rate.")
tf.app.flags.DEFINE_float('keep_prob', ARGS.getfloat('keep_prob'), 
                          "Keeping probability for dropout.")

###################################################################################
tf.app.flags.DEFINE_integer('epoch_size', ARGS.getint('epoch_size'),
                            "Epochs for training.")
tf.app.flags.DEFINE_string('save_path', ARGS['save_path'], 
                           "Output folder where training logs and models are dumped.")
tf.app.flags.DEFINE_integer('random_seed', ARGS.getint('random_seed'), 
                            "Seed used to initialize rng.")
tf.app.flags.DEFINE_bool('xavier_init', ARGS.getboolean('xavier_init'),
                         "Xavier initialization or truncated normal.")

## TODO
## Need to implement
'''
tf.app.flags.DEFINE_integer('learning_rate_reduce_life', ARGS.getint('learning_rate_reduce_life'), 
                            "Number of batches until learning rate is reduced. lr *= 0.1")
tf.app.flags.DEFINE_float('learning_rate_percentage', ARGS.getfloat('learning_rate_percentage'), 
                          "Number of batches until learning rate is reduced. lr *= 0.1")
tf.app.flags.DEFINE_integer('checkpoint_period', ARGS.getint('checkpoint_period'), 
                            "Number of batches in between checkpoints")
tf.app.flags.DEFINE_string('last_trained_checkpoint', ARGS['last_trained_checkpoint'], 
                           "The model used for testing")
tf.app.flags.DEFINE_bool('restore_from_last', ARGS.getboolean('restore_from_last'), 
                         "whether start training from last trained checkpoint")
'''

def main():
    pass

if __name__ == '__main__':
    main()
