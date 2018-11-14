import tensorflow as tf
from model import create_model, create_optimizers
from train import train_model
import random
import numpy as np
import os
from generator import get_training_and_testing_generators
from copy import deepcopy
from config import FLAGS


def prepare_dirs(delete_train_dir=False):
	# Create checkpoint dir (do not delete anything)
	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
	
	# Cleanup train dir
	if delete_train_dir:
		if tf.gfile.Exists(FLAGS.checkpoint_dir):
			tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def setup_tensorflow():
	
	config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=config)

	# Initialize rng with a deterministic seed
	with sess.graph.as_default():
		tf.set_random_seed(FLAGS.random_seed)
		
	random.seed(FLAGS.random_seed)
	np.random.seed(FLAGS.random_seed)

	## Editted by Siyang Jing on Nov 4
	## Try to add validation summary writer
	tf.gfile.MkDir('%s/training_log' % (FLAGS.checkpoint_dir,))
	tf.gfile.MkDir('%s/validation_log' % (FLAGS.checkpoint_dir,))
	summary_writer = tf.summary.FileWriter('%s/training_log' % (FLAGS.checkpoint_dir,), sess.graph_def)
	val_sum_writer = tf.summary.FileWriter('%s/validation_log' % (FLAGS.checkpoint_dir,), sess.graph_def)

	return sess, summary_writer, val_sum_writer

class TrainData(object):
	def __init__(self, dictionary):
		self.__dict__.update(dictionary)


def train():
	prepare_dirs(delete_train_dir=True)
	sess, summary_writer, val_sum_writer = setup_tensorflow()


	(tf_t1_input, tf_t2_input, tf_label, 
            aux1_pred, aux2_pred, main_pred,
            aux1_loss, aux2_loss, main_loss, 
            loss, gene_vars, main_possibility) = create_model(train_phase=True)

	train_minimize, learning_rate, global_step = create_optimizers(loss)

	train_data = TrainData(locals())
	train_model(train_data)
	

	

def main(argv=None):
	train()




if __name__ == '__main__':
	tf.app.run()