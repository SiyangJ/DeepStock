import tensorflow as tf
from model import create_model, create_optimizers
from train import train_model
import random
import numpy as np
import os
# from generator import get_training_and_testing_generators
from predict import predict
import generator
from copy import deepcopy
from config import FLAGS



def prepare_dirs(delete_train_dir=False):
	## Theoretically, should do nothing. It's just test.
	pass
	'''    
	# Create checkpoint dir (do not delete anything)
	if not tf.gfile.Exists(FLAGS.save_path):
		tf.gfile.MakeDirs(FLAGS.save_path)
	
	# Cleanup train dir
	if delete_train_dir:
		if tf.gfile.Exists(FLAGS.save_path):
			tf.gfile.DeleteRecursively(FLAGS.save_path)
		tf.gfile.MakeDirs(FLAGS.save_path)
	'''

def setup_tensorflow():
	
	config = tf.ConfigProto()
	sess = tf.Session(config=config)

	# Initialize rng with a deterministic seed
	with sess.graph.as_default():
		tf.set_random_seed(FLAGS.random_seed)
		
	random.seed(FLAGS.random_seed)
	np.random.seed(FLAGS.random_seed)

	tf.gfile.MkDir('%s/test_log' % (FLAGS.save_path,))
	summary_writer = tf.summary.FileWriter('%s/test_log' % (FLAGS.save_path,), sess.graph)
	return sess, summary_writer



class TestData(object):
	def __init__(self, dictionary):
		self.__dict__.update(dictionary)




def test():
	prepare_dirs(delete_train_dir=False)
	sess, summary_writer = setup_tensorflow()

	(X_variable, Y_variable,
     pred, loss, final_loss,
     gene_vars) = create_model(train_phase=True)

	saver = tf.train.Saver()
	## Load from the model save path instead of the last_trained_checkpoint, 
	## which is supposed to be the save_path of last training
	model_path = tf.train.latest_checkpoint(FLAGS.save_path)
	print('saver restore from:%s' % model_path)
	saver.restore(sess, model_path)
	
	test_data = TestData(locals())
	X_test, Y_test, array_pred = predict(test_data)
	return X_test, Y_test, array_pred


def main(argv=None):
	print ('>> start testing phase...')
	test()


if __name__ == '__main__':
	tf.app.run()