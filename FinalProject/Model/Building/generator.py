import os
from random import shuffle
import random
import sys
import numpy as np
from config import FLAGS
import h5py
import math
from util.utils import pickle_dump, pickle_load
from util.utils import parse_patch_size
from util.utils import load_nifti, save_nifti
def get_training_and_testing_generators(hdf5_train_list_file=FLAGS.hdf5_train_list_path,
										hdf5_validation_list_file=FLAGS.hdf5_validation_list_path,
									 	batch_size=FLAGS.batch_size,
									    overwrite_split=True):
	'''
	after split the training and testing , the split will be stored as pkl.
	overwrite_split: True is to overwrite the pkl file, which states what the 
	trainging and testing hdf5 file are
	'''

	training_list, validation_list = get_validation_split( hdf5_train_list_file, hdf5_validation_list_file,
													    overwrite_split=overwrite_split)
	
	training_generator = data_random_generator(training_list, batch_size=batch_size)
	validation_generator = data_random_generator(validation_list, batch_size=batch_size)
	
	return training_generator, validation_generator

def data_random_generator(hdf5_list, 
							patch_size_str=FLAGS.patch_size_str, 
							batch_size=1,
							extract_batches_one_image=FLAGS.batches_one_image):
	'''
		randome crop patches from volume images. hdf5_data contains several volume datas.
		hdf5_data: num*1*Depth*H*W
		random strategy: 0. each time extract a batch_size from 
	'''
	while True:
		shuffle(hdf5_list)
		patch_size = parse_patch_size(patch_size_str)
		for _local_file in hdf5_list:
			print ('generate random patch from file %s ...' % _local_file)
			file_handle   = h5py.File(_local_file, 'r')
			img_data_t1 = file_handle['t1data']
			img_data_t2 = file_handle['t2data']
			img_label = file_handle['label']
			
			
			img_data_t1 = np.asarray(img_data_t1, 'float')
			img_data_t2 = np.asarray(img_data_t2, 'float')
			img_label = np.asarray(img_label, 'float')
			print '>> img_data_t1.shape=',img_data_t1.shape

			img_data_t1 = img_data_t1[np.newaxis, np.newaxis, ... ]
			img_data_t2 = img_data_t2[np.newaxis, np.newaxis, ... ]
			img_label = img_label[np.newaxis, np.newaxis, ... ]
			
			assert len(img_label.shape)==5, 'label must be in 5 dimentional..'
			assert len(img_data_t1.shape)==5, ' dimentional of volume image data must be 5..'

			d = img_data_t1.shape[-3]
			h = img_data_t1.shape[-2]
			w = img_data_t1.shape[-1]
			crop_pad = FLAGS.training_crop_pad

			print '>> crop center [%d:-%d]... d=%d,h=%d,w=%d' %(crop_pad,crop_pad,d,h,w,)
			# how many times that we extract a batch of patches in one image
			for _ in xrange(extract_batches_one_image):
				x1_list = list()
				x2_list = list()
				# dm_list = list()
				y_list = list()
				for _ in xrange(batch_size):
					d_ran = random.randrange(crop_pad,d - patch_size[0]-crop_pad)
					h_ran = random.randrange(crop_pad,h - patch_size[1]-crop_pad)
					w_ran = random.randrange(crop_pad,w - patch_size[2]-crop_pad)

					# print ('>> random crop from (%s,%s,%s) ' % (d_ran, h_ran, w_ran))
					random_crop_data_t1 = img_data_t1[0,0,d_ran : d_ran+patch_size[0], 
											h_ran : h_ran+patch_size[1],
											w_ran: w_ran+patch_size[2]]
					random_crop_data_t1 = np.asarray(random_crop_data_t1)

					random_crop_data_t2 = img_data_t2[0,0,d_ran : d_ran+patch_size[0], 
											h_ran : h_ran+patch_size[1],
											w_ran: w_ran+patch_size[2]]
					random_crop_data_t2 = np.asarray(random_crop_data_t2)

					random_crop_truth = img_label[0,0, d_ran : d_ran+patch_size[0], 
											h_ran : h_ran+patch_size[1],
											w_ran: w_ran+patch_size[2]]
					random_crop_truth = np.asarray(random_crop_truth)

					assert random_crop_data_t1.shape==(patch_size[0],patch_size[1],patch_size[2]), \
							'random_crop_data shape(%s) is not in (%s,%s,%s)'%(random_crop_data_t1.shape,patch_size[0],patch_size[1],patch_size[2])

					assert random_crop_truth.shape==(patch_size[0],patch_size[1],patch_size[2]), \
							'random_crop_label shape is not in (%s,%s,%s)'%(patch_size[0],patch_size[1],patch_size[2])

					
					x1_list.append(random_crop_data_t1)
					x2_list.append(random_crop_data_t2)
					y_list.append(random_crop_truth)
					# dm_list.append(random_crop_data_dm)

				yield convert_data_multimodality(x1_list, x2_list, y_list)
				

			file_handle.close()



def get_validation_split( hdf5_train_list_file, hdf5_validation_list_file,
							# data_split=0.8, 
							shuffle_list=True, 
							overwrite_split=True):
	'''
	split the whole dataset to training and testing part
	'''
	# if overwrite_split or not os.path.exists(FLAGS.training_file):
	# print("Creating validation split...")
	if not os.path.exists(hdf5_train_list_file) or not os.path.exists(hdf5_validation_list_file):
		print ("hdf5_list_file_path: %s does not exists..." % (hdf5_train_list_file))
		print ("hdf5_validation_list_file: %s does not exists..." % (hdf5_validation_list_file))
		sys.exit(0)

	with open(hdf5_train_list_file) as f:
		hdf5_files = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		hdf5_training_files_list = [x.strip() for x in hdf5_files] 

	with open(hdf5_validation_list_file) as f:
		hdf5_files = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		hdf5_validation_files_list = [x.strip() for x in hdf5_files] 



	if shuffle_list:
		shuffle(hdf5_training_files_list)
		shuffle(hdf5_validation_files_list)

	training_list = hdf5_training_files_list
	validation_list = hdf5_validation_files_list

	
	return training_list, validation_list


def normalize_data(data, mean, std):
	data2 = data - mean
	data2 = data2 / std
	return data2


def normalize_data_storage(data_storage):
    assert len(data_storage.shape)==4, 'normalize data, must be in 4-d..'
    
    ret_data = np.zeros(data_storage.shape)
    for index in range(data_storage.shape[0]):
    	_mean = data_storage[index].mean()
    	_std = data_storage[index].std()
    	_std = 1.0 if math.fabs(_std) < 1e-8 else _std
        ret_data[index] = normalize_data(data_storage[index], _mean, _std)
    return ret_data



def convert_test_input(test_input,normalise=True):
	test_input = np.asarray(test_input)
	
	if normalise:
		test_input = normalize_data_storage(test_input)
	
	test_input = test_input[..., np.newaxis]
	
	return test_input
	


def convert_data(train_input, train_label):
	train_input = np.asarray(train_input) 
	train_label = np.asarray(train_label) 

	train_input2 = normalize_data_storage(train_input)
	train_input2 = train_input2[..., np.newaxis]
	
	return train_input2, train_label


def convert_data_multimodality(x1_list, x2_list, y_list):
	x1_list = np.asarray(x1_list)
	x2_list = np.asarray(x2_list)
	y_list = np.asarray(y_list)

	t1_data = normalize_data_storage(x1_list)
	t1_data = t1_data[..., np.newaxis]

	t2_data = normalize_data_storage(x2_list)
	t2_data = t2_data[..., np.newaxis]

	return t1_data, t2_data, y_list


def main():
	training_generator, testing_generator = get_training_and_testing_generators()
	train_input1,  train_input2,  train_label = training_generator.next()
	print train_input1.shape





if __name__ == '__main__':
	main()
	

	