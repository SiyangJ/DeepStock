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

tf.app.flags.DEFINE_integer('cls_out', ARGS.getint('cls_out'), 
                            "classfy how many categories")
tf.app.flags.DEFINE_integer('batch_size', ARGS.getint('batch_size'), 
                            "Number of samples per batch.")
tf.app.flags.DEFINE_string('patch_size_str', ARGS['patch_size_str'], 
                           "patch size that we will extract from 3D image")
tf.app.flags.DEFINE_integer('batches_one_image', ARGS.getint('batches_one_image'), 
                            "how many batches extraction from a 3D image for training")
tf.app.flags.DEFINE_integer('overlap_add_num', ARGS.getint('overlap_add_num'), 
                            "patch_num = Len/patch_size + overlap_add_num when extracting patches for test")
tf.app.flags.DEFINE_integer('prepost_pad', ARGS.getint('prepost_pad'), 
                            "padding when remove zero backgrounds in preprocess")
tf.app.flags.DEFINE_integer('training_crop_pad', ARGS.getint('training_crop_pad'), 
                            "padding when remove zero backgrounds in preprocess")

############### Training and Learning rate decay ##################################
tf.app.flags.DEFINE_float('momentum', ARGS.getfloat('momentum'), 
                          "momentum for accelearating training")
tf.app.flags.DEFINE_float('learning_rate_start', ARGS.getfloat('learning_rate_start'), 
                          "start learning rate ")
tf.app.flags.DEFINE_integer('learning_rate_reduce_life', ARGS.getint('learning_rate_reduce_life'), 
                            "Number of batches until learning rate is reduced. lr *= 0.1")
tf.app.flags.DEFINE_float('learning_rate_percentage', ARGS.getfloat('learning_rate_percentage'), 
                          "Number of batches until learning rate is reduced. lr *= 0.1")
tf.app.flags.DEFINE_integer('max_batch', ARGS.getint('max_batch'),
                            "max batch number")
tf.app.flags.DEFINE_integer('checkpoint_period', ARGS.getint('checkpoint_period'), 
                            "Number of batches in between checkpoints")
tf.app.flags.DEFINE_string('checkpoint_dir', ARGS['checkpoint_dir'], 
                           "Output folder where training logs and models are dumped.")
#tf.app.flags.DEFINE_string('last_trained_checkpoint', './checkpoint_t1_t2_9case_10000', "The model used for testing..")
tf.app.flags.DEFINE_string('last_trained_checkpoint', ARGS['last_trained_checkpoint'], 
                           "The model used for testing")
tf.app.flags.DEFINE_bool('restore_from_last', ARGS.getboolean('restore_from_last'), 
                         "whether start training from last trained checkpoint")

############### Deep supervision######################
tf.app.flags.DEFINE_float('aux1_weight', ARGS.getfloat('aux1_weight'), 
                          "loss weight of aux1 classifier")
tf.app.flags.DEFINE_float('aux2_weight', ARGS.getfloat('aux2_weight'), 
                          "loss weight of aux2 classifier")
tf.app.flags.DEFINE_float('main_weight', ARGS.getfloat('main_weight'), 
                          "loss weight of main classifier")
tf.app.flags.DEFINE_float('L2_loss_weight', ARGS.getfloat('L2_loss_weight'), 
                          "loss weight of main classifier")
# tf.app.flags.DEFINE_float('reject_T', 0.05, "remove isolated regions, when the area is less then reject_T")



################### Train Data################
# /proj/NIRAL/users/siyangj/myData/BernNet
tf.app.flags.DEFINE_string('train_data_dir', ARGS['train_data_dir'],
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_list_path', ARGS['hdf5_list_path'],
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_train_list_path', ARGS['hdf5_train_list_path'],
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_validation_list_path', ARGS['hdf5_validation_list_path'], 
                           "Store the training hdf5 file list.")
tf.app.flags.DEFINE_string('hdf5_dir', ARGS['hdf5_dir'],
                           "Store the path which contains hdf5 files.")


################# Pretrain Model: Partial Transfer Learning  ########################################################
tf.app.flags.DEFINE_bool('from_pretrain', ARGS.getboolean('from_pretrain'), 
                         "when init value from pretrain-ed model")
tf.app.flags.DEFINE_string('hdf5_hip_transfer_model', ARGS['hdf5_hip_transfer_model'],
                           "where is the pre-trained model")
tf.app.flags.DEFINE_string('hdf5_sports_3d_model', ARGS['hdf5_sports_3d_model'],
                           "where is the pre-trained model")
tf.app.flags.DEFINE_string('model_saved_hdf5', ARGS['model_saved_hdf5'],
                           "where is the pre-trained model")
tf.app.flags.DEFINE_bool('xavier_init', ARGS.getboolean('xavier_init'),
                         "whether multi-modality is used")


tf.app.flags.DEFINE_bool('log_device_placement', ARGS.getboolean('log_device_placement'), 
                         "Log the device where variables are placed.")
tf.app.flags.DEFINE_integer('random_seed', ARGS.getint('random_seed'), 
                            "Seed used to initialize rng.")
tf.app.flags.DEFINE_float('epsilon', ARGS.getfloat('epsilon'), 
                          "Fuzz term to avoid numerical instability")


################ Test Data ###############################
#tf.app.flags.DEFINE_string('test_dir','/proj/NIRAL/users/jphong/6moSegData/IBIS/Test',"the directory which contains nifti images to be segmented.")
tf.app.flags.DEFINE_string('test_dir', ARGS['test_dir'],
                           "the directory which contains nifti images to be segmented.")

def main():
  print FLAGS.testing_file

if __name__ == '__main__':
  main()
