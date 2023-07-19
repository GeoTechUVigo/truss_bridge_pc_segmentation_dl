"""
Daniel Lamas Novoa.
Enxeñaría dos materiais, mecánica aplicada e construción.
Escola de enxeñería industrial.
Grupo de xeotecnoloxía aplicada.
Universidade de Vigo.
https://orcid.org/0000-0001-7275-183X
25/05/2022
"""
#==============================================================================

import argparse
import socket
import sys
import pathlib
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.resolve().parent.parent

sys.path.append(str(BASE_DIR))
sys.path.append(str(ROOT_DIR.joinpath('utils')))

import model
from update_train_parametres import get_learning_rate
from update_train_parametres import get_bn_decay
from one_epoch_functions import test_on_dataset, train_one_epoch
from one_epoch_functions import val_one_epoch
from one_epoch_functions import test_on_dataset
from log_util import get_logger
from load_las_data import GeneralDataset
from utils.check_path import check_path
#==============================================================================

# Check GPU
if not tf.test.is_gpu_available():
     raise TypeError('Tensorflow-gpu: \
         Invalid device or cannot modify virtual devices once initialized.')
#==============================================================================

# FLAGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 16384]')
parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size [default: 0.05]')
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to run [default: 0]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', type=str, default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=12500, help='Decay step for lr decay [default: 12500]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--restore_model', type=str, default='log/', help='Pretrained model')
parser.add_argument('--val_size', type=float, default=0.2, help='Ratio val/train [default:0.2]')
parser.add_argument('--n_train_fold', type=int, default=1, help='Numbers of folds for K fold validation [default: 5]')
parser.add_argument('--seed', type=int, default=10, help='Seed for pseudorandom processes [default: 10]')
parser.add_argument('--cube_size', type=float, default=10.0, help='Size of the cube of subclouds [default: 10.0]')
parser.add_argument('--num_dims', type=int, default=3, help='Number of dimensions of the point cloud [default: 3 (xyz)]')
parser.add_argument('--num_classes', type=int, default=4, help='Number os semantic classes [default: 4]')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode [default: True]')
parser.add_argument('--bandwidth', type=float, default=0.6, help='Bandwidth for meanshift clustering [default: 0.6]')
parser.add_argument('--early_stopping', type=int, default=10, help='Max number of epochs without progress [default: 10]')
parser.add_argument('--decimals', type=float, default=4, help='Number of decimals to save the metrics [default: 4]')
parser.add_argument('--path_results', type=str, default='data/results/real', help='Log dir [default: data/trained_models]')
parser.add_argument('--noise_train_sigma', type=float, default=0.01, help='Standar deviation of the normal distribution used to augment train data [default: 0.01]')
parser.add_argument('--overlap', type=float, default=1.0, help='Overlap between cubes [default: 1.0]')
parser.add_argument('--dataset_path', type=str, default='data/data/occlusions/k_0', help='Path of the dataset')
parser.add_argument('--dataset_specific_test', type=str, default='data/data/uniform', help='Path of the dataset')
parser.add_argument('--dataset_real', type=str, default='data/data/test_real', help='Path of the dataset')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT=FLAGS.num_point
VOXEL_SIZE=FLAGS.voxel_size
START_EPOCH = FLAGS.start_epoch
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
PRETRAINED_MODEL_PATH = FLAGS.restore_model
DATASET_PATH = FLAGS.dataset_path
DATASET_SPECIFIC_TEST = FLAGS.dataset_specific_test
DATASET_REAL = FLAGS.dataset_real
VAL_SIZE = FLAGS.val_size
N_TRAIN_FOLD = FLAGS.n_train_fold
NUM_DIMS = FLAGS.num_dims
SEED = FLAGS.seed
NUM_CLASSES = FLAGS.num_classes
VERBOSE = FLAGS.verbose
BANDWIDTH = FLAGS.bandwidth
EARLY_STOPPING = FLAGS.early_stopping
DECIMALS = FLAGS.decimals
PATH_RESULTS = FLAGS.path_results
NOISE_TRAIN_SIGMA = FLAGS.noise_train_sigma
OVERLAP=FLAGS.overlap

# CHECK PATHS
DATASET_PATH = check_path(DATASET_PATH)
DATASET_REAL = check_path(DATASET_REAL)
DATASET_SPECIFIC_TEST=check_path(DATASET_SPECIFIC_TEST)
PATH_RESULTS = check_path(PATH_RESULTS)
#==============================================================================

# Hiperparametres
CUBE_SIZE=FLAGS.cube_size

#==============================================================================
# Dataset files
input_las = sorted(DATASET_PATH.iterdir())
input_las = np.asarray([str(file) for file in input_las])


# Model
BATCH_SIZE=1
MODEL_PATH = '/workspaces/truss_bridge_pc_segmentation_dl/data/trained_model/epoch_0636.ckpt'
path_k = PATH_RESULTS

#==============================================================================
# Build network and create session
with tf.Graph().as_default(), tf.device('/gpu:'+str(GPU_INDEX)):

        # CONFIGURE MODEL ARCHITECTURE
        # Create placeholders
        pointclouds_pl, labels_pl, sem_labels_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_DIMS)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get model
        pred_sem, pred_ins = model.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
        pred_sem_softmax = tf.nn.softmax(pred_sem)
        pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)

        loader = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        is_training = False

        # Add summary writers
        merged = tf.summary.merge_all()

        ops = {'pointclouds_pl': pointclouds_pl,
                'labels_pl': labels_pl,
                'sem_labels_pl': sem_labels_pl,
                'is_training_pl': is_training_pl,
                'pred_ins': pred_ins,
                'pred_sem_label': pred_sem_label,
                'pred_sem_softmax': pred_sem_softmax}

        # TEST

        # Dataset
        test = GeneralDataset(files=input_las, num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, voxel_size=VOXEL_SIZE, overlap=1.0, split='test', seed=SEED)

        # Folder for saving the point clouds
        save_folder = path_k.joinpath('test_'+str(DATASET_PATH.name))
        save_folder.mkdir(exist_ok=True)

        # Restore variables from disk.
        loader.restore(sess, str(MODEL_PATH))

        # Test
        mean_num_pts_in_group = np.ones(NUM_CLASSES)
        oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov, accs = test_on_dataset(sess, ops, test, BATCH_SIZE, NUM_CLASSES, mean_num_pts_in_group, save_folder=save_folder, bandwidth=BANDWIDTH)
