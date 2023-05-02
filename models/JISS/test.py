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
parser.add_argument('--log_dir', default='data/test/logs', help='Log dir [default: logs]')
parser.add_argument('--models_dir', default='trained_models', help='Log dir [default: trained_models]')
parser.add_argument('--num_point', type=int, default=16384, help='Point number [default: 16384]')
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to run [default: 0]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=12500, help='Decay step for lr decay [default: 12500]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--dataset_path', type=str, default='data/data_test/k_4', help='Path of the dataset')
parser.add_argument('--val_size', type=float, default=0.2, help='Ratio val/train [default:0.2]')
parser.add_argument('--k_fold', type=int, default=5, help='Numbers of folds for K fold validation [default: 5]')
parser.add_argument('--seed', type=int, default=10, help='Seed for pseudorandom processes [default: 10]')
parser.add_argument('--cube_size', type=float, default=10.0, help='Size of the cube of subclouds [default: 10.0]')
parser.add_argument('--num_dims', type=int, default=3, help='Number of dimensions of the point cloud [default: 3 (xyz)]')
parser.add_argument('--num_classes', type=int, default=4, help='Number os semantic classes [default: 4]')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode [default: True]')
parser.add_argument('--path_val', type=str, default='data/validation', help='Folder to save validation point clouds [default: validation]')
parser.add_argument('--path_test', type=str, default='data/test/k_4', help='Folder to save test point clouds [default: data/test]')
parser.add_argument('--path_test_cubes', type=str, default='data/test/k_4/cubes', help='Folder to save test point clouds cubes [default: data/test_cubes]')
parser.add_argument('--bandwidth', type=float, default=0.5, help='Bandwidth for meanshift clustering [default: 0.5]')
parser.add_argument('--epochs_without_progress', type=float, default=10, help='Max number of epochs without progress [default: 10]')
parser.add_argument('--decimals', type=float, default=4, help='Number of decimals to save the metrics [default: 4]')
parser.add_argument('--model_path', default='data/trained_models/k_4/epoch_1100.ckpt', help='Log dir [default: ]')
k=4
FLAGS = parser.parse_args()

BATCH_SIZE = 1 # required for testing
NUM_POINT = FLAGS.num_point
START_EPOCH = FLAGS.start_epoch
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DATASET_PATH = FLAGS.dataset_path
VAL_SIZE = FLAGS.val_size
K_FOLD = FLAGS.k_fold
NUM_DIMS = FLAGS.num_dims
SEED = FLAGS.seed
NUM_CLASSES = FLAGS.num_classes
VERBOSE = FLAGS.verbose
PATH_VAL = FLAGS.path_val
PATH_TEST = FLAGS.path_test
PATH_TEST_CUBES = FLAGS.path_test_cubes
BANDWIDTH = FLAGS.bandwidth
EPOCHS_WITHOUT_PROGRESS = FLAGS.epochs_without_progress
DECIMALS = FLAGS.decimals
LOG_DIR = FLAGS.log_dir
MODELS_DIR = FLAGS.models_dir
MODEL_PATH = FLAGS.model_path

# CHECK PATHS
LOG_DIR = check_path(LOG_DIR)
MODELS_DIR = check_path(MODELS_DIR)
DATASET_PATH = check_path(DATASET_PATH)
PATH_VAL= check_path(PATH_VAL)
PATH_TEST = check_path(PATH_TEST)
PATH_TEST_CUBES = check_path(PATH_TEST_CUBES)
MODEL_PATH = check_path(MODEL_PATH)
PATH_ERRORS = 'data/test/k_4/errors'
#==============================================================================

# Hiperparametres
CUBE_SIZE = FLAGS.cube_size
DECAY_STEP = FLAGS.decay_step
DECAY_STEP = int(DECAY_STEP / (BATCH_SIZE / 24))
DECAY_RATE = FLAGS.decay_rate
BASE_LEARNING_RATE = FLAGS.learning_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# backup model
# os.system('cp model.py {}'.format(LOG_DIR))
# os.system('cp train.py {}'.format(LOG_DIR))

# logger
#logger = get_logger(__file__, str(LOG_DIR), 'log_train.txt')
#logger.info(str(FLAGS) + '\n')

#==============================================================================

# Dataset files
input_las = sorted(DATASET_PATH.iterdir())
input_las = np.asarray([str(file) for file in input_las])

#==============================================================================
# Build network and create session
with tf.Graph().as_default(), tf.device('/gpu:'+str(GPU_INDEX)):

    # CONFIGURE MODEL ARCHITECTURE
    # Create placeholders
    pointclouds_pl, labels_pl, sem_labels_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_DIMS)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    # Note the global_step=batch parameter to minimize.
    # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
    batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
    bn_decay = get_bn_decay(BN_INIT_DECAY, batch, BN_DECAY_DECAY_STEP, BN_DECAY_DECAY_RATE, BN_DECAY_CLIP)
    tf.summary.scalar('bn_decay', bn_decay)

    # Get model
    pred_sem, pred_ins = model.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
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

    # train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    #==================================================================================

    # Restore variables from disk.
    loader.restore(sess, str(MODEL_PATH))
    
    #==============================================================================
    # TEST

    # Dataset
    test = GeneralDataset(files=input_las, num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, overlap=1.0, split='test', seed=SEED)

    # Test
    mean_num_pts_in_group = np.ones(NUM_CLASSES)
    oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov, accs = test_on_dataset(sess, ops, test, NUM_CLASSES, mean_num_pts_in_group, save_folder=PATH_TEST, save_errors=PATH_ERRORS, save_cubes=PATH_TEST_CUBES, bandwidth=BANDWIDTH)

    # Save in test metrics
    metrics = np.array([oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov])
    metrics = np.concatenate((metrics, accs))
    best_model_path=str(MODEL_PATH)
    test_metrics_path = LOG_DIR.joinpath('metrics_test.csv')
    with open(str(test_metrics_path), 'a') as csvfile:
            writer = csv.writer(csvfile)
            metrics_csv = ["{:0.{precision}f}".format(v, precision=DECIMALS) for v in metrics]
            metrics_csv.append(k)
            metrics_csv.append(best_model_path)
            writer.writerow(metrics_csv)
            csvfile.close()
