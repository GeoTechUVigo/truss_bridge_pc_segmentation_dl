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
parser.add_argument('--log_dir', default='data/data_loss_nodes/test/logs', help='Log dir [default: logs]')
parser.add_argument('--models_dir', default='data/data_loss_nodes/trained_models', help='Log dir [default: trained_models]')
parser.add_argument('--num_point', type=int, default=16384, help='Point number [default: 16384]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 2]')
parser.add_argument('--dataset_path', type=str, default='data/data_loss_nodes/synthetic_point_clouds', help='Path of the dataset')
parser.add_argument('--seed', type=int, default=10, help='Seed for pseudorandom processes [default: 10]')
parser.add_argument('--cube_size', type=float, default=10.0, help='Size of the cube of subclouds [default: 10.0]')
parser.add_argument('--num_dims', type=int, default=3, help='Number of dimensions of the point cloud [default: 3 (xyz)]')
parser.add_argument('--num_classes', type=int, default=5, help='Number os semantic classes [default: 4]')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode [default: True]')
parser.add_argument('--path_test', type=str, default='data/data_loss_nodes/test', help='Folder to save test point clouds [default: data/test]')
parser.add_argument('--path_test_cubes', type=str, default=None, help='Folder to save test point clouds cubes [default: None]')
parser.add_argument('--path_errors', type=str, default='data/data_loss_nodes/errors', help='Folder to save the errors [default: data/errors]')
parser.add_argument('--bandwidth', type=float, default=0.5, help='Bandwidth for meanshift clustering [default: 0.5]')
parser.add_argument('--decimals', type=float, default=4, help='Number of decimals to save the metrics [default: 4]')
parser.add_argument('--train_test_idx', type=str, default='data/data_loss_nodes/kfold', help='Folder to load the indexes to split train and test for each k fold in .npy [default: None]')

FLAGS = parser.parse_args()

BATCH_SIZE = 1 # required for testing
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
DATASET_PATH = FLAGS.dataset_path
NUM_DIMS = FLAGS.num_dims
SEED = FLAGS.seed
NUM_CLASSES = FLAGS.num_classes
VERBOSE = FLAGS.verbose
PATH_TEST = FLAGS.path_test
PATH_TEST_CUBES=FLAGS.path_test_cubes
PATH_ERRORS=FLAGS.path_errors
BANDWIDTH = FLAGS.bandwidth
DECIMALS = FLAGS.decimals
LOG_DIR = FLAGS.log_dir
MODELS_DIR=FLAGS.models_dir
TRAIN_TEST_IDX = FLAGS.train_test_idx

# CHECK PATHS
LOG_DIR = check_path(LOG_DIR)
MODELS_DIR = check_path(MODELS_DIR)
DATASET_PATH = check_path(DATASET_PATH)
PATH_TEST = check_path(PATH_TEST)
PATH_TEST_CUBES = check_path(PATH_TEST_CUBES)
PATH_ERRORS=check_path(PATH_ERRORS)
TRAIN_TEST_IDX = check_path(TRAIN_TEST_IDX)
#==============================================================================

# Hiperparametres
CUBE_SIZE=FLAGS.cube_size

#==============================================================================
# CSV
header = ["loss", "sem_loss", "dist_loss", "box_loss", "n_nodes_loss", "oAcc", "mAcc", "mIoU", "mPrec", "mRec", "cov", "wCov"]
for num in range(NUM_CLASSES):
    header.append('acc_'+str(num))
header.append("k_fold")
header.append("model_path")
header.append(str(FLAGS))

# CSV for saving test metrics
header.remove("loss")
header.remove("sem_loss")
header.remove("dist_loss")
header.remove("box_loss")
header.remove("n_nodes_loss")
test_metrics_path = LOG_DIR.joinpath('metrics_test.csv')
with open(str(test_metrics_path), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    csvfile.close()

# Dataset files
input_las = sorted(DATASET_PATH.iterdir())
input_las = np.asarray([str(file) for file in input_las])

# Index of the files used in each kfold for testing
test_idx_data = [dir for dir in TRAIN_TEST_IDX.iterdir() if dir.suffix == '.npy']

# Models
k_folds = [dir for dir in MODELS_DIR.iterdir()]

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

    # train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    #==================================================================================
    # TEST
    k = 0
    for fold in k_folds:
        
        # Restore variables from disk.
        model = [x for x in fold.iterdir()]
        select = -1 if not 'checkpoint' in model[-1].stem else -2
        loader.restore(sess, str(fold.joinpath(model[select].stem)))

        # Dataset
        test = np.load(str(test_idx_data[k]))
        test = GeneralDataset(files=input_las[test], num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, overlap=1.0, split='test', seed=SEED)

        # Folder for saving the point clouds
        if PATH_TEST is not None:
                save_folder = pathlib.Path(PATH_TEST).joinpath('k_' + str(k))
                save_folder.mkdir(exist_ok=True)
        else:
                save_folder=None

        if PATH_ERRORS is not None:
                save_errors_folder = pathlib.Path(PATH_ERRORS).joinpath('k_' + str(k))
                save_errors_folder.mkdir(exist_ok=True)
        else:
                save_errors_folder=None

        # Test
        mean_num_pts_in_group = np.ones(NUM_CLASSES)
        oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov, accs = test_on_dataset(sess, ops, test, NUM_CLASSES, mean_num_pts_in_group, save_folder=save_folder, save_errors=save_errors_folder, bandwidth=BANDWIDTH)

        # Save in test metrics
        metrics = np.array([oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov])
        metrics = np.concatenate((metrics, accs))
        with open(str(test_metrics_path), 'a') as csvfile:
                writer = csv.writer(csvfile)
                metrics_csv = ["{:0.{precision}f}".format(v, precision=DECIMALS) for v in metrics]
                metrics_csv.append(k)
                metrics_csv.append(best_model_path)
                writer.writerow(metrics_csv)
                csvfile.close()

        #Udate k of K-fold
        k+=1