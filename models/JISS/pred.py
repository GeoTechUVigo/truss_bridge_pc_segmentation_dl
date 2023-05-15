"""
Daniel Lamas Novoa.
Enxeñaría dos materiais, mecánica aplicada e construción.
Escola de enxeñería industrial.
Grupo de xeotecnoloxía aplicada.
Universidade de Vigo.
https://orcid.org/0000-0001-7275-183X
14/12/2022
"""
#==============================================================================

import argparse
import socket
import sys
import pathlib
import tensorflow as tf
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.resolve().parent.parent

sys.path.append(str(BASE_DIR))
sys.path.append(str(ROOT_DIR.joinpath('utils')))

import model
from one_epoch_functions import pred_on_dataset
from load_las_data import GeneralDataset
#==============================================================================

# Check GPU
if not tf.test.is_gpu_available():
     raise TypeError('Tensorflow-gpu: \
         Invalid device or cannot modify virtual devices once initialized.')
#==============================================================================

# FLAGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=16384, help='Point number [default: 16384]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--seed', type=int, default=10, help='Seed for pseudorandom processes [default: 10]')
parser.add_argument('--cube_size', type=float, default=10.0, help='Size of the cube of subclouds [default: 10.0]')
parser.add_argument('--num_dims', type=int, default=3, help='Number of dimensions of the point cloud [default: 3 (xyz)]')
parser.add_argument('--num_classes', type=int, default=5, help='Number os semantic classes [default: 4]')
parser.add_argument('--bandwidth', type=float, default=0.5, help='Bandwidth for meanshift clustering [default: 0.5]')
parser.add_argument('--dataset_path', type=str, default='data/data_loss_nodes', help='Path of the dataset')
parser.add_argument('--model_path', default='data/data_loss_nodes/trained_models/k_0/epoch_0307.ckpt', help='Log dir [default: ]')
parser.add_argument('--path_pred', type=str, default='data/data_loss_nodes/pred', help='Folder to save predicted point clouds [default: data/pred]')
parser.add_argument('--path_pred_cubes', type=str, default=None, help='Folder to save predicted point clouds cubes [default: data/pred]')

FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch_size
SEED = FLAGS.seed
CUBE_SIZE = FLAGS.cube_size
NUM_DIMS = FLAGS.num_dims
NUM_CLASSES = FLAGS.num_classes
BANDWIDTH = FLAGS.bandwidth
DATASET_PATH = FLAGS.dataset_path
MODEL_PATH = FLAGS.model_path
PATH_PRED = FLAGS.path_pred
PATH_PRED_CUBES = FLAGS.path_pred_cubes

# CHECK PATHS
DATASET_PATH = pathlib.Path(DATASET_PATH)
PATH_PRED = pathlib.Path(PATH_PRED)
MODEL_PATH=pathlib.Path(MODEL_PATH)

if not PATH_PRED_CUBES is None:
    PATH_PRED_CUBES=pathlib.Path(PATH_PRED_CUBES)
    if not PATH_PRED_CUBES.is_dir(): raise ValueError("PATH_PRED_CUBES {} does not exist.".format(str(PATH_PRED_CUBES)))

if not DATASET_PATH.is_dir(): raise ValueError("DATASET_PATH {} does not exist.".format(str(DATASET_PATH)))
if not PATH_PRED.is_dir(): raise ValueError("PATH_PRED {} does not exist.".format(str(PATH_VAL)))
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
    #==================================================================================

    # Restore variables from disk.
    loader.restore(sess, str(MODEL_PATH))

    # Dataset
    data = GeneralDataset(files=input_las, num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, overlap=1.0, split='pred', seed=SEED)

    # Prediction
    mean_num_pts_in_group = np.ones(NUM_CLASSES)
    pred_on_dataset(sess, ops, data, num_classes=NUM_CLASSES, mean_num_pts_in_group=mean_num_pts_in_group, save_folder=PATH_PRED, save_cubes=PATH_PRED_CUBES, bandwidth=BANDWIDTH)