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
from one_epoch_functions import train_one_epoch
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
parser.add_argument('--log_dir', type=str, default='data/data_loss_nodes/logs', help='Log dir [default: logs]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 16384]')
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to run [default: 0]')
parser.add_argument('--max_epoch', type=int, default=2, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', type=str, default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=12500, help='Decay step for lr decay [default: 12500]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--restore_model', type=str, default='log/', help='Pretrained model')
parser.add_argument('--val_size', type=float, default=0.2, help='Ratio val/train [default:0.2]')
parser.add_argument('--k_fold', type=int, default=5, help='Numbers of folds for K fold validation [default: 5]')
parser.add_argument('--seed', type=int, default=10, help='Seed for pseudorandom processes [default: 10]')
parser.add_argument('--cube_size', type=float, default=10.0, help='Size of the cube of subclouds [default: 10.0]')
parser.add_argument('--num_dims', type=int, default=3, help='Number of dimensions of the point cloud [default: 3 (xyz)]')
parser.add_argument('--num_classes', type=int, default=5, help='Number os semantic classes [default: 4]')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode [default: True]')
parser.add_argument('--bandwidth', type=float, default=0.6, help='Bandwidth for meanshift clustering [default: 0.6]')
parser.add_argument('--early_stopping', type=int, default=10, help='Max number of epochs without progress [default: 10]')
parser.add_argument('--decimals', type=float, default=4, help='Number of decimals to save the metrics [default: 4]')
parser.add_argument('--models_dir', type=str, default='data/data_loss_nodes/trained_models', help='Log dir [default: data/trained_models]')
parser.add_argument('--noise_train_sigma', type=float, default=0.01, help='Standar deviation of the normal distribution used to augment train data [default: 0.01]')
parser.add_argument('--overlap', type=float, default=1.0, help='Overlap between cubes [default: 1.0]')
parser.add_argument('--idx_node', type=int, default=4, help='Semantic index of node points[default: 4]')
parser.add_argument('--rate_node_min', type=float, default=0.002, help='Loss parameter. Minimum and maximum rate of nodes/points: 0.002]')
parser.add_argument('--rate_node_max', type=float, default=0.07, help='Loss parameter. Maximum and maximum rate of nodes/points: 0.07]')
parser.add_argument('--dataset_path', type=str, default='data/data_loss_nodes/synthetic_point_clouds', help='Path of the dataset')
parser.add_argument('--path_test', type=str, default='data/data_loss_nodes/test', help='Folder to save test point clouds [default: test]')
parser.add_argument('--path_errors', type=str, default='data/data_loss_nodes/errors', help='Folder to save errors in tested point clouds [default: errors]')
parser.add_argument('--path_val', type=str, default=None, help='Folder to save validation point clouds [default: None]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
START_EPOCH = FLAGS.start_epoch
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
PRETRAINED_MODEL_PATH = FLAGS.restore_model
DATASET_PATH = FLAGS.dataset_path
VAL_SIZE = FLAGS.val_size
K_FOLD = FLAGS.k_fold
NUM_DIMS = FLAGS.num_dims
SEED = FLAGS.seed
NUM_CLASSES = FLAGS.num_classes
VERBOSE = FLAGS.verbose
PATH_VAL = FLAGS.path_val
PATH_TEST = FLAGS.path_test
PATH_ERRORS = FLAGS.path_errors
BANDWIDTH = FLAGS.bandwidth
EARLY_STOPPING = FLAGS.early_stopping
DECIMALS = FLAGS.decimals
LOG_DIR = FLAGS.log_dir
MODELS_DIR = FLAGS.models_dir
NOISE_TRAIN_SIGMA = FLAGS.noise_train_sigma
OVERLAP=FLAGS.overlap
IDX_NODE = FLAGS.idx_node
RATE_NODE_MIN = FLAGS.rate_node_min
RATE_NODE_MAX = FLAGS.rate_node_min
RATE_NODE = [RATE_NODE_MIN, RATE_NODE_MAX]

# CHECK PATHS
LOG_DIR =check_path(LOG_DIR)
MODELS_DIR = check_path(MODELS_DIR)
DATASET_PATH = check_path(DATASET_PATH)
PATH_VAL= check_path(PATH_VAL)
PATH_TEST = check_path(PATH_TEST)
PATH_ERRORS = check_path(PATH_ERRORS)

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
logger = get_logger(__file__, str(LOG_DIR), 'log_train.txt')
logger.info(str(FLAGS) + '\n')

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

    # Get model and loss
    pred_sem, pred_ins = model.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
    pred_sem_softmax = tf.nn.softmax(pred_sem)
    pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)
    pred_sem_softmax_nodes = pred_sem_softmax[...,IDX_NODE]

    # loss, sem_loss, disc_loss, l_var, l_dist = model.get_loss(pred_ins, labels_pl, pred_sem_label, pred_sem, sem_labels_pl)
    loss, sem_loss, disc_loss, box_loss, n_loss = model.get_loss_nodes(pred_ins, labels_pl, pred_sem_label, pred_sem, sem_labels_pl, pointclouds_pl, IDX_NODE, pred_sem_softmax_nodes, RATE_NODE)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('sem_loss', sem_loss)
    tf.summary.scalar('disc_loss', disc_loss)
    #tf.summary.scalar('l_var', l_var)
    #tf.summary.scalar('l_dist', l_dist)
    tf.summary.scalar('box_loss', box_loss)
    tf.summary.scalar('n_loss', n_loss)

    # Get training operator
    learning_rate = get_learning_rate(BASE_LEARNING_RATE, batch, DECAY_STEP, DECAY_RATE)
    tf.summary.scalar('learning_rate', learning_rate)
    if OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, var_list=tf.trainable_variables(), global_step=batch)

    # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(max_to_keep=None)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Add summary writers
    merged = tf.summary.merge_all()

    ops = {'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'sem_labels_pl': sem_labels_pl,
            'is_training_pl': is_training_pl,
            'loss': loss,
            'sem_loss': sem_loss,
            'disc_loss': disc_loss,
            #'l_var': l_var,
            #'l_dist': l_dist,
            'box_loss': box_loss,
            'n_loss': n_loss,
            'train_op': train_op,
            'merged': merged,
            'step': batch,
            'learning_rate': learning_rate,
            'pred_ins': pred_ins,
            'pred_sem_label': pred_sem_label,
            'pred_sem_softmax': pred_sem_softmax}

    # train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    #==================================================================================
    # K-fold cross validation

    # CSV
    header = ["loss", "sem_loss", "dist_loss", "box_loss", "n_nodes_loss", "oAcc", "mAcc", "mIoU", "mPrec", "mRec", "cov", "wCov"]
    for num in range(NUM_CLASSES):
        header.append('acc_'+str(num))
    header.append("k_fold")
    header.append("model_path")
    header.append(str(FLAGS))

    # CSV for saving val metrics
    val_metrics_path = LOG_DIR.joinpath('metrics_val.csv')
    with open(str(val_metrics_path), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        csvfile.close()

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

    k=0
    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
    for train, test in kfold.split(input_las):

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=1)

        #==============================================================================
        # TRAIN
        # Split the dataset in train and val
        train_dataset_paths, val_dataset_paths = train_test_split(input_las[train], test_size=VAL_SIZE, random_state=SEED)

        # Prepare the dataset to be load in each epoch
        train = GeneralDataset(files=train_dataset_paths, num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, split='train', sigma=NOISE_TRAIN_SIGMA, zyx_max_angle=[np.pi, 5*np.pi/180, 5*np.pi/180], seed=SEED)
        val = GeneralDataset(files=val_dataset_paths, num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, split='train', zyx_max_angle=[np.pi, 5*np.pi/180, 5*np.pi/180], seed=SEED)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
        sess.run(adam_initializers)

        # old_metrics_epoch = np.zeros(7)
        best_metric = 0
        no_progress = 0

        # Folder for saving point clouds of validation
        if PATH_VAL is not None:
            val_point_clouds_path = PATH_VAL.joinpath('k_' + str(k))
            val_point_clouds_path.mkdir(exist_ok=True)

        # Folder for saving the models
        models_k_dir = MODELS_DIR.joinpath('k_' + str(k))
        models_k_dir.mkdir(exist_ok=True)

        for epoch in range(START_EPOCH, MAX_EPOCH):
            
            #Train
            loss_train, _, _, _, _ = train_one_epoch(sess, ops, train, BATCH_SIZE)

            # Name for saving one cloud used in validation
            if PATH_VAL is not None:
                file_name = "val_" + str(epoch).zfill(len(str(MAX_EPOCH))) + ".las"
                file_name = val_point_clouds_path.joinpath(file_name)
            else:
                file_name = None

            # Validation
            loss_val, loss_sem, loss_dist, loss_box, loss_n, oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov, accs = val_one_epoch(
                sess, ops, val, BATCH_SIZE, NUM_CLASSES, file_name=file_name, bandwidth=BANDWIDTH)

            # Save in logger
            logger.info('K:' + str(k) + ' ' +
                'Epoch: ' + str(epoch) + ' ' + 
                'Mean loss train: ' + np.array2string(loss_train, precision=5, separator=',') + ' ' +
                'Mean loss val: ' + np.array2string(loss_val, precision=5, separator=',') + ' ' + 
                'Mean loss val sem: ' + np.array2string(loss_sem, precision=5, separator=',') + ' ' +
                'Mean loss val dist: ' + np.array2string(loss_dist, precision=5, separator=',') + ' ' +
                'Mean loss val box: ' + np.array2string(loss_box, precision=5, separator=',') + ' ' +
                'Mean loss val n nodes: ' + np.array2string(loss_n, precision=5, separator=',') + ' ' + 
                'oAcc: ' + np.array2string(oAcc, precision=5, separator=',') + ' ' +
                'mAcc: ' + np.array2string(mAcc, precision=5, separator=',') + ' ' +
                'mIoU: ' + np.array2string(mIoU, precision=5, separator=',') + ' ' +
                'mPrec: ' + np.array2string(mPrec, precision=5, separator=',') + ' ' +
                'mRec: ' + np.array2string(mRec, precision=5, separator=',') + ' ' +
                'cov: ' + np.array2string(cov, precision=5, separator=',') + ' ' +
                'wCov: ' + np.array2string(wCov, precision=5) + ' ' +
                'accs: ' + np.array2string(accs, precision=5) + '\n')
            
            metrics = np.array([loss_val, loss_sem, loss_dist, loss_box, loss_n, oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov])
            metrics = np.concatenate((metrics, accs))

            # Save in validation metrics
            with open(str(val_metrics_path), 'a') as csvfile:
                writer = csv.writer(csvfile)
                metrics_csv = ["{:0.{precision}f}".format(v, precision=DECIMALS) for v in metrics]
                metrics_csv.append(epoch)
                metrics_csv.append(k)
                writer.writerow(metrics_csv)
                csvfile.close()

            # Check if there is progress
            check_metric = accs[IDX_NODE]
            if not check_metric > best_metric:
                no_progress +=1

            # If there are progress
            else:
                # Update variables
                best_metric = check_metric
                no_progress=0
                # Save model
                best_model_path = models_k_dir.joinpath('epoch_' + str(epoch).zfill(len(str(MAX_EPOCH))) + '.ckpt')
                model_path = saver.save(sess, str(best_model_path))
                logger.info("Model saved in file: %s" % model_path + '\n')

            if no_progress>=EARLY_STOPPING: break
        
        #==============================================================================
        # TEST

        # CONFIGURE MODEL ARCHITECTURE
        # Create placeholders
        BATCH_SIZE = 1 # required for testing
        pointclouds_pl, labels_pl, sem_labels_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_DIMS)
        is_training_pl = False

        # Get model
        pred_sem, pred_ins = model.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
        pred_sem_softmax = tf.nn.softmax(pred_sem)
        pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)
        pred_sem_softmax_nodes = pred_sem_softmax[...,IDX_NODE]

        ops = {'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'sem_labels_pl': sem_labels_pl,
            'is_training_pl': is_training_pl,
            'pred_ins': pred_ins,
            'pred_sem_label': pred_sem_label,
            'pred_sem_softmax': pred_sem_softmax}
        
        # Load best model
        loader = tf.train.Saver()
        loader.restore(sess, str(best_model_path))

        # Dataset
        test = GeneralDataset(files=input_las[test], num_dims=NUM_DIMS, cube_size=CUBE_SIZE, npoints=NUM_POINT, overlap=OVERLAP, split='test', seed=SEED)

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
