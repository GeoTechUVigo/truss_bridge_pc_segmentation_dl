"""
Daniel Lamas Novoa.
Enxeñaría dos materiais, mecánica aplicada e construción.
Escola de enxeñería industrial.
Grupo de xeotecnoloxía aplicada.
Universidade de Vigo.
https://orcid.org/0000-0001-7275-183X
13/06/2022
"""

import pathlib
import numpy as np
from ismember import ismember
from clustering import cluster
import metrics
from save_las import save_las
from scipy import stats
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


from test_utils import *


def train_one_epoch(sess, ops, dataset, batch_size):
    """
    Function for artificial neural network training in the tensorflow session.
    It uses the data from dataset in order by taking batches of the indicated size.

    :param sess: tensorflow session.
    :param ops: dict mapping from string to tf ops.
    :param train_writer: tf.summary.FileWriter.
    :param dataset: GeneralDataset obeject with the dataset used for training.
    :param batch_size: size of the batch.
    :return loss_train: loss.
    """
    
    is_training = True
    file_size = len(dataset)
    num_batches = file_size // batch_size

    loss_sum = 0
    loss_sem_sum = 0
    loss_dist_sum = 0
    loss_box_sum = 0
    loss_n_sum = 0


    for batch_idx in range(num_batches):
        # Load data
        _, current_data, current_sem, current_label = dataset.get_batch(batch_idx * batch_size, (batch_idx+1) * batch_size)
        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['sem_labels_pl']: current_sem,
                     ops['is_training_pl']: is_training}

        #summary, step, lr_rate, _, loss_val, sem_loss_val, disc_loss_val, l_var_val, l_dist_val = sess.run(
        #    [ops['merged'], ops['step'], ops['learning_rate'], ops['train_op'], ops['loss'], ops['sem_loss'],
        #     ops['disc_loss'], ops['l_var'], ops['l_dist']], feed_dict=feed_dict)

        summary, step, lr_rate, _, loss_val, sem_loss_val, disc_loss_val, box_los_val, n_loss_val = sess.run(
            [ops['merged'], ops['step'], ops['learning_rate'], ops['train_op'], ops['loss'], ops['sem_loss'],
             ops['disc_loss'], ops['box_loss'], ops['n_loss']], feed_dict=feed_dict)

        # train_writer.add_summary(summary, step)
        loss_sum += loss_val
        loss_sem_sum += sem_loss_val
        loss_dist_sum += disc_loss_val
        loss_box_sum += box_los_val
        loss_n_sum += n_loss_val

    loss_train = loss_sum / num_batches
    loss_sem_sum = loss_sem_sum / num_batches
    loss_dist_sum = loss_dist_sum / num_batches
    loss_box_sum = loss_box_sum / num_batches
    loss_n_sum = loss_n_sum / num_batches

    return loss_train, loss_sem_sum, loss_dist_sum, loss_box_sum, loss_n_sum


def val_one_epoch(sess, ops, dataset, batch_size, num_classes, file_name: str=None, bandwidth=1):
    """
    Function for artificial neural network validation in the tensorflow sessions.
    It uses the data from dataset in order by taking batches of the indicated size.
    If file_name parametre is specified, it saves the first segmented point cloud used for validating.
    Besides, it computes and return the following metrics:
    Segmentation metrics: overall accuracy, mean accuracy and mean intersection over union.
    Instance metrics: mean precision (IoU>0.5), mean recall (IoU>0.5), coverage and weighted coverage.
    Mean-shift clustering is used with the specified bandwith to generate instance objects and merge instances
    of different blocksby using BlockMerging algorithm (DOI 10.1109/CVPR.2018.00272).

    :param sess: tensorflow session.
    :param ops: dict mapping from string to tf ops.
    :param train_writer: tf.summary.FileWriter.
    :param dataset: GeneralDataset obeject with the dataset used for training.
    :param batch_size: size of the batch.
    :param num_classes: snumber of semantic classes.
    :param file_name: name for saving the first point cloud segmented [default: None].
    :param bandwidth: parametre used by the mean-shift clustering algorithm for the grouping of instances [default: 1].
    :return:
        loss_val: loss.
        oAcc: overall accuracy (semantic segmentation).
        mAcc: mean accuracy (semantic segmentation).
        mIoU: mean intersection over union (semantic segmentation).
        mPrec: mean precision (semantic segmentation).
        mRec: mean recall (instance segmentation).
        cov: coverage (instance segmentation).
        wCov: weighted coverage (instance segmentation).
    """
    
    is_training = False
    file_size = len(dataset)
    num_batches = file_size // batch_size

    save = True

    # Variables to calculate the metrics. One for each class in segmentation.
    loss_sum = 0
    loss_sem_sum = 0
    loss_dist_sum = 0
    loss_box_sum = 0
    loss_n_sum = 0

    # Semantic: oAcc, mAcc and mIoU
    sem_metrics_sum = np.zeros(3)
    # accuracy of each class
    sem_metrics_classbyclass = np.zeros(num_classes)
    # Instance: mPrec(IoU>0.5), mREc(IoU>0.5), Cov and WCov
    ins_metrics_sum = np.zeros(4)

    for batch_idx in range(num_batches):

        # Load data
        raw_data, current_data, current_sem, current_label = dataset.get_batch(batch_idx * batch_size, (batch_idx+1) * batch_size)

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['sem_labels_pl']: current_sem,
                     ops['is_training_pl']: is_training}

        # Prediction
        pred_ins_val, pred_sem_label_val, loss_val, sem_loss_val, disc_loss_val, box_los_val, n_loss_val = sess.run(
            [ops['pred_ins'], ops['pred_sem_label'], ops['loss'], ops['sem_loss'], ops['disc_loss'], ops['box_loss'], 
             ops['n_loss']], feed_dict=feed_dict)

        # Loss
        loss_sum += loss_val
        loss_sem_sum += sem_loss_val
        loss_dist_sum += disc_loss_val
        loss_box_sum += box_los_val
        loss_n_sum += n_loss_val

        # Semantic metrics
        oAcc, mAcc, mIoU, accs, _ = metrics.semantic_metrics(current_sem.reshape(-1), pred_sem_label_val.reshape(-1), num_classes)
        sem_metrics_sum += oAcc, mAcc, mIoU
        sem_metrics_classbyclass += accs

        # Analyse each point cloud 
        for i in range(batch_size):
            
            # Ids of each instance
            _, groupids, _ = cluster(pred_ins_val[i], bandwidth)
            # Instance metrics
            mPrec, mRecall, cov, wCov, _ = metrics.instance_metrics(current_label[i], groupids)
            ins_metrics_sum += mPrec, mRecall, cov, wCov

            # Save a segmented point cloud
            if not file_name == None and save:
                
                save_las(file_name, raw_data[i], pred_sem_label_val[i], groupids)

                save = False

    loss_val = loss_sum / num_batches
    loss_sem = loss_sem_sum / num_batches
    loss_dist = loss_dist_sum / num_batches
    loss_box = loss_box_sum / num_batches
    loss_n = loss_n_sum / num_batches

    sem_metrics = sem_metrics_sum / num_batches
    sem_metrics_classbyclass = sem_metrics_classbyclass / len(dataset)
    ins_metrics = ins_metrics_sum / (num_batches*batch_size)
    
    oAcc = sem_metrics[0]
    mAcc = sem_metrics[1]
    mIoU = sem_metrics[2]
    mPrec = ins_metrics[0]
    mRec = ins_metrics[1]
    cov = ins_metrics[2]
    wCov = ins_metrics[3]

    return loss_val, loss_sem, loss_dist, loss_box, loss_n, oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov, sem_metrics_classbyclass


def test_on_dataset(sess, ops, dataset, num_classes, mean_num_pts_in_group, save_folder=None, save_cubes=None, save_errors=None, bandwidth=1):
    """
    Function for artificial neural network test in the tensorflow sessions.
    It uses the data from dataset in order by taking batches of the indicated size.
    If save_folder parametre is specified, the segmented point clouds are saved there.
    Besides, it computes and return the following metrics:
    Segmentation metrics: overall accuracy, mean accuracy and mean intersection over union.
    Instance metrics: mean precision (IoU>0.5), mean recall (IoU>0.5), coverage and weighted coverage.
    Mean-shift clustering is used with the specified bandwith to generate instance objects and merge instances
    of different blocksby using BlockMerging algorithm (DOI 10.1109/CVPR.2018.00272).

    :param sess: tensorflow session.
    :param ops: dict mapping from string to tf ops.
    :param train_writer: tf.summary.FileWriter.
    :param dataset: GeneralDataset obeject with the dataset used for test.
    :param save_folder: folder for saving the point clouds segmented [default: None].
    :param save_cubes: folder for saving the cubes of each point cloud [default: None].
    :param errors: folder for saving the point cloud with its errors (semantic: (TP->0, otherwise-> 1), instance: (TP->0, otherwise-> 1))[default: None].
    :param bandwidth: parametre used by the mean-shift clustering algorithm for the grouping of instances [default: 1].
    :returns:
        - oAcc: overall accuracy (semantic segmentation).
        - mAcc: mean accuracy (semantic segmentation).
        - mIoU: mean intersection over union (semantic segmentation).
        - mPrec: mean precision (semantic segmentation).
        - mRec: mean recall (instance segmentation).
        - cov: coverage (instance segmentation).
        - wCov: weighted coverage (instance segmentation).
    """
    
    is_training = False

    # Variables to calculate the metrics. One for each class in segmentation.
    # Semantic: oAcc, mAcc and mIoU
    sem_metrics_sum = np.zeros(3)
    # accuracy of each class
    sem_metrics_classbyclass = np.zeros(num_classes)
    # Instance: mPrec(IoU>0.5), mREc(IoU>0.5), Cov and WCov
    ins_metrics_sum = np.zeros(4)

    for i in range(len(dataset)):

        # Load data
        raw_data, cur_sem, cur_group = dataset[i]

        # Arrays to save the prediction of each cube
        sem_pred_cubes = np.zeros((raw_data[:,:,0].shape))
        inst_pred_cubes = np.zeros((raw_data[:,:,0].shape))

        cur_pred_sem = np.zeros_like(cur_sem)
        cur_pred_sem_softmax = np.zeros([cur_sem.shape[0], cur_sem.shape[1], num_classes])
        group_output = np.zeros_like(cur_group)


        gap = 0.1
        cur_data = raw_data - raw_data.reshape(-1,3).min(axis=0) # move min to 0,0,0
        range_xyz = cur_data.max()
        volume_num = int(range_xyz / gap) + 1
        volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        volume_seg = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)

        # Split in batches
        for j in range(0, len(raw_data)):

            # Seletc batch
            coordinates_norm = raw_data[j]
            pts = cur_data[j]
            group = cur_group[j]
            sem = cur_sem[j]

            # Move points to the origin and resize to values [0,1]
            coordinates_norm = (coordinates_norm - coordinates_norm.min(axis=0))/dataset.cube_size

            feed_dict = {ops['pointclouds_pl']: coordinates_norm.reshape(1,coordinates_norm.shape[0], coordinates_norm.shape[1]),
                            ops['is_training_pl']: is_training}
        
            # Prediction
            pred_ins_val, pred_sem_label_val, pred_sem_softmax_val = sess.run(
                [ops['pred_ins'], ops['pred_sem_label'], ops['pred_sem_softmax']], feed_dict=feed_dict)

            pred_val = np.squeeze(pred_ins_val, axis=0)
            pred_sem = np.squeeze(pred_sem_label_val, axis=0)
            pred_sem_softmax = np.squeeze(pred_sem_softmax_val, axis=0)
            cur_pred_sem[j, :] = pred_sem
            cur_pred_sem_softmax[j, ...] = pred_sem_softmax

            # cluster
            group_seg = {}
            num_clusters, labels, cluster_centers = cluster(pred_val, bandwidth)
            for idx_cluster in range(num_clusters):
                tmp = (labels == idx_cluster)
                estimated_seg = int(stats.mode(pred_sem[tmp])[0])
                group_seg[idx_cluster] = estimated_seg

            groupids_block = labels

            groupids = BlockMerging(volume, volume_seg, pts,
                                    groupids_block.astype(np.int32), group_seg, gap)

            group_output[j, :] = groupids

            # Save each cube as a point cloud
            if save_cubes != None:
                file_name = pathlib.PurePath(save_cubes).joinpath(pathlib.PurePath(dataset.files[i]).stem + '_' + str(j) + pathlib.PurePath(dataset.files[i]).suffix)
                save_las(str(file_name), raw_data[j], cur_pred_sem[j], group_output[j])


        group_pred = group_output.reshape(-1)
        seg_pred = cur_pred_sem.reshape(-1)
        seg_pred_softmax = cur_pred_sem_softmax.reshape([-1, num_classes])
        pts = cur_data.reshape([-1, 3])

        # filtering
        x = (pts[:, 0] / gap).astype(np.int32)
        y = (pts[:, 1] / gap).astype(np.int32)
        z = (pts[:, 2] / gap).astype(np.int32)
        for j in range(group_pred.shape[0]):
            if volume[x[j], y[j], z[j]] != -1:
                group_pred[j] = volume[x[j], y[j], z[j]]

        seg_gt = cur_sem.reshape(-1)
        un = np.unique(group_pred)
        pts_in_pred = [[] for itmp in range(num_classes)]
        group_pred_final = -1 * np.ones_like(group_pred)
        grouppred_cnt = 0
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (group_pred == g)
            sem_seg_g = int(stats.mode(seg_pred[tmp])[0])
            # if np.sum(tmp) > 500:
            if np.sum(tmp) > 0.25 * mean_num_pts_in_group[sem_seg_g]:
                group_pred_final[tmp] = grouppred_cnt
                pts_in_pred[sem_seg_g] += [tmp]
                grouppred_cnt += 1

        ins = group_pred_final.astype(np.int32)
        sem = seg_pred.astype(np.int32)
        sem_softmax = seg_pred_softmax
        sem_gt = seg_gt
        ins_gt = cur_group.reshape(-1)

        # Adapt variables to the rest of the code
        raw_data = raw_data.reshape(-1,3)
        sem_pred = sem
        inst_pred = ins
        label = cur_group.reshape(inst_pred.shape)
        sem = cur_sem.reshape(sem_pred.shape)

        # Instance metrics
        mPrec, mRecall, cov, wCov, errors_ins = metrics.instance_metrics(label, inst_pred)
        ins_metrics_sum += mPrec, mRecall, cov, wCov
        # Semantic metrics
        oAcc, mAcc, mIoU, accs, errors_sem = metrics.semantic_metrics(sem, sem_pred, num_classes)
        sem_metrics_sum += oAcc, mAcc, mIoU
        sem_metrics_classbyclass += accs

        # save point cloud
        if save_folder != None:
            file_name = pathlib.PurePath(save_folder).joinpath(pathlib.PurePath(dataset.files[i]).name)
            save_las(str(file_name), raw_data, sem_pred, inst_pred)

        if save_errors!= None:
            file_name = pathlib.PurePath(save_errors).joinpath(pathlib.PurePath(dataset.files[i]).name)
            save_las(str(file_name), raw_data, errors_sem, errors_ins)

    sem_metrics = sem_metrics_sum / len(dataset)
    sem_metrics_classbyclass = sem_metrics_classbyclass / len(dataset)
    ins_metrics = ins_metrics_sum / len(dataset)
    
    oAcc = sem_metrics[0]
    mAcc = sem_metrics[1]
    mIoU = sem_metrics[2]
    mPrec = ins_metrics[0]
    mRec = ins_metrics[1]
    cov = ins_metrics[2]
    wCov = ins_metrics[3]

    return oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov, sem_metrics_classbyclass


def pred_on_dataset(sess, ops, dataset, save_folder, num_classes, mean_num_pts_in_group, save_cubes=None, bandwidth=1):
    """
    Function to make predictions.
    It uses the data from dataset in order by taking batches of the indicated size.
    If save_folder parametre is specified, the segmented point clouds are saved there.
    Mean-shift clustering is used with the specified bandwith to generate instance objects and merge instances
    of different blocksby using BlockMerging algorithm (DOI 10.1109/CVPR.2018.00272).

    :param sess: tensorflow session.
    :param ops: dict mapping from string to tf ops.
    :param train_writer: tf.summary.FileWriter.
    :param dataset: GeneralDataset obeject with the dataset used for pred.
    :param save_folder: folder for saving the point clouds segmented [default: None].
    :param save_cubes: folder for saving the cubes of each point cloud [default: None].
    :param bandwidth: parametre used by the mean-shift clustering algorithm for the grouping of instances [default: 1].
    """
    
    is_training = False

    for i in range(len(dataset)):

        # Load data
        raw_data = dataset[i]

        cur_pred_sem = np.zeros(raw_data[:,:,0].shape, dtype='int')
        cur_pred_sem_softmax = np.zeros([raw_data.shape[0], raw_data.shape[1], num_classes])
        group_output = np.zeros(raw_data[:,:,0].shape, dtype='int')


        gap = 0.1
        cur_data = raw_data - raw_data.reshape(-1,3).min(axis=0) # move min to 0,0,0
        range_xyz = cur_data.max()
        volume_num = int(range_xyz / gap) + 1
        volume = -1 * np.ones([volume_num, volume_num, volume_num], dtype='int32')
        volume_seg = -1 * np.ones([volume_num, volume_num, volume_num], dtype='int32')

        # Split in batches
        for j in range(0, len(raw_data)):

            # Seletc batch
            coordinates_norm = raw_data[j]
            pts = cur_data[j]

            # Move points to the origin and resize to values [0,1]
            coordinates_norm = (coordinates_norm - coordinates_norm.min(axis=0))/dataset.cube_size

            feed_dict = {ops['pointclouds_pl']: coordinates_norm.reshape(1,coordinates_norm.shape[0], coordinates_norm.shape[1]),
                            ops['is_training_pl']: is_training}
        
            # Prediction
            pred_ins_val, pred_sem_label_val, pred_sem_softmax_val = sess.run(
                [ops['pred_ins'], ops['pred_sem_label'], ops['pred_sem_softmax']], feed_dict=feed_dict)

            pred_val = np.squeeze(pred_ins_val, axis=0)
            pred_sem = np.squeeze(pred_sem_label_val, axis=0)
            pred_sem_softmax = np.squeeze(pred_sem_softmax_val, axis=0)
            cur_pred_sem[j, :] = pred_sem
            cur_pred_sem_softmax[j, ...] = pred_sem_softmax

            # cluster
            group_seg = {}
            num_clusters, labels, cluster_centers = cluster(pred_val, bandwidth)
            for idx_cluster in range(num_clusters):
                tmp = (labels == idx_cluster)
                estimated_seg = int(stats.mode(pred_sem[tmp])[0])
                group_seg[idx_cluster] = estimated_seg

            groupids_block = labels

            groupids = BlockMerging(volume, volume_seg, pts,
                                    groupids_block.astype(np.int32), group_seg, gap)

            group_output[j, :] = groupids

            # Save each cube as a point cloud
            if save_cubes != None:
                file_name = pathlib.PurePath(save_cubes).joinpath(pathlib.PurePath(dataset.files[i]).stem + '_' + str(j) + pathlib.PurePath(dataset.files[i]).suffix)
                save_las(str(file_name), raw_data[j], cur_pred_sem[j], group_output[j], semantic_softmax=np.max(pred_sem_softmax, axis=1))


        group_pred = group_output.reshape(-1)
        seg_pred = cur_pred_sem.reshape(-1)
        seg_pred_softmax = cur_pred_sem_softmax.reshape([-1, num_classes])
        pts = cur_data.reshape([-1, 3])

        # filtering
        x = (pts[:, 0] / gap).astype(np.int32)
        y = (pts[:, 1] / gap).astype(np.int32)
        z = (pts[:, 2] / gap).astype(np.int32)
        for j in range(group_pred.shape[0]):
            if volume[x[j], y[j], z[j]] != -1:
                group_pred[j] = volume[x[j], y[j], z[j]]

        un = np.unique(group_pred)
        pts_in_pred = [[] for itmp in range(num_classes)]
        group_pred_final = -1 * np.ones_like(group_pred)
        grouppred_cnt = 0
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (group_pred == g)
            sem_seg_g = int(stats.mode(seg_pred[tmp])[0])
            # if np.sum(tmp) > 500:
            if np.sum(tmp) > 0.25 * mean_num_pts_in_group[sem_seg_g]:
                group_pred_final[tmp] = grouppred_cnt
                pts_in_pred[sem_seg_g] += [tmp]
                grouppred_cnt += 1

        ins = group_pred_final.astype(np.int32)
        sem = seg_pred.astype(np.int32)
        sem_softmax = seg_pred_softmax

        # Adapt variables to the rest of the code
        raw_data = raw_data.reshape(-1,3)
        sem_pred = sem
        inst_pred = ins


        # save point cloud
        if save_folder != None:
            file_name = pathlib.PurePath(save_folder).joinpath(pathlib.PurePath(dataset.files[i]).name)
            save_las(str(file_name), raw_data, sem_pred, inst_pred, semantic_softmax=np.max(sem_softmax, axis=1))

    return 