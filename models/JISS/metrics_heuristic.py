import metrics
import pathlib
from laspy.file import File
import csv
import numpy as np
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.resolve().parent.parent

sys.path.append(str(BASE_DIR))
sys.path.append(str(ROOT_DIR.joinpath('utils')))

from save_las import save_las


dataset_heuristic = "/workspaces/truss_bridge_pc_segmentation_dl/data/segmented_heuristic/original"
dataset_gt = "/workspaces/truss_bridge_pc_segmentation_dl/data/segmented_heuristic/gt"

save_folder = "/workspaces/truss_bridge_pc_segmentation_dl/data/segmented_heuristic/remaped_2"
path_metrics = "/workspaces/truss_bridge_pc_segmentation_dl/data/segmented_heuristic/metrics.csv"
NUM_CLASSES = 4
DECIMALS = 4

dataset_heuristic = pathlib.Path(dataset_heuristic)
dataset_gt = pathlib.Path(dataset_gt)
save_folder = pathlib.Path(save_folder)
path_metrics = pathlib.Path(path_metrics)

# CSV
header = ["oAcc", "mAcc", "mIoU", "mPrec", "mRec", "cov", "wCov"]
for num in range(NUM_CLASSES):
    header.append('acc_'+str(num))
header.append("file")

# CSV for saving val metrics
with open(str(path_metrics), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    csvfile.close()

for file_name in dataset_heuristic.iterdir():

    # load
    point_cloud_heuristic = File(str(file_name), mode="r")

    # load gt
    gt = File(str(dataset_gt.joinpath(file_name.name)), mode="r")
    
    # remap heuristic
    classification = np.zeros(point_cloud_heuristic.classification.shape)
    classification[point_cloud_heuristic.classification == 1] = 3
    classification[point_cloud_heuristic.classification == 2] = 2
    classification[point_cloud_heuristic.classification == 3] = 1
    classification[point_cloud_heuristic.classification == 4] = 3
    classification[point_cloud_heuristic.classification == 5] = 2
    classification[point_cloud_heuristic.classification == 6] = 3
    classification[point_cloud_heuristic.classification == 7] = 1

    mask= np.ones(point_cloud_heuristic.classification.shape, dtype=np.bool_)
    mask[gt.classification == 0] = 0

    # Instance metrics
    mPrec, mRec, cov, wCov, errors_ins_mask = metrics.instance_metrics(gt.instances[mask], point_cloud_heuristic.pt_src_id[mask])
    # Semantic metrics
    oAcc, mAcc, mIoU, accs, errors_sem_mask = metrics.semantic_metrics(gt.classification[mask], classification[mask], NUM_CLASSES)

    errors_ins = np.zeros(point_cloud_heuristic.classification.shape, dtype=np.int_)
    errors_ins[mask] = errors_ins_mask.astype(np.int_)
    errors_ins[~mask] = 2
    errors_sem = np.zeros(point_cloud_heuristic.classification.shape, dtype=np.int_)
    errors_sem[mask] = errors_sem_mask.astype(np.int_)
    errors_sem[~mask] = 2

    # save point cloud
    if save_folder != None:
        file_name = pathlib.PurePath(save_folder).joinpath(file_name.name)
        xyz = np.zeros((len(point_cloud_heuristic),3))
        xyz[:, 0] = point_cloud_heuristic.x
        xyz[:, 1] = point_cloud_heuristic.y
        xyz[:, 2] = point_cloud_heuristic.z
        
        save_las(str(file_name), xyz, classification, point_cloud_heuristic.pt_src_id, errors_sem, errors_ins)

    if path_metrics != None:
        # Save in test metrics
        calc_metrics = np.array([oAcc, mAcc, mIoU, mPrec, mRec, cov, wCov])
        calc_metrics = np.concatenate((calc_metrics, accs))
        with open(str(path_metrics), 'a') as csvfile:
                writer = csv.writer(csvfile)
                metrics_csv = ["{:0.{precision}f}".format(v, precision=DECIMALS) for v in calc_metrics]
                metrics_csv.append(file_name.name)
                writer.writerow(metrics_csv)
                csvfile.close()
