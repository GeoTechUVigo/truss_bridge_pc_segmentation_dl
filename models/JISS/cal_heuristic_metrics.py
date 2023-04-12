import metrics
import pathlib
from laspy.file import File
import numpy as np
import csv
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(ROOT_DIR.joinpath('utils')))

from save_las import save_las


folder_pred = "data/test_heuristic"
folder_real = "data/synthetic_point_clouds"
log_dir = "data/heuristic_metrics"
folder_errors = "data/test_heuristic/errors"
suffix = '.las'
heuristic_method=True
DECIMALS = 4

log_dir = pathlib.Path(log_dir)
folder_pred = pathlib.Path(folder_pred)
folder_real = pathlib.Path(folder_real)
folder_errors = pathlib.Path(folder_errors)

# metrics
sem_metrics_sum = np.zeros(3)
ins_metrics_sum = np.zeros(4)

for point_cloud_path in folder_pred.glob('*' + suffix):
    # prediction
    point_cloud = File(str(point_cloud_path), mode = "r")
    
    ins_pred = point_cloud.pt_src_id if heuristic_method else point_cloud.user_data
    sem_pred_ = point_cloud.Classification

    # real
    point_cloud_path = folder_real.joinpath(point_cloud_path.name)
    point_cloud = File(str(point_cloud_path), mode = "r")

    ins_real = point_cloud.user_data
    sem_real = point_cloud.Classification

    if heuristic_method:
        # Classification labels heurist method.
        # Vertical face Lateral -> 1
        # Vertical face Vertical -> 2
        # Chord -> 3
        # Horizontal face Lateral -> 4
        # Horizontal face Vertical -> 5
        # Inner face Lateral -> 6
        # Inner face Horizontal -> 7

        # Adapt labels to real labels
        sem_pred = np.zeros(sem_pred_.shape)
        sem_pred[sem_pred_==1] = 3
        sem_pred[sem_pred_==2] = 2
        sem_pred[sem_pred_==3] = 1
        sem_pred[sem_pred_==4] = 3
        sem_pred[sem_pred_==5] = 2
        sem_pred[sem_pred_==6] = 3
        sem_pred[sem_pred_==7] = 2

        # remove deck points since the heuristic method does not segment it.
        deck_points = sem_real==0
        ins_pred = ins_pred[~deck_points]
        sem_pred = sem_pred[~deck_points]
        ins_real = ins_real[~deck_points]
        sem_real = sem_real[~deck_points]

    # metrics
    mPrec, mRecall, cov, wCov, errors_ins = metrics.instance_metrics(ins_real, ins_pred)
    ins_metrics_sum += mPrec, mRecall, cov, wCov

    oAcc, mAcc, mIoU, errors_sem = metrics.semantic_metrics(sem_real, sem_pred)
    sem_metrics_sum += oAcc, mAcc, mIoU

    # Add label 3 for not analysed points
    errors_ins = errors_ins.astype('int')
    errors_sem = errors_sem.astype('int')
    
    errors_ins[sem_pred==0] = 2
    errors_sem[sem_pred==0] = 2

    save_las(str(folder_errors.joinpath(point_cloud_path.name)), np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()[~deck_points], errors_sem, errors_ins)


sem_metrics = sem_metrics_sum / len(list(folder_pred.glob('*' + suffix)))
ins_metrics = ins_metrics_sum / len(list(folder_pred.glob('*' + suffix)))

metrics = np.concatenate((sem_metrics, ins_metrics))

test_metrics_path = log_dir.joinpath('metrics_test.csv')
header = ["oAcc", "mAcc", "mIoU", "mPrec", "mRec", "cov", "wCov", str(folder_pred)]
with open(str(test_metrics_path), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    csvfile.close()

with open(str(test_metrics_path), 'a') as csvfile:
    writer = csv.writer(csvfile)
    metrics_csv = ["{:0.{precision}f}".format(v, precision=DECIMALS) for v in metrics]
    writer.writerow(metrics_csv)
    csvfile.close()