import laspy
import numpy as np


def save_las(path, location, semantic_seg, instance_seg, erros_sem=None, erros_ins=None, semantic_softmax=None, scale = 0.01):

    # Header of the LAS
    header = laspy.header.Header(file_version=1.4, point_format=6)

    # LAS object in write mode
    las = laspy.file.File(str(path), header=header, mode="w")

    if not semantic_softmax is None:
        las.define_new_dimension(name='sem_softmax', data_type=10, description='sem_softmax') # data_type=10 is double 8 bytes https://laspy.readthedocs.io/en/1.x/tut_part_3.html
    if not erros_sem is None:
        las.define_new_dimension(name='erros_sem', data_type=10, description='erros_sem')
    if not erros_ins is None:
        las.define_new_dimension(name='erros_ins', data_type=10, description='erros_ins') 
    # Offset and scale
    las.header.offset = np.mean(np.concatenate((location.max(axis=0).reshape(-1,1),location.min(axis=0).reshape(-1,1)),axis=1),axis=1).tolist()
    las.header.scale = [scale,scale,scale]

    # Wirte in las object.
    las.x = location[:,0]
    las.y = location[:,1]
    las.z = location[:,2]
    las.classification = semantic_seg
    las.user_data = instance_seg

    if not semantic_softmax is None:
        setattr(las,'sem_softmax', semantic_softmax)
    if not erros_sem is None:
        setattr(las, 'erros_sem', erros_sem)
    if not erros_ins is None:
        setattr(las,'erros_ins', erros_ins)        
    # Save
    las.close()
