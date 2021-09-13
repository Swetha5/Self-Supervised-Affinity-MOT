

import numpy as np
import pandas as pd
def files_convert_to_JRDB_formate(read_dir,write_dir ):
    #read_dir=root_dir+'validation_results_no_eval_false/last/val/'+val[i]+'.txt'
    #write_dir=root_dir+'validation_results_no_eval_false/last/val/fixed_bbox_with_val_files/'+val[i]+'.txt'



    datatype = {0: int, 1: int, 2: str, 3: int, 4: int, 5: float,
                6: float, 7: float, 8: float, 9: float,
                10: float, 11: float, 12: float, 13: float, 14: float,
                15: float, 16: float}

    detection = pd.read_csv(read_dir, sep=' ', header=None, dtype=datatype)
    detection2=detection[[0, 1, 6,7,8,9,10,11,12,13,14,15,16,17]]
    detection2[17]=1
    

    rot_z=np.array(detection2[16])
    rotation_y=[]
    for rot in rot_z:
        rotation_y.append((-rot if rot < np.pi else 2 * np.pi - rot))
    detection2[16]=np.array(rotation_y)
    h=np.array(detection2[10])
    w=np.array(detection2[11])
    l=np.array(detection2[12])
    cx=np.array(detection2[13])
    cy=np.array(detection2[14])
    cz=np.array(detection2[15])

    n_cx=-cy

    n_cy= -cz + h/2
    n_cz=cx

    detection2[13]=l
    detection2[14]=h
    detection2[15]=w

    detection2[10]=n_cx
    detection2[11]=n_cy
    detection2[12]=n_cz
    
    detection2.to_csv(write_dir, header=None, index=None, sep=',', mode='a')