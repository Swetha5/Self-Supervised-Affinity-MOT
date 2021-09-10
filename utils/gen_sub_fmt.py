"""
@author swetha
"""
import numpy as np
import os


def gen_mot_submission_format(input_file, output_file):
    pred_mot = np.loadtxt(input_file, delimiter=' ')
    # 1-based framework
    for i in range(len(pred_mot)):
        pred_mot[i][0] += 1
        pred_mot[i][1] += 1

    np.savetxt(output_file, pred_mot.astype(int), fmt='%i', delimiter=', ')


if __name__ == '__main__':
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/train_ss_s2cyc_sdp_e0.8_a0.5_lr5_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/csm_sst_train_ss_frcnn_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/csm_sst_train_sdp_ssaug_rf_ang15_lr5_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/csm_sst_train_frcnn_ssaug_rf_ang15_lr5_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/train_ss_s2_cyc_frcnn_e0.8_a0.5_lr5_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/nwi/csm_sst_train_sdp_nwi_ssaug_rf_ang15_lr3_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/nwi/csm_sst_train_frcnn_nwi_ssaug_rf_ang15_lr3_b4_g2/'
    # pred_base_path = '/home/c3-0/swetha/runs/predictions/nwi/NEWTON/train_nwi_ssaug_rfbct2_rang15_alblur5_21_frcnn_lr3_b8_g2/'
    pred_base_path = '/home/c3-0/swetha/runs/predictions/nwi/NEWTON/train_nwi_ssaug_rfbct2_rang15_alblur5_21_sdp_lr3_b8_g2_w55112/'

    # train and test sequences
    test_files = ['MOT17-01-SDP.txt', 'MOT17-03-SDP.txt', 'MOT17-06-SDP.txt', 'MOT17-07-SDP.txt', 'MOT17-08-SDP.txt', 'MOT17-12-SDP.txt', 'MOT17-14-SDP.txt']
    # test_files = ['MOT17-01-FRCNN.txt', 'MOT17-03-FRCNN.txt', 'MOT17-06-FRCNN.txt', 'MOT17-07-FRCNN.txt', 'MOT17-08-FRCNN.txt', 'MOT17-12-FRCNN.txt', 'MOT17-14-FRCNN.txt']
    train_files = ['MOT17-02-SDP.txt', 'MOT17-04-SDP.txt', 'MOT17-05-SDP.txt', 'MOT17-09-SDP.txt', 'MOT17-10-SDP.txt', 'MOT17-11-SDP.txt', 'MOT17-13-SDP.txt']
    # train_files = ['MOT17-02-FRCNN.txt', 'MOT17-04-FRCNN.txt', 'MOT17-05-FRCNN.txt', 'MOT17-09-FRCNN.txt', 'MOT17-10-FRCNN.txt', 'MOT17-11-FRCNN.txt', 'MOT17-13-FRCNN.txt']
    suffix = '0_0_4_0_3_3/'
    out_path = pred_base_path + 'sub/'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for f in train_files:
        gen_mot_submission_format(pred_base_path+'train/'+suffix+f, out_path+f)

    for f in test_files:
        gen_mot_submission_format(pred_base_path+'test/'+suffix+f, out_path+f)