# PC-DAN: Point Cloud based Deep Affinity Network for 3D Multi-Object Tracking

### Dependencies
cuda_10.0 <br/>
cudnn_v7.6.5

### Environmental setup
```
conda create -n pcdan python=3.7 cython
conda activate pcdan
conda install pytorch torchvision -c pytorch
conda install numba
conda install -c conda-forge tensorboardx
```

### Install other dependencies
```
pip install -r requirements.txt
```

### Usage
#### Config file
Update following variables in the **'config.yaml'** file:
```
data_root: # update path to kitti-dataset
pre_data_root: # update path to downloaded 'data' folder
```

#### Data
[Download](https://drive.google.com/drive/folders/1j5o1pklCGtgemG05mXCwQGHozALn3Jt2) and add to **'data'** folder

#### Train
Use below command in terminal:
```
python -u main.py
```

#### Validation
Download and add pre-trained weights to **'results'** folder <br/>
[Supervised](https://drive.google.com/drive/u/1/folders/1qm_SRWejbcCKaT9fCMqb9gnYnQGCpbZk) <br/>
[Self-supervised](https://drive.google.com/drive/u/1/folders/1qm_SRWejbcCKaT9fCMqb9gnYnQGCpbZk)

Use below command in terminal:
```
python -u eval_seq.py
```

#### Test
We use 2D [RRC detections](https://github.com/xiaohaoChen/rrc_detection) (Accurate Single Stage Detector Using Recurrent Rolling Convolution) and generate frustums during testing, hence we need to update the below mentioned **'config.yaml'** variables:
```
det_type: 2D
use_frustum: True
```
Use below command in terminal:
```
python -u test.py
```

### Citation
```
@article{kumar2021pc
	title={PC-DAN: Point Cloud based Deep Affinity Network for 3D Multi-Object Tracking},
	author={Kumar, Aakash and Kini, Jyoti and Shah, Mubarak and Mian, Ajmal},
	conference={IEEE Conference on Computer Vision and Pattern Recognition Workshops},
	year={2021}
}
```

### Acknowledgement
Our code benefits from the implementations by [Zhang et al.](https://github.com/ZwwWayne/mmMOT) (Robust Multi-Modality Multi-Object Tracking) and [Sun et al.](https://github.com/shijieS/SST) (DAN-Deep Affinity Network)
