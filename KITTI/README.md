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
[Download]([https://knightsucfedu39751-my.sharepoint.com/:f:/g/personal/jyoti_kini_knights_ucf_edu/Eqz1oFgLZcFBu4Wvk8x84PwB9Z1RVY9NSbmWEe654ERtWA?e=zw6raH
](https://ucf-my.sharepoint.com/personal/jy435956_ucf_edu/_layouts/15/onedrive.aspx?e=5%3Ae0e74817d279417b9a7cb2eb531ff7fe&sharingv2=true&fromShare=true&at=9&CT=1722973339638&OR=OWA%2DNT%2DMail&CID=1a612cd6%2Da351%2D4ec3%2D6071%2Dd7e022b4f91f&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNDA3MTkwMDIuMjMiLCJPUyI6IkxpbnV4IHVuZGVmaW5lZCJ9&cidOR=Client&id=%2Fpersonal%2Fjy435956%5Fucf%5Fedu%2FDocuments%2FPC%2DDAN&FolderCTID=0x012000024558296AB9334480041ED5625BCEF1&view=0)) and add to **'data'** folder

#### Train
Use below command in terminal:
```
python -u main.py
```

#### Validation
Download and add pre-trained weights to **'results'** folder <br/>
[Supervised]([https://knightsucfedu39751-my.sharepoint.com/:u:/g/personal/jyoti_kini_knights_ucf_edu/EevLRCiXvpZPhquIZ2FNXKMBGJEs6dkAbus8947ACACx9A?e=elsDBI](https://ucf-my.sharepoint.com/personal/jy435956_ucf_edu/_layouts/15/onedrive.aspx?e=5%3Ae0e74817d279417b9a7cb2eb531ff7fe&sharingv2=true&fromShare=true&at=9&CT=1722973339638&OR=OWA%2DNT%2DMail&CID=1a612cd6%2Da351%2D4ec3%2D6071%2Dd7e022b4f91f&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNDA3MTkwMDIuMjMiLCJPUyI6IkxpbnV4IHVuZGVmaW5lZCJ9&cidOR=Client&id=%2Fpersonal%2Fjy435956%5Fucf%5Fedu%2FDocuments%2FPC%2DDAN%2Fweights&FolderCTID=0x012000024558296AB9334480041ED5625BCEF1&view=0) <br/>
[Self-supervised]([https://knightsucfedu39751-my.sharepoint.com/:f:/g/personal/jyoti_kini_knights_ucf_edu/EnKg8n4iRkZDqmKoThEeFZIBas4myhgI7L9Gjia92g_b0g?e=VRnaRD](https://ucf-my.sharepoint.com/personal/jy435956_ucf_edu/_layouts/15/onedrive.aspx?e=5%3Ae0e74817d279417b9a7cb2eb531ff7fe&sharingv2=true&fromShare=true&at=9&CT=1722973339638&OR=OWA%2DNT%2DMail&CID=1a612cd6%2Da351%2D4ec3%2D6071%2Dd7e022b4f91f&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNDA3MTkwMDIuMjMiLCJPUyI6IkxpbnV4IHVuZGVmaW5lZCJ9&cidOR=Client&id=%2Fpersonal%2Fjy435956%5Fucf%5Fedu%2FDocuments%2FPC%2DDAN%2Fweights&FolderCTID=0x012000024558296AB9334480041ED5625BCEF1&view=0)

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
