# JRDB-Dataset-Supervised

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

### Download Data:
Download Data from following link https://jrdb.stanford.edu/dataset/about

### Data Preparation (prepare_dataset.py):
edit prepare_dataset.py
run prepare_dataset.py after giving it path to the downloaded train and test dataset. Also give path where you want to save the processed data
Note: you will get following folders Sequences and Test_sequences which will contained lidar frames only.

### Download Weights:
https://drive.google.com/drive/folders/1Y_lv_JI7xsaLDQBvSB7gZ0ThMZ7mrviF?usp=sharing

### Download Validation Lables:
https://drive.google.com/drive/folders/1Y_lv_JI7xsaLDQBvSB7gZ0ThMZ7mrviF?usp=sharing
Validation lables int the txt files. 

### Update config file:
open config file, edit paths to root_dir, train_label_dir, test_label_dir, val_label_dir, load_weights, val_prediction_dir

root_dir takes the path to processed dataset (Train and Test data) genereted after running prepare_dataset.py
train_label_dir takes the path to labels in the downloded train dataset (JSON files). Path would be something like, 'train_dataset_with_activity/labels/labels_3d/'
test_label_dir takes the path to detections of downloaded test dataset(JSON files).Path would be something like, 'test_dataset_without_labels/detections/detections_3d/'
val_label_dir takes the path to downloaded validation labels from google drive. These are txt files used to compare validtion prediction with these ground truths.
load_weights takes to path to downloaded weights from google_drive. Path can be something like 'weights/ssj300_0712_epoch_60.0_Loss_ 0.0617.pth'
val_prediction_dir takes the path where you want to save your validation results 

### Perform evalulation:
Run eval_seq.py file. you will get the results in val_prediction_dir path defined in cofig file

### Traning:
You can train with main.py
Note: it takes around 60 epochs to get the desires results. you will get the weights on every epoch.

### Test: (Optional)
you can get the prediction on test data with test_seq.py file. Output will be in kitti and JRDB formate. 
To evaluate results on test files, you need to upload the JRDB formate files on leaderboard at JRDB website.

# JRDB-Dataset-Self-Supervised

You need to follow the same steps as above 
Just downloaded the saved weights from here https://drive.google.com/drive/folders/1Y_lv_JI7xsaLDQBvSB7gZ0ThMZ7mrviF?usp=sharing
Changes are made in the code to perform the self-supervised training. Changes are made in train_dataset.py from line 51 to 57. For self-supervised we only use single frame and make two samples out of it.

You can compare train_dataset.py of JRDB_supervised and JRDB_self_supervised from line 50 to 58

Moreover we use augmentation for self-supervised which you can see in get_pointcloud function of train_dataset_preprocess.py

you can also check by printing the det_info of dataloader which will give the information of which two frames are used train the model. In case of self-supervised two same frames are used.

Note: Your model will be trained in just 3 to 5 epochs. You can check evaluate the weights for these epochs
