# Self-Supervised-Affinity-MOT
Self supervised deep affinity network for multi object tracking in EO

> Create a new environment named **SS_DAN**, then run the following script:
>
> ```shell
> conda create -n SS_DAN python=3.5
> source activate SS_DAN
> pip install -r requirement.txt
> ```

## Train & Test On MOT17
### Download dataset

1. Download the [mot 17 dataset 5.5 GB](https://motchallenge.net/data/MOT17.zip) and [development kit 0.5 MB](https://motchallenge.net/data/devkit.zip).

2. Unzip this the dataset. Here is my file structure.

   ```shell
   MOT17
   ├── test
   └── train
   ```
   
### Test

1. Download the weigths from [*OneDrive*](ToDo) to the **weights** folder

2. Modify **config/config.py** as follows:

   ```python
   # You need to modify line 8, 72, 73 and 74.
   8	current_select_configure = 'init_test_mot17' # need use 'init_test_mot17'
   ...	...
   70	def init_test_mot17():
   71        config['resume'] = './weights/model_name.pth'
   72        config['mot_root'] = 'replace by your dataset folder' 
   73		  config['save_folder'] = 'replace by your save folder'
   74        config['log_folder'] = 'replace by your log folder'
   75        config['batch_size'] = 1
   76        config['write_file'] = True
   77        config['tensorboard'] = True
   78        config['save_combine'] = False
   79        config['type'] = 'test' # or 'train'
   ```

3. run *test_mot17.py*

   ```shell
   python test_mot17.py
   ```

### Train

1. Modify **config/config.py** as follows:

   ```python
   # you need to modify line 8, 87, 89, 90 and 91.
   8	current_select_configure = 'init_train_mot17' # need use 'init_train_mot17'
   ...	...
   85	def init_train_mot17():
   86		config['epoch_size'] = 664
   87		config['mot_root'] = 'replace by your mot17 dataset folder'
   89		config['log_folder'] = 'replace by your log folder'
   90		config['save_folder'] = 'replace by your save folder'
   91		config['save_images_folder'] = 'replace by your image save folder'
   92		config['type'] = 'train'
   93		config['resume'] = None # None means training from sketch.
   94		config['detector'] = 'SDP'
   95		config['start_iter'] = 0
   96		config['iteration_epoch_num'] = 120
   97		config['iterations'] = config['start_iter'] + config['epoch_size'] *     config['iteration_epoch_num'] + 50
   98		config['batch_size'] = 4
   99		config['learning_rate'] = 1e-2
   100		config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
   101		config['save_weight_every_epoch_num'] = 5
   102		config['min_gap_frame'] = 0
   103		config['max_gap_frame'] = 30
   104		config['false_constant'] = 10
   105		config['num_workers'] = 16
   106		config['cuda'] = True
   107		config['max_object'] = 80
   108		config['min_visibility'] = 0.3
   ```

3. Run *train_mot17.py*

   ```shell
   python train_ss_mot17.py
   ```



## Acknowledgement

This code is based on [**SST**](https://github.com/shijieS/SST)

## Citation
> Sun. S., Akhtar, N., Song, H.,  Mian A., & Shah M. (2018). Deep Affinity Network for Multiple Object Tracking, Retrieved from [https://arxiv.org/abs/1810.11780](https://arxiv.org/abs/1810.11780)

