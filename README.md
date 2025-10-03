# Monocular Visual Odometry with YOLO-Based Dynamic Object Filtering on KITTI daatset

## Project Structure

project/
├── data/ 
│ ├── data_object_label_2/ # https://www.cvlibs.net/datasets/kitti/eval_object.php
│ │ └── training/
│ │ └── label_2/ 
│ ├── data_odometry_color/ # https://www.cvlibs.net/datasets/kitti/eval_odometry.php
│ │ └── dataset/
│ │ └── sequences/ 
│ ├── dataset_train_yolo/ # https://www.cvlibs.net/datasets/kitti/eval_object.php
│ │ ├── data.yaml
│ │ ├── train/
│ │ │ ├── images/
│ │ │ └── labels/
│ │ ├── val/
│ │ │ ├── images/
│ │ │ └── labels/
│ │ └── test/
│ │ ├── images/
│ │ └── labels/
│ └── ground_truth/ # https://www.cvlibs.net/datasets/kitti/eval_odometry.php
│ └── dataset/
│ └── poses/ # KITTI ground-truth poses
│
├── results/ # Outputs of the experiments
│ ├── evaluation/
│ │ ├── 00/
│ │ ├── 01/
│ │ ├── 02/
│ │ └── ... # per-sequence evaluation
│ ├── plots_results/
│ │ ├── 00/
│ │ ├── 01/
│ │ ├── 02/
│ │ └── ... # plots for each sequence
│ ├── txt_results/ # estimated poses in txt format
│ └── videos/ 
│
├── best.pt 
├── generate_results.py # main script to generate results
├── pose_evaluation_utils.py 
└── workspace.ipynb # main Jupyter notebook
