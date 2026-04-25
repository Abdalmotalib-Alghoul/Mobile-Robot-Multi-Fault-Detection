# Mobile Robot Low-Cost Sensors Multi-Fault Detection

This repository contains the code and supplementary materials for the paper:

**Mobile Robot Low-Cost Sensors Multi-Fault Detection Experiment using Machine-Learning approach**

### Overview
A practical and reproducible framework for injecting and detecting five realistic faults in low-cost 2D LiDAR and IMU sensors for indoor mobile robot navigation using the Hello Stretch robot in ROS Noetic and Gazebo 11.

### Repository Contents
- `scripts/` — Fault injection Python nodes
- `Pipline/` — Feature extraction and XGBoost training scripts
- `config/` — YAML configuration files
- `launch/` — ROS launch files
- `worlds/` — Gazebo worlds
- `maps/` — Gazebo maps
- `Models/` — Trained XGBoost models
- `Features_49.pdf` — Detailed description of the 49 selected features
- `Datasets/` — Dataset folder (samples only)

### Datasets
The full dataset is available on Zenodo:  
**https://zenodo.org/records/19672705**

### Reproduction Instructions
To run the simulation:
1. Copy the relevant folders and files into `~/catkin_ws/src/stretch_ros/stretch_navigation`.
2. Run the desired automation script for fault injection.
  
Note: All scripts currently use hardcoded paths. You must edit the file and map paths in each script to match your own package location.

Example:
cd ~/catkin_ws/src/stretch_ros/stretch_navigation/scripts
python3 automate_data_collection_drift.py


### Citation
@misc{alghoul2026multifault,
  author       = {Abdalmotalib Alghoul and Alessandro Freddi and Andrea Monteriu},
  title        = {Mobile Robot Low-Cost Sensors Multi-Fault Detection Experiment using Machine-Learning approach},
  year         = {2026},
  howpublished = {https://github.com/Abdalmotalib-Alghoul/Mobile-Robot-Multi-Fault-Detection},
  note         = {Full datasets available at Zenodo: https://zenodo.org/records/19672705}
}


License
MIT License
