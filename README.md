# YDM-SLAM: YOLOv8-powered Dynamic Mapping of Environment with ORB-SLAM3

**This is an improved version of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) that adds an semantic mask-based object detection segmentation module implemented with [YOLOv8-Seg](https://docs.ultralytics.com/tasks/segment/) to achieve SLAM in dynamic environments.**
- Object Detection
- Instance Segmentation
- Dynamic SLAM

# Publication
https://ieeexplore.ieee.org/abstract/document/10691024

# Getting Started

## 0. Prerequisites

We have tested on:

> C++14
> 
> OS = Ubuntu 20.04
> 
> [OpenCV](http://opencv.org) = 4.2
> 
> [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) = 3.3.9
>
> [Pangolin](https://github.com/stevenlovegrove/Pangolin) = 0.5
>


## 1. Download ONNX Runtime

You can download the compatible version of onnx-runtime from [Microsoft ONNX-Runtime](https://github.com/microsoft/onnxruntime/releases/)
We tested on onnxruntime-linux-x64-gpu-1.9.0

then extract the folder using the below command to third-party folder and update <path_to_onnxruntime> in CMakeLists.txt
```bash
tar -xzvf onnxruntime-linux-x64-gpu-1.9.0.tgz -C PATH/YDM-SLAM/Thirdparty
```

## 2. Build
```bash
cd YDM-SLAM
chmod +x build.sh
./build.sh
```

Only the rgbd_tum target will be build.

## 3. Execute

#### TUM-RGBD Dataset
Download any dynamic sequence from [TUM-RGBD dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```

## 4. Demo
[![Watch the Demo Video Here](https://github.com/user-attachments/assets/91f1dc26-fd0d-4b2b-8218-9372cc1c0ee7)](https://youtu.be/J7V5pD78oNM)

