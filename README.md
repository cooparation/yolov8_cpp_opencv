# YOLOv8 deployment
the project code heavily borrowed from: https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp and https://github.com/triple-Mu/ncnn-examples
This is the C++ deployment implementation of yolov8, using OpenCV, the df module of yolov8 detection branch in the post-processing is implemented in the post-processing, 
which can be deployed on the Horizon X3 board. To train or convert the yolov8 model, you can refer to the implementation of [triple-mu](https://github.com/triple-Mu/yolov8) with branch triplemu/x3pi

## Implemented
* detection
* instance segmentation
* pose estimation
 
## requirements: opencv-dn:
 > OpenCV >= 4.5.5<br>

## build and run
    *   build:
    ```
    mkdir build
    cd build && cmake .. && make -j8
    ```
    * run
    ```
    ./build/YOLOv8 seg_dfl
    ```
