# YOLOv8 deployment
the project code heavily borrowed from: https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp and https://github.com/triple-Mu/ncnn-examples
This is the C++ deployment implementation of yolov8, using OpenCV, the df module of yolov8 detection branch in the post-processing is implemented in the post-processing, 
which can be deployed on the Horizon X3 board. To train or convert the yolov8 model, you can refer to the implementation of [triple-mu](https://github.com/triple-Mu/yolov8) with branch triplemu/x3pi

## Implemented
* detection
* instance segmentation
* pose estimation
 
## requirements: opencv-dnn:
 > OpenCV >= 4.5.5<br>

## build and run
*  build:
    ```
    mkdir build
    cd build && cmake .. && make -j8
    ```
* run
    ```
    ./build/YOLOv8 seg_dfl
    ```
# When writing post-processing, always be careful to use [netron](https://netron.app/) to view the model structure
* seg without df branch, c++ details to see yolov8_seg_dfl.cpp
<img width="290" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/5785eb4a-6e3e-41df-9c7f-cd92aa1be2ee">
<img width="140" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/7f7c728e-3d93-43b7-9580-4e76736c99b5">
* det without df branch, c++ details to see yolov8_det_dfl.cpp
<img width="198" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/87af4120-2f50-428f-9fd9-62f5e919c44e">
<img width="134" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/d9d1766d-c8b5-4b8b-93f4-466485df9296">
* pose without df branch, c++ details to see yolov8_pose_dfl.cpp
<img width="290" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/7794b6bd-3edc-44af-82e3-6271ba144e89">
<img width="132" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/30918b4a-e838-41cc-b7d5-1c9a7eae9adb">
* df branch in det model
<img width="178" alt="image" src="https://github.com/cooparation/yolov8_cpp_opencv/assets/15029439/0d6e87bb-305e-4c2f-9cf6-f42107a124a5">

