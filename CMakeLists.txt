CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(YOLOv8)


#SET (OpenCV_DIR path/to/opencv/build)  #opencv root
#SET (ONNXRUNTIME_DIR path/to/onnxruntime)

FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)
#include_directories("")
ADD_EXECUTABLE(YOLOv8 
    ${CMAKE_SOURCE_DIR}/include/yolov8_det.h
    ${CMAKE_SOURCE_DIR}/include/yolov8_det_dfl.h
    #${CMAKE_SOURCE_DIR}/include/yolov8_onnx.h
    ${CMAKE_SOURCE_DIR}/include/yolov8_seg.h
    ${CMAKE_SOURCE_DIR}/include/yolov8_seg_dfl.h
    ${CMAKE_SOURCE_DIR}/include/yolov8_pose_dfl.h
    #${CMAKE_SOURCE_DIR}/include/yolov8_seg_onnx.h
    ${CMAKE_SOURCE_DIR}/include/yolov8_utils.h
    ${CMAKE_SOURCE_DIR}/src/yolov8_det.cpp
    ${CMAKE_SOURCE_DIR}/src/yolov8_det_dfl.cpp
    #${CMAKE_SOURCE_DIR}/src/yolov8_onnx.cpp
    ${CMAKE_SOURCE_DIR}/src/yolov8_seg.cpp
    ${CMAKE_SOURCE_DIR}/src/yolov8_seg_dfl.cpp
    ${CMAKE_SOURCE_DIR}/src/yolov8_pose_dfl.cpp
    #${CMAKE_SOURCE_DIR}/src/yolov8_seg_onnx.cpp
    ${CMAKE_SOURCE_DIR}/src/yolov8_utils.cpp
		main.cpp)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

TARGET_INCLUDE_DIRECTORIES(YOLOv8 PRIVATE "${ONNXRUNTIME_DIR}/include")

TARGET_COMPILE_FEATURES(YOLOv8 PRIVATE cxx_std_14)
TARGET_LINK_LIBRARIES(YOLOv8 ${OpenCV_LIBS})

#if (WIN32)
#    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
#endif(WIN32)
#
#if (UNIX)
#    TARGET_LINK_LIBRARIES(YOLOv8 "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
#endif(UNIX)
