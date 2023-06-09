#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>

#define YOLO_P6 false // whether to use the P6 model
#define ORT_OLD_VISON 12  // ort1.12.0 The previous version is the old version of API


struct OutputSeg {
	int id;
	float confidence;
	cv::Rect box;
	cv::Mat boxMask;       // rectangular box inside mask, save memory space and speed up
    std::vector<cv::Point2f> kps;
    std::vector<float> kps_score;
};


struct MaskParams {
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

struct PoseParams {
	float kpsThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};


bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);
void DrawPred(cv::Mat& img, std::vector<OutputSeg> result, std::vector<std::string> classNames, std::vector<cv::Scalar> color);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, // [ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
void GetKPS2(const std::vector<float>& kpsProposals, OutputSeg& output, const PoseParams& poseParams);
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams);
