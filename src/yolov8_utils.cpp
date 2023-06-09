#include "yolov8_utils.h"

using namespace cv;
using namespace std;

const static unsigned char KPS_COLORS[17][3] = {
    {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
    {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0},
    {51, 153, 255},{51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}
};

const static unsigned char SKELETON[19][2] = {
    {16, 14}, {14, 12}, {17, 15},
    {15, 13}, {12, 13}, {6, 12},
    {7, 13}, {6, 7}, {6, 8},
    {7, 9}, {8, 10}, {9, 11},
    {2, 3}, {1, 2},  {1, 3},
    {2, 4}, {3, 5}, {4, 6},
    {5, 7}
};

const static unsigned char LIMB_COLORS[19][3] = {
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
    {51, 153, 255}, {255, 51, 255}, {255, 51, 255},
    {255, 51, 255}, {255, 128, 0}, {255, 128, 0},
    {255, 128, 0}, {255, 128, 0}, {255, 128, 0},
    {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
    {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
    {0, 255, 0}
};


bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize) {
	if (netHeight % netStride[strideSize - 1] != 0 || netWidth % netStride[strideSize - 1] != 0)
	{
		cout << "Error:_netHeight and _netWidth must be multiple of max stride " << netStride[strideSize - 1] << "!" << endl;
		return false;
	}
	return true;
}


void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void GetKPS2(const std::vector<float>& kpsProposals, OutputSeg& output, const PoseParams& poseParams) {
	float pose_threshold = poseParams.kpsThreshold;
	Vec4f params = poseParams.params;
	Size src_img_shape = poseParams.srcImgShape;

	Rect temp_rect = output.box;
	// crop from mask_protos
	//int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	//int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	//int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
    //int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;
    std::vector<cv::Point2f> kps_tmp;
    std::vector<float> kps_score_tmp;
    for(int i=0; i<kpsProposals.size(); i+=3){ // 51 = 17X3
        int px = floor((kpsProposals[i] - params[2]) / params[0]);
        int py = floor((kpsProposals[i+1] - params[3]) / params[1]);
        float ps = kpsProposals[i+2];
        cv::Point2f a;
        a.x = px; a.y = py;
        kps_tmp.push_back(a);
        kps_score_tmp.push_back(ps);
    }
    cout << "kps num:" << kps_score_tmp.size() << endl; ;
    output.kps = kps_tmp;
    output.kps_score = kps_score_tmp;

}


void GetMask2(const Mat& maskProposals, const Mat& mask_protos, OutputSeg& output, const MaskParams& maskParams) {
	int seg_channels = maskParams.segChannels;
	int net_width = maskParams.netWidth;
	int seg_width = maskParams.segWidth;
	int net_height = maskParams.netHeight;
	int seg_height = maskParams.segHeight;
	float mask_threshold = maskParams.maskThreshold;
	Vec4f params = maskParams.params;
	Size src_img_shape = maskParams.srcImgShape;

	Rect temp_rect = output.box;
	// crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height) {
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	vector<Range> roi_rangs;
	roi_rangs.push_back(Range(0, 1));
	roi_rangs.push_back(Range::all());
	roi_rangs.push_back(Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(Range(rang_x, rang_w + rang_x));

	// crop
	Mat temp_mask_protos = mask_protos(roi_rangs).clone();
	Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	Mat matmul_res = (maskProposals * protos).t();
	Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	Mat dest, mask;

	// sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	resize(dest, mask, Size(width, height), INTER_NEAREST);
	mask = mask(temp_rect - Point(left, top)) > mask_threshold;
	output.boxMask = mask;
}


void DrawPred(Mat& img, vector<OutputSeg> result, std::vector<std::string> classNames, vector<Scalar> color) {
	Mat mask = img.clone();
    cout << "DrawPred:"<<result.size()<<endl;
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
        // draw mask
		if(result[i].boxMask.rows&& result[i].boxMask.cols>0)
			mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);

        // draw pose kps
        bool have_kps = false;
        for(int k = 0; k<result[i].kps.size(); k++){ // 17 points
            if(result[i].kps_score[k] > 0.5){
                circle(img, result[i].kps[k], 3, cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]), -1);
                have_kps = true;
            }
        }
        // draw skeleton
        if(have_kps){
        for(int line_i=0; line_i<19; line_i++){ // 19 skeletons
            int xi = SKELETON[line_i][0] - 1; // start point
            int yi = SKELETON[line_i][1] - 1; // end point
            if(result[i].kps_score[xi] >0.5 && result[i].kps_score[yi] >0.5){
                cv::Point2f pos1 = result[i].kps[xi];
                cv::Point2f pos2 = result[i].kps[yi];
                cv::line(img, pos1, pos2, cv::Scalar(LIMB_COLORS[line_i][0], LIMB_COLORS[line_i][1], LIMB_COLORS[line_i][2]), 2);
            }
        }}

		string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img); // add mask to src
	imshow("1", img);
	//imwrite("out.bmp", img);
	waitKey();
}
