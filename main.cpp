#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "yolov8_seg.h"
#include "yolov8_seg_dfl.h"
#include "yolov8_det.h"
#include "yolov8_det_dfl.h"
#include "yolov8_pose_dfl.h"

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8(_Tp& cls,Mat& img,string& model_path)
{
	Net net;
	if (cls.ReadModel(net, model_path, false)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}
	// generate random colors
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;


	if (cls.Detect(img, net, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	system("pause");
	return 0;
}

int main(int argc, char const* argv[]) {

	string img_path = "./images/zidane.jpg";
	string det_model_path = "./onnx_models/yolov8n.onnx";
	string seg_model_path = "./onnx_models/yolov8n-seg.onnx";
	string det_model_path_dfl = "./onnx_models/x3/yolov8n_x3.onnx";
	string seg_model_path_dfl = "./onnx_models/x3/yolov8n-seg_x3.onnx";
	string pose_model_path_dfl = "./onnx_models/x3/yolov8n-pose_x3.onnx";

	Mat img = imread(img_path);

    std::string task_name = "pose_dfl";

    if(argc == 2){
        task_name = argv[1];
    }else{
        cout << "Usage: "<<argv[0] << " det" <<
            "\ndefault is det, also you can try [det, det_dfl, seg, seg_dfl]"
            << endl;
    }

    if(task_name == "det"){
        Yolov8 task_det;
        yolov8(task_det, img, det_model_path);
    }
    if(task_name == "det_dfl"){
        Yolov8_Det_DFL task_det_x3;
        yolov8(task_det_x3, img, det_model_path_dfl);
    }
    if(task_name == "seg"){
        Yolov8Seg task_segment;
        yolov8(task_segment, img, seg_model_path);
    }
    if(task_name == "seg_dfl"){
        Yolov8_Seg_DFL task_segment_x3;
        yolov8(task_segment_x3, img, seg_model_path_dfl);
    }
    if(task_name == "pose_dfl"){
        Yolov8_Pose_DFL task_pose_x3;
        yolov8(task_pose_x3, img, pose_model_path_dfl);
    }

}
