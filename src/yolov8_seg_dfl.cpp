#include"yolov8_seg_dfl.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define MAX_STRIDE 32 // if yolov8-p6 model modify to 64

bool Yolov8_Seg_DFL::ReadModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNet(netPath);
#if CV_VERSION_MAJOR==4 &&CV_VERSION_MINOR==7&&CV_VERSION_REVISION==0
		net.enableWinograd(false);  //bug of opencv4.7.x in AVX only platform ,https://github.com/opencv/opencv/pull/23112 and https://github.com/opencv/opencv/issues/23080
		//net.enableWinograd(true);		//If your CPU supports AVX2, you can set it true to speed up
#endif
	}
	catch (const std::exception&) {
		return false;
	}

	if (isCuda) {
		//cuda
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //or DNN_TARGET_CUDA_FP16
	}
	else {
		//cpu
		cout << "Inference device: CPU" << endl;
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

static float softmax_dfl(
	const float* src,
	float* dst,
	int length
)
{
	float alpha = -FLT_MAX;
	for (int c = 0; c < length; c++)
	{
		float score = src[c];
		if (score > alpha)
		{
			alpha = score;
		}
	}

	float denominator = 0;
	float dis_sum = 0;
	for (int i = 0; i < length; ++i)
	{
		dst[i] = expf(src[i] - alpha);
		denominator += dst[i];
	}
	for (int i = 0; i < length; ++i)
	{
		dst[i] /= denominator;
		dis_sum += i * dst[i];
	}
	return dis_sum;
}

static void print_output_dims(const std::vector<cv::Mat>& feat_blob){
    for(int n =0; n<feat_blob.size(); n++){
        std::cout << "output " << n << " dims: ";
        for(int dim=0; dim<feat_blob[n].dims; dim++){
            int value = feat_blob[n].size[dim];
            std::cout << value << " ";
        }std::cout << std::endl;
    }
}

static void generate_proposals(
	int stride,
	const std::vector<cv::Mat>& feat_blob,
	const float prob_threshold,
	std::vector<OutputSeg>& objects,
    std::vector<std::vector<float>>& maskcoef
)
{
	const int reg_max = 16;
	float dst[16];
    print_output_dims(feat_blob);
   // for(int n =0; n<feat_blob.size(); n++){
   //     std::cout << "output " << n << " dims: ";
   //     for(int dim=0; dim<feat_blob[n].dims; dim++){
   //         int value = feat_blob[n].size[dim];
   //         std::cout << value << " ";
   //     }std::cout << std::endl;
   // }
    // score
    int batch_size = feat_blob[0].size[0];
    int num_grid_y = feat_blob[0].size[1];
    int num_grid_x = feat_blob[0].size[2];
    int num_feat_w = feat_blob[0].size[3];
	const int num_class = num_feat_w;
    //std::cout << " output dimes:" << feat_blob.dims << std::endl;
    std::cout << " dbox:" << batch_size<< " " << num_grid_x << " "<<num_grid_y <<" "<< num_feat_w << std::endl;

    // loc
    batch_size = feat_blob[1].size[0];
    num_grid_x = feat_blob[1].size[1];
    num_grid_y = feat_blob[1].size[2];
    num_feat_w = feat_blob[1].size[3];
    std::cout << " score:" << batch_size<< " " << num_grid_x << " "<<num_grid_y <<" "<< num_feat_w << std::endl;

    // mcoef
    //const int reg_max = feat_blob[2].size[3]/4; // 16=64/4

    const int prototypes_k = feat_blob[2].size[3]; // 32


    for(int b = 0; b < batch_size; b++){
	for (int i = 0; i < num_grid_x; i++)
	{
		for (int j = 0; j < num_grid_y; j++)
		{

            // score: [1, resolution, resolution, num_class]
            int id = feat_blob[0].step[0] * b + feat_blob[0].step[1] * i + feat_blob[0].step[2] * j; // + w * a.step[3];
            const float* matat_score = (float*)(feat_blob[0].data + id);

			int class_index = 0;
			float class_score = -FLT_MAX;
            //std::cout << "id "<< num_class << std::endl;
            //cout << "score: ==============";
			for (int c = 0; c < num_class; c++)
			{
				float score = matat_score[c];
                score = 1 / (1 + cv::exp(-score));
                //cout << " " << score;
				if (score > class_score)
				{
					class_index = c;
					class_score = score;
				}
			}
                //cout << endl;

            // box: [1, resolution, resolution, 4*reg_max]
            int id_box = feat_blob[1].step[0] * b + feat_blob[1].step[1] * i + feat_blob[1].step[2] * j; // + w * a.step[3];
            const float* matat_dbox = (float*)(feat_blob[1].data + id_box);
            // maskcoef: [1, resolution, resolution, 32]
            int id_mcoef = feat_blob[2].step[0] * b + feat_blob[2].step[1] * i + feat_blob[2].step[2] * j; // + w * a.step[3];
            const float* matat_mcoef = (float*)(feat_blob[2].data + id_mcoef);
			if (class_score >= prob_threshold){
				float x0 = j + 0.5f - softmax_dfl(matat_dbox, dst, 16);
				float y0 = i + 0.5f - softmax_dfl(matat_dbox + reg_max, dst, 16);
				float x1 = j + 0.5f + softmax_dfl(matat_dbox + 2 * reg_max, dst, 16);
				float y1 = i + 0.5f + softmax_dfl(matat_dbox + 3 * reg_max, dst, 16);

				x0 *= stride;
				y0 *= stride;
				x1 *= stride;
				y1 *= stride;

				OutputSeg obj;
				obj.box.x = x0;
				obj.box.y = y0;
				obj.box.width = x1 - x0;
				obj.box.height = y1 - y0;
				obj.id = class_index;
				obj.confidence = class_score;
				objects.push_back(obj);

                std::vector<float> temp_maskcoef(matat_mcoef, matat_mcoef + prototypes_k); // prototypes k = 32
                maskcoef.push_back(temp_maskcoef);

			}
		}
	}
    }
}


bool Yolov8_Seg_DFL::Detect(Mat& srcImg, Net& net, vector<OutputSeg>& output) {
	Mat blob;
	output.clear();
	int col = srcImg.cols;
	int row = srcImg.rows;
	Mat netInputImg;
	Vec4d params;
	LetterBox(srcImg, netInputImg, params, cv::Size(_netWidth, _netHeight));
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(0, 0, 0), true, false);
	//**************************************************************************************************************************************************/
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	// If there is no problem with other settings, but results are a lot different from  Python-onnx , you can try to use the following two sentences
	//
	//$ blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(104, 117, 123), true, false);
	//$ blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(_netWidth, _netHeight), cv::Scalar(114, 114,114), true, false);
	//****************************************************************************************************************************************************/
	net.setInput(blob);
	std::vector<cv::Mat> net_output_img;

    //vector<string> output_layer_names{"lbox", "lmcoef", "lscore", "mbox", "mmcoef", "mscore", "sbox", "smcoef", "sscore","proto"};
    std::vector<string> output_layer_names;
    output_layer_names = net.getUnconnectedOutLayersNames();
    std::cout << "model_output_name:" << std::endl;
    // lbox lmcoef lscore mbox mmcoef mscore proto sbox smcoef sscore
    for(int i=0; i<output_layer_names.size(); i++){
        std::cout << output_layer_names[i] << " ";
    } std::cout << std::endl;
	//net.forward(net_output_img, net.getUnconnectedOutLayersNames()); //get outputs
	net.forward(net_output_img, output_layer_names); //get outputs
	std::vector<int> class_ids;// res-class_id
	std::vector<float> confidences;// res-conf
	std::vector<cv::Rect> boxes;// res-box


    const std::vector<cv::Mat>& feat_blob = {
        // large: score, mask, loc
        net_output_img[0], net_output_img[1], net_output_img[2],
        // middle: score, mask, loc
        net_output_img[3], net_output_img[4], net_output_img[5],
        // prob
        net_output_img[6],
        // small: score, mask, loc
        net_output_img[7], net_output_img[8], net_output_img[9]
    };
    print_output_dims(net_output_img);

	std::vector<OutputSeg> proposals; // cls, box
    std::vector<vector<float>> mask_proposals; // mask
    //const int target_size = 640;
    // stride 8
    {
        std::vector<OutputSeg> objects8;
	    const std::vector<cv::Mat>& feat_blob = {net_output_img[0], net_output_img[2], net_output_img[1]}; 
        generate_proposals(8, feat_blob, _classThreshold, objects8, mask_proposals);
		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }
    // stride 16
    {
        std::vector<OutputSeg> objects16;
	    const std::vector<cv::Mat>& feat_blob = {net_output_img[3], net_output_img[5], net_output_img[4]};
        generate_proposals(16, feat_blob, _classThreshold, objects16, mask_proposals);
		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    // stride 32
    {
        std::vector<OutputSeg> objects32;
	    const std::vector<cv::Mat>& feat_blob = {net_output_img[7], net_output_img[9], net_output_img[8]};
        generate_proposals(32, feat_blob, _classThreshold, objects32, mask_proposals);
		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

	//NMS
	for (auto& pro : proposals){
		boxes.push_back(pro.box);
		confidences.push_back(pro.confidence);
		class_ids.push_back(pro.id);
	}
    std::vector<int> nms_result_index;
	NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result_index);
    // get results
    float ratio_w = params[0]; float ratio_h = params[1];
    int dw = params[2]; int dh = params[3];
    int orin_w = srcImg.cols; int orin_h = srcImg.rows;
	std::vector<vector<float>> temp_mask_proposals;
	Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
    cout << orin_w << orin_h << endl;
    std::vector<vector<float>> picked_mask_proposals;
	for (auto i : nms_result_index){
		auto& bbox = boxes[i];

        float x = (bbox.x - dw) / ratio_h;
        float y = (bbox.y - dh) / ratio_w;
        float w = bbox.width / ratio_w;
        float h = bbox.height / ratio_h;
        int left = MAX(int(x - 0.5 * w + 0.5), 0);
        int top = MAX(int(y - 0.5 * h + 0.5), 0);

		OutputSeg obj;
		obj.box.x = int(x);
		obj.box.y = int(y);
		obj.box.width = int(w);
		obj.box.height = int(h);
		obj.box = obj.box & holeImgRect;
		obj.id = class_ids[i];
		obj.confidence = confidences[i];
		output.push_back(obj);
        //cout << "box info: " << obj.id << " " << obj.confidence << " "
        //    <<  obj.box.x <<" "<< obj.box.y <<" " << obj.box.width << " "<< obj.box.height << endl;
        picked_mask_proposals.push_back(mask_proposals[i]);
	}
    MaskParams mask_params;
    mask_params.params = params;
    mask_params.srcImgShape = srcImg.size();
    for (int i = 0; i < picked_mask_proposals.size(); ++i) {
        GetMask2(cv::Mat(picked_mask_proposals[i]).t(), net_output_img[6], output[i], mask_params);
    }

    std::cout << "output size:" << output.size() << std::endl;
	if (output.size())
		return true;
	else
		return false;
}

