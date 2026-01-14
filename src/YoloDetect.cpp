/*
* This file is part of YDM-SLAM
* Author: Balveer Singh
* GitHub: https://github.com/balveersinghyt/YDM-SLAM
*/
#include <iostream>
#include "yolov8_seg_onnx.h"
#include <opencv2/core.hpp> // For cv::Scalar
#include<time.h>
#include<opencv2/opencv.hpp>
#include <chrono>

#include <YoloDetect.h>

// NOTE: Keep the ONNXRuntime-backed model alive for process lifetime.
// This avoids a known shutdown-time crash where CUDA driver is already shutting down
// and ORT pinned memory cleanup throws (cudaFreeHost -> "driver shutting down").
static Yolov8SegOnnx* gYoloSegModel = nullptr;
YoloDetection::YoloDetection()
{
    std::cout << "Loading Yolo model..." << std::endl;

    std::string model_path_seg = "models/yolo11s-seg.onnx";

	// loading model
    if (!gYoloSegModel)
        gYoloSegModel = new Yolov8SegOnnx();

    if (gYoloSegModel->ReadModel(model_path_seg, true)) {
		std:: cout << "read net ok!" << endl;
	}
    else {
        std:: cout << "read net failed!" << endl;
		// return -1;
	}

    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};

    // Cache palette and kernel once (per-frame rand/srand and kernel alloc are expensive)
    mColors.reserve(80);
    std::srand(0);
    for (int i = 0; i < 80; i++)
    {
        int b = std::rand() % 256;
        int g = std::rand() % 256;
        int r = std::rand() % 256;
        mColors.push_back(cv::Scalar(b, g, r));
    }
    mDilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()
{

    const bool ydm_profile = (std::getenv("YDM_PROFILE") && std::string(std::getenv("YDM_PROFILE")) == "1");
    int ydm_profile_every = 30;
    if (const char* env_pe = std::getenv("YDM_PROFILE_EVERY"))
    {
        try { ydm_profile_every = std::stoi(std::string(env_pe)); } catch (...) { ydm_profile_every = 30; }
        if (ydm_profile_every < 1) ydm_profile_every = 1;
    }
    static int ydm_detect_calls = 0;
    static int ydm_window_calls = 0;
    static double ydm_sum_infer_ms = 0.0;
    static double ydm_sum_total_ms = 0.0;
    static int ydm_sum_det = 0;

    const auto ydm_t0_all = std::chrono::steady_clock::now();

    // loading image
    cv::Mat img;
    if(mRGB.empty())
    {
        std::cout << "Read image failed!" << std::endl;
        return -1;
    }
    cv:: cvtColor(mRGB, img, cv::COLOR_BGR2RGB);
    cv:: Mat image = img; // avoid per-frame clone; OnnxDetect should treat input as read-only
    mvDynamicArea.clear();
    mvDynamicMask.clear();
    mmDetectMap.clear();

    std::vector<OutputParams> result;
    objectMask = cv::Mat::zeros(image.size(), CV_8UC1);
    mInstanceIdMask = cv::Mat::zeros(image.size(), CV_32SC1);
    mvInstanceClassNames.clear();
    // std:: cout<< "objectMask size: " << objectMask.size() << " image size: " << image.size() << endl;

    const auto ydm_t0_infer = std::chrono::steady_clock::now();
    const bool ok_detect = (gYoloSegModel && gYoloSegModel->OnnxDetect(image, result));
    const auto ydm_t1_infer = std::chrono::steady_clock::now();
    const double ydm_ms_infer = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ydm_t1_infer - ydm_t0_infer).count();

    if (ok_detect) {
        mask = cv::Mat::zeros(image.size(), CV_8UC3);
        for (int i = 0; i < result.size(); i++) {
            int left, top;
            if (result[i].box.area() > 0) {
                rectangle(img, result[i].box, mColors[result[i].id % (int)mColors.size()], 2, 8);
                left = result[i].box.x;
                top = result[i].box.y;
            }
            if (result[i].rotatedBox.size.width * result[i].rotatedBox.size.height > 0) {
                DrawRotatedBox(img, result[i].rotatedBox, mColors[result[i].id % (int)mColors.size()], 2);
                left = result[i].rotatedBox.center.x;
                top = result[i].rotatedBox.center.y;
            }
            
            // add masked image for visualization
            if (result[i].boxMask.rows && result[i].boxMask.cols > 0){
                mask(result[i].box).setTo(mColors[result[i].id % (int)mColors.size()], result[i].boxMask);
            }

            // Safety check for class index
            if (gYoloSegModel && result[i].id >= 0 && result[i].id < gYoloSegModel->_className.size()) {
                const std::string &cls = gYoloSegModel->_className[result[i].id];

                // Instance-level outputs: assign IDs for ALL detected instances that have a valid mask
                if (result[i].box.area() > 0 && result[i].boxMask.rows && result[i].boxMask.cols > 0)
                {
                    const int instanceId = static_cast<int>(mvInstanceClassNames.size()) + 1;
                    mvInstanceClassNames.push_back(cls);
                    mInstanceIdMask(result[i].box).setTo(instanceId, result[i].boxMask);
                }

                // Include ALL detected objects (not just predefined dynamic classes)
                // This allows downstream logic (DynamicObjectCleaner + VerifyAndPruneFeatures)
                // to decide what is truly dynamic based on motion/depth evidence
                if (result[i].box.area() > 0 && result[i].boxMask.rows && result[i].boxMask.cols > 0)
                {
                    objectMask(result[i].box).setTo(cv::Scalar(255, 255, 255), result[i].boxMask);
                    cv::Rect2i DynamicArea(left, top, (result[i].box.width), (result[i].box.height));
                    mvDynamicArea.push_back(DynamicArea);
                }
                
                cv:: Rect2i DetectArea(left, top, (result[i].box.width), (result[i].box.height));
                mmDetectMap[gYoloSegModel->_className[result[i].id]].push_back(DetectArea);
            }
        }

        // Dilate the merged dynamic mask
        if (!mvDynamicArea.empty())
        {
            cv::dilate(objectMask, objectMask, mDilateKernel);
            mvDynamicMask.push_back(objectMask.clone());
        }
        else
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }
    }
    else
        cout << "Detect Failed!" << endl;

    // Optional profiling output (averaged, windowed)
    ydm_detect_calls++;
    const auto ydm_t1_all = std::chrono::steady_clock::now();
    const double ydm_ms_total = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ydm_t1_all - ydm_t0_all).count();
    ydm_window_calls++;
    ydm_sum_infer_ms += ydm_ms_infer;
    ydm_sum_total_ms += ydm_ms_total;
    ydm_sum_det += static_cast<int>(result.size());

    if (ydm_profile && ydm_window_calls >= ydm_profile_every)
    {
        const double avg_infer = ydm_sum_infer_ms / static_cast<double>(ydm_window_calls);
        const double avg_total = ydm_sum_total_ms / static_cast<double>(ydm_window_calls);
        const double avg_det = static_cast<double>(ydm_sum_det) / static_cast<double>(ydm_window_calls);
        std::cout << "[YDM][YOLO] call=" << ydm_detect_calls
                  << " every=" << ydm_profile_every
                  << " count=" << ydm_window_calls
                  << " avg_infer=" << avg_infer << "ms"
                  << " avg_total=" << avg_total << "ms"
                  << " avg_det=" << avg_det
                  << std::endl;

        ydm_window_calls = 0;
        ydm_sum_infer_ms = 0.0;
        ydm_sum_total_ms = 0.0;
        ydm_sum_det = 0;
    }
    
    return true;
}

void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}
