#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <onnxruntime_cxx_api.h>

// Minimal ONNX optical-flow wrapper.
// Expected model signature (configurable via env vars in Tracking.cc):
// - 2 inputs: image1, image2 (float32 NCHW), normalized to [-1, 1]
// - 1 output: flow (float32) shape [1,2,H,W] or [1,H,W,2] in pixels
class FlowNetOnnx {
public:
    FlowNetOnnx();
    ~FlowNetOnnx() noexcept;

    bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0);

    // Computes dense flow from img1 -> img2.
    // img1/img2: CV_8UC1 or CV_8UC3.
    // flow: CV_32FC2 with same HxW as inputs (after any internal padding is removed).
    bool InferFlow(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow);

    bool IsLoaded() const { return _session != nullptr; }

private:
    static cv::Mat To3CH(const cv::Mat& img);
    static void NormalizeToNCHW(const cv::Mat& bgr, std::vector<float>& out, int& H, int& W);

    // FastFlowNet-style preprocessing:
    // - input in [0,1] (BGR order, no RGB swap)
    // - centralize: subtract per-channel mean over both frames
    // - resize (not pad) to multiples of 64
    // - model input is typically [1,6,H,W] = concat(img1,img2)
    static cv::Mat ResizeToMultiple(const cv::Mat& img, int multiple, int& outH, int& outW);
    static void NormalizeToNCHW_01_Bgr(const cv::Mat& bgr01, std::vector<float>& out);
    static void CentralizeInPlace(cv::Mat& bgr01_a, cv::Mat& bgr01_b);

    // Pads to multiples of 8 (common for flow nets). Returns padded image.
    static cv::Mat PadToMultiple(const cv::Mat& img, int multiple, int& padBottom, int& padRight);

private:
    Ort::SessionOptions _sessOpts;
    Ort::Session* _session;
    Ort::MemoryInfo _memInfo;

    size_t _modelInputCount = 0;

    std::vector<std::string> _inputNamesStr;
    std::vector<std::string> _outputNamesStr;
    std::vector<const char*> _inputNames;
    std::vector<const char*> _outputNames;
};
