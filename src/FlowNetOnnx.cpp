#include "FlowNetOnnx.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>

#include <opencv2/imgproc/imgproc.hpp>

// Global Ort::Env singleton (recommended by ONNXRuntime for thread safety)
static std::once_flag sOrtEnvOnce;
static Ort::Env* sOrtEnv = nullptr;

static Ort::Env& GetOrtEnv()
{
    std::call_once(sOrtEnvOnce, []() {
        sOrtEnv = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "FlowNet");
    });
    return *sOrtEnv;
}

FlowNetOnnx::FlowNetOnnx()
    : _sessOpts(),
      _session(nullptr),
      _memInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault))
{
    // Optimize for GPU inference - use single thread for better GPU utilization
    _sessOpts.SetIntraOpNumThreads(1);  // Single thread per op avoids CPU overhead
    _sessOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _sessOpts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);  // Sequential for small models
    
    // Enable CUDA GPU acceleration if available
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo for faster allocation
        cuda_options.gpu_mem_limit = SIZE_MAX;   // No limit for better performance
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;  // Best performance
        cuda_options.do_copy_in_default_stream = 1;
        _sessOpts.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[FlowNet] CUDA GPU acceleration enabled (Exhaustive search mode)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FlowNet] CUDA not available, using CPU: " << e.what() << std::endl;
    }
}

FlowNetOnnx::~FlowNetOnnx() noexcept
{
    try
    {
        if (_session)
        {
            delete _session;
            _session = nullptr;
        }
    }
    catch (...) {}
}

bool FlowNetOnnx::ReadModel(const std::string& modelPath, bool /*isCuda*/, int /*cudaID*/)
{
    try
    {
        if (_session)
        {
            delete _session;
            _session = nullptr;
        }

        _session = new Ort::Session(GetOrtEnv(), modelPath.c_str(), _sessOpts);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t inCount = _session->GetInputCount();
        const size_t outCount = _session->GetOutputCount();
        _modelInputCount = inCount;

        _inputNamesStr.clear();
        _outputNamesStr.clear();
        _inputNames.clear();
        _outputNames.clear();

        _inputNamesStr.reserve(inCount);
        for (size_t i = 0; i < inCount; ++i)
        {
#if defined(ORT_API_VERSION) && (ORT_API_VERSION >= 14)
            auto name = _session->GetInputNameAllocated(i, allocator);
            _inputNamesStr.emplace_back(name ? name.get() : "");
#else
            char* name = _session->GetInputName(i, allocator);
            _inputNamesStr.emplace_back(name ? name : "");
            allocator.Free(name);
#endif
        }
        _outputNamesStr.reserve(outCount);
        for (size_t i = 0; i < outCount; ++i)
        {
#if defined(ORT_API_VERSION) && (ORT_API_VERSION >= 14)
            auto name = _session->GetOutputNameAllocated(i, allocator);
            _outputNamesStr.emplace_back(name ? name.get() : "");
#else
            char* name = _session->GetOutputName(i, allocator);
            _outputNamesStr.emplace_back(name ? name : "");
            allocator.Free(name);
#endif
        }

        for (auto& s : _inputNamesStr) _inputNames.push_back(s.c_str());
        for (auto& s : _outputNamesStr) _outputNames.push_back(s.c_str());

        if (!((inCount == 1) || (inCount == 2)) || outCount < 1)
        {
            std::cerr << "FlowNetOnnx: Unexpected IO counts. inputs=" << inCount << " outputs=" << outCount << std::endl;
            return false;
        }

        return true;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "FlowNetOnnx: ReadModel failed: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FlowNetOnnx: ReadModel failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat FlowNetOnnx::To3CH(const cv::Mat& img)
{
    if (img.empty()) return cv::Mat();
    if (img.type() == CV_8UC3) return img;
    if (img.type() == CV_8UC1)
    {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    cv::Mat u8;
    img.convertTo(u8, CV_8U);
    if (u8.channels() == 1)
    {
        cv::Mat bgr;
        cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (u8.channels() == 3) return u8;

    cv::Mat bgr;
    cv::cvtColor(u8, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
}

cv::Mat FlowNetOnnx::PadToMultiple(const cv::Mat& img, int multiple, int& padBottom, int& padRight)
{
    padBottom = 0;
    padRight = 0;
    if (img.empty() || multiple <= 1) return img;

    const int H = img.rows;
    const int W = img.cols;
    const int newH = (int)std::ceil((double)H / (double)multiple) * multiple;
    const int newW = (int)std::ceil((double)W / (double)multiple) * multiple;

    padBottom = newH - H;
    padRight = newW - W;

    if (padBottom == 0 && padRight == 0) return img;

    cv::Mat out;
    cv::copyMakeBorder(img, out, 0, padBottom, 0, padRight, cv::BORDER_REPLICATE);
    return out;
}

void FlowNetOnnx::NormalizeToNCHW(const cv::Mat& bgr, std::vector<float>& out, int& H, int& W)
{
    H = bgr.rows;
    W = bgr.cols;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat f;
    rgb.convertTo(f, CV_32F, 1.0 / 255.0);
    // normalize to [-1, 1]
    f = f * 2.0f - 1.0f;

    std::vector<cv::Mat> ch(3);
    cv::split(f, ch);

    out.resize((size_t)3 * (size_t)H * (size_t)W);
    float* dst = out.data();

    // NCHW
    for (int c = 0; c < 3; ++c)
    {
        const float* src = (const float*)ch[c].data;
        std::copy(src, src + (size_t)H * (size_t)W, dst + (size_t)c * (size_t)H * (size_t)W);
    }
}

cv::Mat FlowNetOnnx::ResizeToMultiple(const cv::Mat& img, int multiple, int& outH, int& outW)
{
    outH = img.rows;
    outW = img.cols;
    if (img.empty() || multiple <= 1) return img;

    const int H = img.rows;
    const int W = img.cols;
    const int newH = (int)std::ceil((double)H / (double)multiple) * multiple;
    const int newW = (int)std::ceil((double)W / (double)multiple) * multiple;
    outH = newH;
    outW = newW;
    if (newH == H && newW == W) return img;

    cv::Mat out;
    cv::resize(img, out, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    return out;
}

void FlowNetOnnx::CentralizeInPlace(cv::Mat& bgr01_a, cv::Mat& bgr01_b)
{
    // Equivalent to demo.py centralize():
    // rgb_mean = mean(cat(img1,img2) along height), then subtract from both.
    // For same-size images, this is just average of per-image means.
    if (bgr01_a.empty() || bgr01_b.empty()) return;
    CV_Assert(bgr01_a.type() == CV_32FC3);
    CV_Assert(bgr01_b.type() == CV_32FC3);
    CV_Assert(bgr01_a.size() == bgr01_b.size());

    const cv::Scalar m1 = cv::mean(bgr01_a);
    const cv::Scalar m2 = cv::mean(bgr01_b);
    const cv::Scalar m = (m1 + m2) * 0.5;

    bgr01_a -= m;
    bgr01_b -= m;
}

void FlowNetOnnx::NormalizeToNCHW_01_Bgr(const cv::Mat& bgr01, std::vector<float>& out)
{
    CV_Assert(bgr01.type() == CV_32FC3);
    const int H = bgr01.rows;
    const int W = bgr01.cols;
    out.resize((size_t)3 * (size_t)H * (size_t)W);

    std::vector<cv::Mat> ch(3);
    cv::split(bgr01, ch);

    const size_t HW = (size_t)H * (size_t)W;
    for (int c = 0; c < 3; ++c)
    {
        const float* src = (const float*)ch[c].data;
        std::copy(src, src + HW, out.data() + (size_t)c * HW);
    }
}

bool FlowNetOnnx::InferFlow(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow)
{
    flow.release();
    if (!_session) return false;
    if (img1.empty() || img2.empty()) return false;
    if (img1.size() != img2.size()) return false;

    try
    {
        const char* env_kind = std::getenv("YDM_FLOW_NET_KIND");
        const bool use_fastflownet = (env_kind && (std::string(env_kind) == "FASTFLOWNET" || std::string(env_kind) == "fastflownet"));

        // FastFlowNet postproc params (match demo.py defaults)
        float div_flow = 20.0f;
        int div_size = 64;
        if (const char* env_div_flow = std::getenv("YDM_FLOW_NET_DIV_FLOW"))
        {
            try { div_flow = std::stof(std::string(env_div_flow)); } catch (...) { div_flow = 20.0f; }
            if (div_flow <= 0.0f) div_flow = 20.0f;
        }
        if (const char* env_div_size = std::getenv("YDM_FLOW_NET_DIV_SIZE"))
        {
            try { div_size = std::stoi(std::string(env_div_size)); } catch (...) { div_size = 64; }
            if (div_size < 8) div_size = 64;
        }

        cv::Mat bgr1 = To3CH(img1);
        cv::Mat bgr2 = To3CH(img2);

        // Orig size
        const int origH = bgr1.rows;
        const int origW = bgr1.cols;

        // Prepared size (may be resized/padded depending on mode)
        int H = 0, W = 0;
        int padB = 0, padR = 0;
        int resizedH = origH, resizedW = origW;

        std::vector<float> in1, in2;

        cv::Mat p1, p2;
        if (use_fastflownet)
        {
            // 1) convert to float in [0,1] (keep BGR order)
            cv::Mat f1, f2;
            bgr1.convertTo(f1, CV_32FC3, 1.0 / 255.0);
            bgr2.convertTo(f2, CV_32FC3, 1.0 / 255.0);

            // 2) centralize (subtract mean over both frames)
            CentralizeInPlace(f1, f2);

            // 3) PAD to multiples of div_size (default 64) - FIXED: use padding instead of resize
            //    This preserves pixel geometry and keeps optical flow sharp
            int padB1 = 0, padR1 = 0;
            int padB2 = 0, padR2 = 0;
            p1 = PadToMultiple(f1, div_size, padB1, padR1);
            p2 = PadToMultiple(f2, div_size, padB2, padR2);
            if (padB1 != padB2 || padR1 != padR2) return false;

            padB = padB1;
            padR = padR1;
            H = p1.rows;
            W = p1.cols;
            
            NormalizeToNCHW_01_Bgr(p1, in1);
            NormalizeToNCHW_01_Bgr(p2, in2);
        }
        else
        {
            // Default path for existing FlowNet-like models
            int padB1 = 0, padR1 = 0;
            int padB2 = 0, padR2 = 0;
            // pad both identically
            p1 = PadToMultiple(bgr1, 8, padB1, padR1);
            p2 = PadToMultiple(bgr2, 8, padB2, padR2);
            if (padB1 != padB2 || padR1 != padR2) return false;
            padB = padB1;
            padR = padR1;

            NormalizeToNCHW(p1, in1, H, W);
            NormalizeToNCHW(p2, in2, H, W);
        }

        std::vector<Ort::Value> inputs;
        inputs.reserve(2);
        size_t inputCount = _modelInputCount;
        if (inputCount == 0) inputCount = _inputNames.size();

        // NOTE: ORT tensors created with CreateTensor() do not copy input buffers.
        // Keep any backing storage alive until after _session->Run().
        std::vector<float> in6;

        if (inputCount == 1)
        {
            // FlowNetS-style: single input tensor [1,6,H,W] = concat(img1,img2)
            in6.resize((size_t)6 * (size_t)H * (size_t)W);
            const size_t HW = (size_t)H * (size_t)W;
            std::copy(in1.begin(), in1.end(), in6.begin() + 0 * HW);
            std::copy(in2.begin(), in2.end(), in6.begin() + 3 * HW);

            std::array<int64_t, 4> shape{1, 6, (int64_t)H, (int64_t)W};
            Ort::Value t = Ort::Value::CreateTensor<float>(_memInfo, in6.data(), in6.size(), shape.data(), shape.size());
            inputs.emplace_back(std::move(t));
        }
        else
        {
            // Two-input models: [1,3,H,W] + [1,3,H,W]
            std::array<int64_t, 4> shape{1, 3, (int64_t)H, (int64_t)W};
            Ort::Value t1 = Ort::Value::CreateTensor<float>(_memInfo, in1.data(), in1.size(), shape.data(), shape.size());
            Ort::Value t2 = Ort::Value::CreateTensor<float>(_memInfo, in2.data(), in2.size(), shape.data(), shape.size());
            inputs.emplace_back(std::move(t1));
            inputs.emplace_back(std::move(t2));
            inputCount = 2;
        }

        auto outputs = _session->Run(
            Ort::RunOptions{nullptr}, _inputNames.data(), inputs.data(), inputCount, _outputNames.data(), 1);
        if (outputs.empty()) return false;

        Ort::Value& out0 = outputs[0];
        auto ti = out0.GetTensorTypeAndShapeInfo();
        auto shp = ti.GetShape();
        const float* outData = out0.GetTensorData<float>();

        // Determine layout
        int outH = 0, outW = 0;
        bool isNCHW = false;
        bool isNHWC = false;

        if (shp.size() == 4)
        {
            if (shp[1] == 2)
            {
                isNCHW = true;
                outH = (int)shp[2];
                outW = (int)shp[3];
            }
            else if (shp[3] == 2)
            {
                isNHWC = true;
                outH = (int)shp[1];
                outW = (int)shp[2];
            }
        }

        if (!isNCHW && !isNHWC)
        {
            std::cerr << "FlowNetOnnx: Unsupported output shape." << std::endl;
            return false;
        }

        cv::Mat flowPadded(outH, outW, CV_32FC2);
        if (isNCHW)
        {
            // [1,2,H,W]
            const size_t HW = (size_t)outH * (size_t)outW;
            const float* fx = outData;
            const float* fy = outData + HW;
            for (int y = 0; y < outH; ++y)
            {
                cv::Vec2f* row = flowPadded.ptr<cv::Vec2f>(y);
                const size_t base = (size_t)y * (size_t)outW;
                for (int x = 0; x < outW; ++x)
                {
                    row[x][0] = fx[base + (size_t)x];
                    row[x][1] = fy[base + (size_t)x];
                }
            }
        }
        else
        {
            // [1,H,W,2]
            for (int y = 0; y < outH; ++y)
            {
                cv::Vec2f* row = flowPadded.ptr<cv::Vec2f>(y);
                for (int x = 0; x < outW; ++x)
                {
                    const size_t idx = ((size_t)y * (size_t)outW + (size_t)x) * 2;
                    row[x][0] = outData[idx + 0];
                    row[x][1] = outData[idx + 1];
                }
            }
        }

        if (use_fastflownet)
        {
            // FastFlowNet output is usually low-res (e.g., H/4 x W/4) and needs:
            // 1) upsample to padded input size
            // 2) multiply by div_flow
            // 3) CROP back to original size (not resize - preserves geometry)

            cv::Mat flow_up;
            cv::resize(flowPadded, flow_up, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
            flow_up *= div_flow;

            // Crop the padded region to get back to original size
            cv::Rect roi(0, 0, origW, origH);
            if (roi.width <= flow_up.cols && roi.height <= flow_up.rows)
                flow = flow_up(roi).clone();
            else
                flow = flow_up.clone();
        }
        else
        {
            // unpad (default path)
            cv::Rect roi(0, 0, img1.cols, img1.rows);
            if (roi.width <= flowPadded.cols && roi.height <= flowPadded.rows)
                flow = flowPadded(roi).clone();
            else
                flow = flowPadded.clone();
        }

        return true;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "FlowNetOnnx: InferFlow failed: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FlowNetOnnx: InferFlow failed: " << e.what() << std::endl;
        return false;
    }
}
