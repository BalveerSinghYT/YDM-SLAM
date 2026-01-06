#include "FlowNetOnnx.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static void PrintUsage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " --model /abs/path/model.onnx --img1 /path/1.png --img2 /path/2.png --out flow.png [--kind fastflownet|default]\n\n"
        << "Notes:\n"
        << "  - If --kind fastflownet, FlowNetOnnx will apply FastFlowNet demo-equivalent preprocessing/postprocessing.\n"
        << "  - You can also tweak via env vars: YDM_FLOW_NET_DIV_SIZE, YDM_FLOW_NET_DIV_FLOW.\n";
}

static bool GetArg(int argc, char** argv, const std::string& key, std::string& value)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (key == argv[i])
        {
            value = argv[i + 1];
            return true;
        }
    }
    return false;
}

static bool HasArg(int argc, char** argv, const std::string& key)
{
    for (int i = 1; i < argc; ++i)
    {
        if (key == argv[i]) return true;
    }
    return false;
}

static cv::Mat FlowToColorBgr(const cv::Mat& flow, float maxMag = -1.0f)
{
    CV_Assert(flow.type() == CV_32FC2);

    // Split flow into components
    std::vector<cv::Mat> ch(2);
    cv::split(flow, ch);

    cv::Mat mag, ang;
    cv::cartToPolar(ch[0], ch[1], mag, ang, true); // angle in degrees

    if (maxMag <= 0.0f)
    {
        double mx = 0.0;
        cv::minMaxLoc(mag, nullptr, &mx);
        maxMag = (float)mx;
        if (maxMag < 1e-6f) maxMag = 1.0f;
    }

    // HSV: H=angle, S=1, V=normalized magnitude
    cv::Mat hsv(flow.rows, flow.cols, CV_8UC3);
    for (int y = 0; y < flow.rows; ++y)
    {
        const float* magRow = mag.ptr<float>(y);
        const float* angRow = ang.ptr<float>(y);
        cv::Vec3b* hsvRow = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < flow.cols; ++x)
        {
            // OpenCV Hue range: [0,180)
            float hue = angRow[x] * 0.5f;
            if (hue < 0.0f) hue += 180.0f;
            if (hue >= 180.0f) hue = std::fmod(hue, 180.0f);

            float v = magRow[x] / maxMag;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;

            hsvRow[x][0] = (unsigned char)std::round(hue);
            hsvRow[x][1] = 255;
            hsvRow[x][2] = (unsigned char)std::round(v * 255.0f);
        }
    }

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

int main(int argc, char** argv)
{
    if (argc <= 1 || HasArg(argc, argv, "-h") || HasArg(argc, argv, "--help"))
    {
        PrintUsage(argv[0]);
        return 0;
    }

    std::string modelPath, img1Path, img2Path, outPath, kind;
    if (!GetArg(argc, argv, "--model", modelPath) || !GetArg(argc, argv, "--img1", img1Path) ||
        !GetArg(argc, argv, "--img2", img2Path) || !GetArg(argc, argv, "--out", outPath))
    {
        PrintUsage(argv[0]);
        return 2;
    }
    GetArg(argc, argv, "--kind", kind);

    if (!kind.empty())
    {
        // Make FlowNetOnnx pick the right pre/post-processing.
        setenv("YDM_FLOW_NET_KIND", kind.c_str(), 1);
    }

    cv::Mat im1 = cv::imread(img1Path, cv::IMREAD_UNCHANGED);
    cv::Mat im2 = cv::imread(img2Path, cv::IMREAD_UNCHANGED);
    if (im1.empty() || im2.empty())
    {
        std::cerr << "Failed to read images: " << img1Path << " or " << img2Path << std::endl;
        return 3;
    }

    // Ensure same size
    if (im1.size() != im2.size())
    {
        std::cerr << "Image sizes differ. img1=" << im1.cols << "x" << im1.rows << ", img2=" << im2.cols << "x" << im2.rows
                  << std::endl;
        return 4;
    }

    FlowNetOnnx net;
    if (!net.ReadModel(modelPath))
    {
        std::cerr << "Failed to load ONNX model: " << modelPath << std::endl;
        return 5;
    }

    cv::Mat flow;
    if (!net.InferFlow(im1, im2, flow) || flow.empty())
    {
        std::cerr << "InferFlow failed." << std::endl;
        return 6;
    }

    if (flow.type() != CV_32FC2 || flow.rows != im1.rows || flow.cols != im1.cols)
    {
        std::cerr << "Unexpected flow output. type=" << flow.type() << " size=" << flow.cols << "x" << flow.rows << std::endl;
        return 7;
    }

    cv::Mat vis = FlowToColorBgr(flow);
    if (!cv::imwrite(outPath, vis))
    {
        std::cerr << "Failed to write: " << outPath << std::endl;
        return 8;
    }

    std::cout << "Wrote: " << outPath << " (" << vis.cols << "x" << vis.rows << ")" << std::endl;
    return 0;
}
