#include "DynamicObjectCleaner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ORB_SLAM3
{

DynamicObjectCleaner::DynamicObjectCleaner() = default;

bool DynamicObjectCleaner::EnsureFlowNetLoaded()
{
    if (mFlowNetLoaded && mFlowNet.IsLoaded())
        return true;

    const char *flow_model_path = std::getenv("YDM_FLOW_NET_MODEL_PATH");
    std::string fallback;
    if (!flow_model_path || std::string(flow_model_path).empty())
    {
        fallback = "FastFlowNet/export/fastflownet.onnx";
        flow_model_path = fallback.c_str();
    }

    if (!std::getenv("YDM_FLOW_NET_KIND"))
        setenv("YDM_FLOW_NET_KIND", "FASTFLOWNET", 0);

    if (!mFlowNet.IsLoaded())
    {
        std::cout << "[YDM] DynamicObjectCleaner loading FlowNet model: " << flow_model_path << std::endl;
        mFlowNetLoaded = mFlowNet.ReadModel(std::string(flow_model_path));
        return mFlowNetLoaded;
    }

    mFlowNetLoaded = true;
    return true;
}

float DynamicObjectCleaner::Median(std::vector<float> &v)
{
    if (v.empty())
        return 0.0f;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    float med = v[n / 2];
    if ((n % 2) == 0)
    {
        std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
        med = 0.5f * (med + v[n / 2 - 1]);
    }
    return med;
}

float DynamicObjectCleaner::MedianAbsDev(std::vector<float> &v, float med)
{
    if (v.empty())
        return 0.0f;
    for (float &x : v)
        x = std::fabs(x - med);
    return Median(v);
}

bool DynamicObjectCleaner::IsNonRigidClass(const std::string &cls)
{
    // Default list aligned with current mvDynamicNames
    static const std::unordered_set<std::string> kNonRigid = {
        "person", "bird", "cat", "dog", "horse", "sheep", "crow", "bear"};
    return kNonRigid.find(cls) != kNonRigid.end();
}

bool DynamicObjectCleaner::IsRigidClass(const std::string &cls)
{
    static const std::unordered_set<std::string> kRigid = {
        "car", "motorbike", "bus", "train", "truck", "boat"};
    return kRigid.find(cls) != kRigid.end();
}

cv::Mat DynamicObjectCleaner::BuildNonRigidMaskFromInstances(
    const cv::Mat &instanceIdMask,
    const std::vector<std::string> &instanceClassNames)
{
    cv::Mat out = cv::Mat::zeros(instanceIdMask.size(), CV_8UC1);
    if (instanceIdMask.empty() || instanceIdMask.type() != CV_32SC1)
        return out;

    const int rows = instanceIdMask.rows;
    const int cols = instanceIdMask.cols;

#pragma omp parallel for if(rows * cols > 640 * 480)
    for (int y = 0; y < rows; ++y)
    {
        const int *iptr = instanceIdMask.ptr<int>(y);
        uchar *optr = out.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x)
        {
            const int id = iptr[x];
            if (id <= 0)
                continue;
            const int idx = id - 1;
            if (idx >= 0 && idx < static_cast<int>(instanceClassNames.size()) && IsNonRigidClass(instanceClassNames[idx]))
                optr[x] = 255;
        }
    }
    return out;
}

void DynamicObjectCleaner::ComputeMotionMaskIterative(
    const cv::Mat &residual,
    const cv::Mat &validMask,
    int iterations,
    float sigmaScale,
    float minPx,
    cv::Mat &outMotionMask,
    float &outThr)
{
    outMotionMask = cv::Mat::zeros(residual.size(), CV_8UC1);
    outThr = minPx;
    if (residual.empty() || residual.type() != CV_32FC1)
        return;
    if (validMask.empty())
        return;

    cv::Mat moving = cv::Mat::zeros(residual.size(), CV_8UC1);
    cv::Mat staticMask = cv::Mat::ones(residual.size(), CV_8UC1);

    const int iters = std::max(1, iterations);
    for (int it = 0; it < iters; ++it)
    {
        std::vector<float> samples;
        samples.reserve((residual.rows * residual.cols) / 8);

        // Collect residuals from pixels currently considered static
        for (int y = 0; y < residual.rows; ++y)
        {
            const float *rptr = residual.ptr<float>(y);
            const uchar *vptr = validMask.ptr<uchar>(y);
            const uchar *sptr = staticMask.ptr<uchar>(y);
            for (int x = 0; x < residual.cols; ++x)
            {
                if (!vptr[x] || !sptr[x])
                    continue;
                const float r = rptr[x];
                if (std::isfinite(r))
                    samples.push_back(r);
            }
        }

        float med = 0.0f, mad = 0.0f;
        if (!samples.empty())
        {
            med = Median(samples);
            mad = MedianAbsDev(samples, med);
        }
        const float sigma = 1.4826f * mad;
        const float thr = std::max(minPx, med + sigmaScale * sigma);
        outThr = thr;

        // Update moving mask
#pragma omp parallel for if(residual.rows * residual.cols > 640 * 480)
        for (int y = 0; y < residual.rows; ++y)
        {
            const float *rptr = residual.ptr<float>(y);
            const uchar *vptr = validMask.ptr<uchar>(y);
            uchar *mptr = moving.ptr<uchar>(y);
            uchar *sptr = staticMask.ptr<uchar>(y);
            for (int x = 0; x < residual.cols; ++x)
            {
                if (!vptr[x])
                {
                    mptr[x] = 0;
                    sptr[x] = 0;
                    continue;
                }
                const float r = rptr[x];
                const bool isMoving = (std::isfinite(r) && r > thr);
                mptr[x] = isMoving ? 255 : 0;
                sptr[x] = isMoving ? 0 : 255;
            }
        }
    }

    outMotionMask = moving;
}

cv::Mat DynamicObjectCleaner::PropagateMaskWithFlow(const cv::Mat &prevMask,
                                                   const cv::Mat &flowPrevToCurr,
                                                   const cv::Size &size)
{
    cv::Mat out = cv::Mat::zeros(size, CV_8UC1);
    if (prevMask.empty() || flowPrevToCurr.empty())
        return out;
    if (prevMask.size() != size || flowPrevToCurr.size() != size)
        return out;
    if (prevMask.type() != CV_8UC1 || flowPrevToCurr.type() != CV_32FC2)
        return out;

    const int rows = size.height;
    const int cols = size.width;

#pragma omp parallel for if(rows * cols > 640 * 480)
    for (int y = 0; y < rows; ++y)
    {
        const uchar *mptr = prevMask.ptr<uchar>(y);
        const cv::Vec2f *fptr = flowPrevToCurr.ptr<cv::Vec2f>(y);
        for (int x = 0; x < cols; ++x)
        {
            if (!mptr[x])
                continue;
            const float nx = static_cast<float>(x) + fptr[x][0];
            const float ny = static_cast<float>(y) + fptr[x][1];
            const int xi = static_cast<int>(std::lround(nx));
            const int yi = static_cast<int>(std::lround(ny));
            if (xi >= 0 && yi >= 0 && xi < cols && yi < rows)
                out.at<uchar>(yi, xi) = 255;
        }
    }

    return out;
}

void DynamicObjectCleaner::ComputeEgoFlow(const cv::Mat &prevDepthMeters,
                                         const Sophus::SE3f &TcwPrev,
                                         const Sophus::SE3f &TcwCurr,
                                         const Eigen::Matrix3f &K,
                                         cv::Mat &egoFlow,
                                         cv::Mat &validMask) const
{
    egoFlow = cv::Mat::zeros(prevDepthMeters.size(), CV_32FC2);
    validMask = cv::Mat::zeros(prevDepthMeters.size(), CV_8UC1);

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    const Sophus::SE3f TwcPrev = TcwPrev.inverse();

    const int rows = prevDepthMeters.rows;
    const int cols = prevDepthMeters.cols;

#pragma omp parallel for if(rows * cols > 640 * 480)
    for (int y = 0; y < rows; ++y)
    {
        const float *dptr = prevDepthMeters.ptr<float>(y);
        cv::Vec2f *fptr = egoFlow.ptr<cv::Vec2f>(y);
        uchar *vptr = validMask.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x)
        {
            const float z = dptr[x];
            if (!(z > 0.0f) || !std::isfinite(z))
                continue;

            const float X = (static_cast<float>(x) - cx) * z / fx;
            const float Y = (static_cast<float>(y) - cy) * z / fy;

            const Eigen::Vector3f Xprev(X, Y, z);
            const Eigen::Vector3f Xw = TwcPrev * Xprev;
            const Eigen::Vector3f Xcur = TcwCurr * Xw;

            const float zc = Xcur(2);
            if (!(zc > 1e-6f) || !std::isfinite(zc))
                continue;

            const float u2 = fx * (Xcur(0) / zc) + cx;
            const float v2 = fy * (Xcur(1) / zc) + cy;

            if (!std::isfinite(u2) || !std::isfinite(v2))
                continue;

            fptr[x][0] = u2 - static_cast<float>(x);
            fptr[x][1] = v2 - static_cast<float>(y);
            vptr[x] = 255;
        }
    }
}

bool DynamicObjectCleaner::CleanRGBD(const cv::Mat &prevGray,
                                    const cv::Mat &prevDepthMeters,
                                    const Sophus::SE3f &TcwPrev,
                                    const cv::Mat &currGray,
                                    const cv::Mat &currDepthMeters,
                                    const Sophus::SE3f &TcwCurr,
                                    const Eigen::Matrix3f &K,
                                    const cv::Mat &instanceIdMask,
                                    const std::vector<std::string> &instanceClassNames,
                                    const cv::Mat &prevDynamicRigidMask,
                                    const std::vector<cv::KeyPoint> &currKeypoints,
                                    std::vector<bool> &outlierFlags,
                                    cv::Mat &outRefinedDynamicMask,
                                    cv::Mat &outDynamicRigidMask)
{
    (void)currDepthMeters;

    if (prevGray.empty() || currGray.empty() || prevDepthMeters.empty())
        return false;
    if (prevGray.size() != currGray.size() || prevGray.size() != prevDepthMeters.size())
        return false;

    outlierFlags.assign(currKeypoints.size(), false);
    outRefinedDynamicMask = cv::Mat::zeros(currGray.size(), CV_8UC1);
    outDynamicRigidMask = cv::Mat::zeros(currGray.size(), CV_8UC1);

    if (instanceIdMask.empty() || instanceIdMask.type() != CV_32SC1)
        return true;

    if (!EnsureFlowNetLoaded())
        return false;

    cv::Mat flowPrevToCurr;
    if (!mFlowNet.InferFlow(prevGray, currGray, flowPrevToCurr) || flowPrevToCurr.empty())
        return false;

    cv::Mat egoFlow, valid;
    ComputeEgoFlow(prevDepthMeters, TcwPrev, TcwCurr, K, egoFlow, valid);

    // Compute residual image: ||f - e||
    cv::Mat residual = cv::Mat::zeros(currGray.size(), CV_32FC1);
#pragma omp parallel for if(currGray.rows * currGray.cols > 640 * 480)
    for (int y = 0; y < currGray.rows; ++y)
    {
        const uchar *vptr = valid.ptr<uchar>(y);
        const cv::Vec2f *fptr = flowPrevToCurr.ptr<cv::Vec2f>(y);
        const cv::Vec2f *eptr = egoFlow.ptr<cv::Vec2f>(y);
        float *rptr = residual.ptr<float>(y);
        for (int x = 0; x < currGray.cols; ++x)
        {
            if (!vptr[x])
                continue;
            const cv::Vec2f df = fptr[x] - eptr[x];
            rptr[x] = std::sqrt(df[0] * df[0] + df[1] * df[1]);
        }
    }

    cv::Mat motionMask;
    float thr = mParams.residualMinPx;
    ComputeMotionMaskIterative(residual, valid, mParams.residualIterations,
                               mParams.residualSigmaScale, mParams.residualMinPx,
                               motionMask, thr);

    // Semantic prefilter: non-rigid instances are always dynamic
    cv::Mat nonRigidMask = BuildNonRigidMaskFromInstances(instanceIdMask, instanceClassNames);

    // Optional propagation of previous dynamic rigid mask via flow (rigid only)
    cv::Mat propagatedRigid = cv::Mat::zeros(currGray.size(), CV_8UC1);
    if (mParams.enableMaskPropagation && !prevDynamicRigidMask.empty())
    {
        propagatedRigid = PropagateMaskWithFlow(prevDynamicRigidMask, flowPrevToCurr, currGray.size());
        if (mParams.propagationDilate > 0)
        {
            const int k = 2 * mParams.propagationDilate + 1;
            cv::dilate(propagatedRigid, propagatedRigid, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
        }
    }

    // Decide dynamic rigid instances by overlap with motionMask
    const int totalMotion = cv::countNonZero(motionMask);
    if (totalMotion > 0)
    {
        const int rows = currGray.rows;
        const int cols = currGray.cols;
        const int nInst = static_cast<int>(instanceClassNames.size());
        if (nInst > 0)
        {
            std::vector<int> instCount(nInst + 1, 0);
            std::vector<int> overlapCount(nInst + 1, 0);

            // Count per-instance pixels and overlap with motion mask
            for (int y = 0; y < rows; ++y)
            {
                const int *iptr = instanceIdMask.ptr<int>(y);
                const uchar *mptr = motionMask.ptr<uchar>(y);
                for (int x = 0; x < cols; ++x)
                {
                    const int id = iptr[x];
                    if (id <= 0 || id > nInst)
                        continue;
                    instCount[id]++;
                    if (mptr[x])
                        overlapCount[id]++;
                }
            }

            // Add dynamic rigid instances into mask
            cv::Mat dynamicRigidMask = cv::Mat::zeros(currGray.size(), CV_8UC1);
            for (int id = 1; id <= nInst; ++id)
            {
                const std::string &cls = instanceClassNames[id - 1];
                if (!IsRigidClass(cls))
                    continue;
                const int N = instCount[id];
                const int L = overlapCount[id];
                const int denom = std::min(N, totalMotion);
                if (denom <= 0)
                    continue;
                const float ratio = static_cast<float>(L) / static_cast<float>(denom);
                if (ratio <= mParams.overlapDynamicTh)
                    continue;

                // Paint this instance as dynamic
                for (int y = 0; y < rows; ++y)
                {
                    const int *iptr = instanceIdMask.ptr<int>(y);
                    uchar *dptr = dynamicRigidMask.ptr<uchar>(y);
                    for (int x = 0; x < cols; ++x)
                    {
                        if (iptr[x] == id)
                            dptr[x] = 255;
                    }
                }
            }

            // Rigid dynamic mask = detected rigid dynamic OR propagated rigid dynamic
            cv::bitwise_or(dynamicRigidMask, propagatedRigid, outDynamicRigidMask);
        }
    }

    // Refined mask for keypoint filtering = non-rigid OR rigid-dynamic
    cv::bitwise_or(nonRigidMask, outDynamicRigidMask, outRefinedDynamicMask);

    if (mParams.refineDilate > 0)
    {
        const int k = 2 * mParams.refineDilate + 1;
        cv::dilate(outRefinedDynamicMask, outRefinedDynamicMask,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    }

    // Mark outliers
    for (size_t i = 0; i < currKeypoints.size(); ++i)
    {
        const int x = cvRound(currKeypoints[i].pt.x);
        const int y = cvRound(currKeypoints[i].pt.y);
        if (x < 0 || y < 0 || x >= outRefinedDynamicMask.cols || y >= outRefinedDynamicMask.rows)
            continue;

        bool out = (outRefinedDynamicMask.at<uchar>(y, x) != 0);
        if (!out && mParams.pruneByResidualOutsideInstances)
        {
            if (valid.at<uchar>(y, x))
            {
                const float r = residual.at<float>(y, x);
                if (std::isfinite(r) && r > thr)
                    out = true;
            }
        }

        outlierFlags[i] = out;
    }

    return true;
}

} // namespace ORB_SLAM3
