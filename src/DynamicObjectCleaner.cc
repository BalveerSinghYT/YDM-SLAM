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

DynamicObjectCleaner::DynamicObjectCleaner()
{
    // Read residualIterations from environment variable
    if (const char* env_iters = std::getenv("YDM_RESIDUAL_ITERATIONS"))
    {
        int val = std::atoi(env_iters);
        if (val > 0 && val <= 20)
        {
            mParams.residualIterations = val;
            std::cout << "[DynamicObjectCleaner] Set residualIterations=" << val << " from YDM_RESIDUAL_ITERATIONS" << std::endl;
        }
    }

    // Background peeling (Option B) parameters
    if (const char* env_peel = std::getenv("YDM_BG_PEEL"))
    {
        if (std::string(env_peel) == "1")
        {
            mParams.enableBgPeeling = true;
        }
    }
    
    int iters = mParams.bgPeelIterations;
    if (const char* env_iters = std::getenv("YDM_BG_PEEL_ITERS"))
    {
        try {
            iters = std::stoi(std::string(env_iters));
            if (iters > 0 && iters <= 20)
                mParams.bgPeelIterations = iters;
        } catch (...) {}
    }
    
    int stride = mParams.bgPeelStride;
    if (const char* env_stride = std::getenv("YDM_BG_PEEL_STRIDE"))
    {
        try {
            stride = std::stoi(std::string(env_stride));
            if (stride > 0 && stride <= 32)
                mParams.bgPeelStride = stride;
        } catch (...) {}
    }
    
    if (const char* env_sigma = std::getenv("YDM_BG_PEEL_SIGMA_SCALE"))
    {
        try {
            float sigma = std::stof(std::string(env_sigma));
            if (std::isfinite(sigma) && sigma > 0.0f)
                mParams.bgPeelSigmaScale = sigma;
        } catch (...) {}
    }
    
    if (const char* env_min = std::getenv("YDM_BG_PEEL_MIN_PX"))
    {
        try {
            float minPx = std::stof(std::string(env_min));
            if (std::isfinite(minPx) && minPx >= 0.0f)
                mParams.bgPeelMinPx = minPx;
        } catch (...) {}
    }
    
    if (const char* env_huber = std::getenv("YDM_BG_PEEL_HUBER_DELTA"))
    {
        try {
            float delta = std::stof(std::string(env_huber));
            if (std::isfinite(delta) && delta > 0.0f)
                mParams.bgPeelHuberDelta = delta;
        } catch (...) {}
    }
    
    if (const char* env_ratio = std::getenv("YDM_BG_PEEL_MIN_INLIER_RATIO"))
    {
        try {
            float ratio = std::stof(std::string(env_ratio));
            if (std::isfinite(ratio) && ratio >= 0.0f && ratio <= 1.0f)
                mParams.bgPeelMinInlierRatio = ratio;
        } catch (...) {}
    }
    
    // Overlap threshold for instance-level dynamic detection
    if (const char* env_overlap = std::getenv("YDM_OVERLAP_DYNAMIC_TH"))
    {
        try {
            float overlap = std::stof(std::string(env_overlap));
            if (std::isfinite(overlap) && overlap >= 0.0f && overlap <= 1.0f)
            {
                mParams.overlapDynamicTh = overlap;
                std::cout << "[DynamicObjectCleaner] Set overlapDynamicTh=" << overlap 
                          << " from YDM_OVERLAP_DYNAMIC_TH" << std::endl;
            }
        } catch (...) {}
    }
    
    // Refinement dilation radius
    if (const char* env_dilate = std::getenv("YDM_REFINE_DILATE"))
    {
        try {
            int dilate = std::stoi(std::string(env_dilate));
            if (dilate >= 0 && dilate <= 20)
            {
                mParams.refineDilate = dilate;
                std::cout << "[DynamicObjectCleaner] Set refineDilate=" << dilate 
                          << " from YDM_REFINE_DILATE" << std::endl;
            }
        } catch (...) {}
    }
    
    if (mParams.enableBgPeeling)
    {
        std::cout << "[DynamicObjectCleaner] Background Peeling ENABLED: iters=" << mParams.bgPeelIterations
                  << " stride=" << mParams.bgPeelStride << " sigma=" << mParams.bgPeelSigmaScale
                  << " minPx=" << mParams.bgPeelMinPx << " huber=" << mParams.bgPeelHuberDelta << std::endl;
    }
}

bool DynamicObjectCleaner::RefineMotionWithFlowPeeling(
    const cv::Mat &prevGray,
    const cv::Mat &prevDepthMeters,
    const Sophus::SE3f &TcwPrev,
    const cv::Mat &currGray,
    const Eigen::Matrix3f &K,
    Sophus::SE3f &inoutTcwCurr,
    cv::Mat &outFlowPrevToCurr,
    cv::Mat &outResidual,
    cv::Mat &outValidMask,
    cv::Mat &outMotionMask,
    float &outThr,
    const cv::Mat &staticPriorMask)
{
    if (prevGray.empty() || currGray.empty() || prevDepthMeters.empty())
        return false;
    if (prevGray.size() != currGray.size() || prevGray.size() != prevDepthMeters.size())
        return false;
    if (prevGray.type() != CV_8UC1 || currGray.type() != CV_8UC1)
        return false;
    if (prevDepthMeters.type() != CV_32FC1)
        return false;
    if (!EnsureFlowNetLoaded())
        return false;

    const bool ydm_verbose = (std::getenv("YDM_VERBOSE") && std::string(std::getenv("YDM_VERBOSE")) == "1");
    const bool ydm_debug = (std::getenv("YDM_DEBUG_DYNAMIC") && std::string(std::getenv("YDM_DEBUG_DYNAMIC")) == "1");

    // Step 1: Compute Forward FlowNet (prev -> curr)
    outFlowPrevToCurr.release();
    if (!mFlowNet.InferFlow(prevGray, currGray, outFlowPrevToCurr) || outFlowPrevToCurr.empty())
        return false;
    if (outFlowPrevToCurr.type() != CV_32FC2)
        return false;

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    const int rows = prevDepthMeters.rows;
    const int cols = prevDepthMeters.cols;
    
    // ✅ FORWARD-BACKWARD CONSISTENCY CHECK (optional but recommended)
    const bool enable_consistency = (std::getenv("YDM_FLOW_CONSISTENCY") != nullptr && 
                                     std::string(std::getenv("YDM_FLOW_CONSISTENCY")) == "1");
    
    if (enable_consistency)
    {
        // Compute backward flow (curr -> prev)
        cv::Mat backwardFlow;
        if (mFlowNet.InferFlow(currGray, prevGray, backwardFlow) && !backwardFlow.empty())
        {
            // Check consistency: forward + backward ≈ 0
            cv::Mat consistencyMask;
            if (CheckFlowConsistency(outFlowPrevToCurr, backwardFlow, consistencyMask))
            {
                // Mask out inconsistent flow (likely occlusions or moving objects)
                #pragma omp parallel for if(rows * cols > 640 * 480)
                for (int y = 0; y < rows; ++y)
                {
                    cv::Vec2f *fptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
                    const uchar *cptr = consistencyMask.ptr<uchar>(y);
                    for (int x = 0; x < cols; ++x)
                    {
                        if (!cptr[x])  // Inconsistent → zero out
                            fptr[x] = cv::Vec2f(0.0f, 0.0f);
                    }
                }
                
                if (ydm_debug)
                {
                    int consistent = cv::countNonZero(consistencyMask);
                    float ratio = 100.0f * consistent / (rows * cols);
                    std::cout << "[YDM][FB_CHECK] Consistency: " << ratio << "% (" 
                              << consistent << "/" << (rows*cols) << " pixels)" << std::endl;
                }
            }
        }
    }
    
    // ✅ TEMPORAL SMOOTHING: Reduce flicker by exponential moving average
    // This stabilizes the optical flow across frames
    const bool enable_temporal_smooth = (std::getenv("YDM_FLOW_SMOOTH") == nullptr || 
                                         std::string(std::getenv("YDM_FLOW_SMOOTH")) != "0");
    
    if (enable_temporal_smooth && !mPrevFlow.empty() && 
        mPrevFlow.size() == outFlowPrevToCurr.size() && 
        mPrevFlow.type() == CV_32FC2)
    {
        const float alpha = mFlowSmoothAlpha;  // Weight for current frame (0.3 = 30% current, 70% previous)
        
        #pragma omp parallel for if(rows * cols > 640 * 480)
        for (int y = 0; y < rows; ++y)
        {
            cv::Vec2f *curr_ptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
            const cv::Vec2f *prev_ptr = mPrevFlow.ptr<cv::Vec2f>(y);
            
            for (int x = 0; x < cols; ++x)
            {
                // EMA: smoothed = alpha * current + (1-alpha) * previous
                curr_ptr[x][0] = alpha * curr_ptr[x][0] + (1.0f - alpha) * prev_ptr[x][0];
                curr_ptr[x][1] = alpha * curr_ptr[x][1] + (1.0f - alpha) * prev_ptr[x][1];
            }
        }
    }
    
    // Cache for next frame
    mPrevFlow = outFlowPrevToCurr.clone();
    mPrevGray = currGray.clone();
    
    // ✅ IMPROVEMENT: Mask out optical flow in depth-invalid regions EARLY
    // This prevents noise in depth-missing areas (e.g., sensor corners) from
    // polluting the MAD-based thresholding statistics
    #pragma omp parallel for if(rows * cols > 640 * 480)
    for (int y = 0; y < rows; ++y)
    {
        const float *dptr = prevDepthMeters.ptr<float>(y);
        cv::Vec2f *fptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
        for (int x = 0; x < cols; ++x)
        {
            const float z = dptr[x];
            // Zero out flow where depth is invalid (missing/NaN/negative)
            if (!(z > 0.0f && std::isfinite(z)))
                fptr[x] = cv::Vec2f(0.0f, 0.0f);
        }
    }
    
    const int stride = std::max(1, mParams.bgPeelStride);
    const int iters = std::max(1, mParams.bgPeelIterations);
    const float sigmaScale = mParams.bgPeelSigmaScale;
    const float minPx = mParams.bgPeelMinPx;
    const float minInlierRatio = mParams.bgPeelMinInlierRatio;

    // ✅ FIX 1: Correct SE(3) decomposition using Sophus log/exp (numerically stable)
    const Sophus::SE3f T_rel = inoutTcwCurr * TcwPrev.inverse();
    const Eigen::Matrix<float, 6, 1> xi_total = T_rel.log();
    if (!xi_total.array().isFinite().all())
        return false;

    // ✅ FIX 2: Auto pose direction detection using cosine similarity
    auto computeEgoFlowAndCosSim = [&](const Sophus::SE3f &T) -> std::pair<cv::Mat, float>
    {
        cv::Mat ego = cv::Mat::zeros(prevDepthMeters.size(), CV_32FC2);
        cv::Mat valid = cv::Mat::zeros(prevDepthMeters.size(), CV_8UC1);
        
        #pragma omp parallel for if(rows * cols > 640 * 480)
        for (int y = 0; y < rows; ++y)
        {
            const float *dptr = prevDepthMeters.ptr<float>(y);
            cv::Vec2f *eptr = ego.ptr<cv::Vec2f>(y);
            uchar *vptr = valid.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x)
            {
                const float z = dptr[x];
                if (!(z > 0.0f) || !std::isfinite(z))
                    continue;

                const float X = (static_cast<float>(x) - cx) * z / fx;
                const float Y = (static_cast<float>(y) - cy) * z / fy;
                const Eigen::Vector3f Xprev(X, Y, z);
                const Eigen::Vector3f Xcur = T * Xprev;

                const float zc = Xcur(2);
                if (!(zc > 1e-6f) || !std::isfinite(zc))
                    continue;

                const float u2 = fx * (Xcur(0) / zc) + cx;
                const float v2 = fy * (Xcur(1) / zc) + cy;
                
                // ✅ FIX 3: Strict in-bounds check
                if (!std::isfinite(u2) || !std::isfinite(v2) ||
                    u2 < 0.0f || u2 >= cols || v2 < 0.0f || v2 >= rows)
                    continue;

                eptr[x][0] = u2 - static_cast<float>(x);
                eptr[x][1] = v2 - static_cast<float>(y);
                vptr[x] = 255;
            }
        }
        
        // Compute mean cosine similarity with optical flow
        double cosSim = 0.0;
        int count = 0;
        for (int y = 0; y < rows; ++y)
        {
            const cv::Vec2f *fptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
            const cv::Vec2f *eptr = ego.ptr<cv::Vec2f>(y);
            const uchar *vptr = valid.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x)
            {
                if (!vptr[x])
                    continue;
                const float fx = fptr[x][0];
                const float fy = fptr[x][1];
                const float ex = eptr[x][0];
                const float ey = eptr[x][1];
                const float fmag = std::sqrt(fx * fx + fy * fy);
                const float emag = std::sqrt(ex * ex + ey * ey);
                if (fmag > 1e-6f && emag > 1e-6f)
                {
                    cosSim += (fx * ex + fy * ey) / (fmag * emag);
                    count++;
                }
            }
        }
        return {ego, count > 0 ? static_cast<float>(cosSim / count) : 0.0f};
    };

    auto [egoFlow1, cos1] = computeEgoFlowAndCosSim(T_rel);
    auto [egoFlow2, cos2] = computeEgoFlowAndCosSim(T_rel.inverse());
    
    Sophus::SE3f T_rel_corrected = T_rel;
    if (cos2 > cos1)
    {
        T_rel_corrected = T_rel.inverse();
        if (ydm_debug)
            std::cout << "[YDM][BG_PEEL] Auto-corrected pose direction (cosSim: " 
                      << cos1 << " → " << cos2 << ")" << std::endl;
    }
    else if (ydm_debug)
    {
        std::cout << "[YDM][BG_PEEL] Pose direction correct (cosSim: " << cos1 << ")" << std::endl;
    }

    // Compute initial valid mask with STRICT checks
    cv::Mat validMask = cv::Mat::zeros(prevDepthMeters.size(), CV_8UC1);
    #pragma omp parallel for if(rows * cols > 640 * 480)
    for (int y = 0; y < rows; ++y)
    {
        const float *dptr = prevDepthMeters.ptr<float>(y);
        uchar *vptr = validMask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x)
        {
            const float z = dptr[x];
            // ✅ FIX 4: Strict validity - must be positive AND finite
            if (z > 0.0f && std::isfinite(z))
                vptr[x] = 255;
        }
    }

    const int totalValid = cv::countNonZero(validMask);
    if (totalValid <= 0)
        return false;

    // Initialize background mask (all valid pixels are candidates)
    cv::Mat bgMask;
    if (!staticPriorMask.empty() && staticPriorMask.type() == CV_8UC1 && staticPriorMask.size() == validMask.size())
        cv::bitwise_and(validMask, staticPriorMask, bgMask);
    else
        bgMask = validMask.clone();

    // ✅ FIX 5: Initialize residual flow (will be progressively refined)
    cv::Mat flowRem = outFlowPrevToCurr.clone();
    cv::Mat residual = cv::Mat::zeros(currGray.size(), CV_32FC1);

    float thr = minPx;
    float lastMed = 0.0f;
    float lastMad = 0.0f;
    size_t lastSampleCount = 0;

    int performedIters = 0;
    
    // ✅ FIX 6: Iterative background peeling with correct SE(3) decomposition
    for (int iter = 0; iter < iters; ++iter)
    {
        performedIters = iter + 1;
        
        // Compute cumulative poses using Sophus log/exp (numerically stable!)
        // T_acc^k = exp(log(T_rel) * k/K)
        const float ratio_k = static_cast<float>(iter + 1) / static_cast<float>(iters);
        const float ratio_km1 = static_cast<float>(iter) / static_cast<float>(iters);
        
        const Sophus::SE3f T_acc_k = Sophus::SE3f::exp(xi_total * ratio_k);
        const Sophus::SE3f T_acc_km1 = Sophus::SE3f::exp(xi_total * ratio_km1);
        
        // Compute incremental ego-flow: delta_d^(k) = ego(T_acc^k) - ego(T_acc^{k-1})
        cv::Mat egoFlow_k = cv::Mat::zeros(prevDepthMeters.size(), CV_32FC2);
        cv::Mat egoFlow_km1 = cv::Mat::zeros(prevDepthMeters.size(), CV_32FC2);
        
        #pragma omp parallel for if(rows * cols > 640 * 480)
        for (int y = 0; y < rows; ++y)
        {
            const float *dptr = prevDepthMeters.ptr<float>(y);
            cv::Vec2f *ek_ptr = egoFlow_k.ptr<cv::Vec2f>(y);
            cv::Vec2f *ekm1_ptr = egoFlow_km1.ptr<cv::Vec2f>(y);
            const uchar *vptr = validMask.ptr<uchar>(y);
            
            for (int x = 0; x < cols; ++x)
            {
                if (!vptr[x])
                    continue;
                    
                const float z = dptr[x];
                const float X = (static_cast<float>(x) - cx) * z / fx;
                const float Y = (static_cast<float>(y) - cy) * z / fy;
                const Eigen::Vector3f Xprev(X, Y, z);
                
                // Transform with T_acc_k
                const Eigen::Vector3f Xcur_k = T_acc_k * Xprev;
                const float zc_k = Xcur_k(2);
                if (zc_k > 1e-6f && std::isfinite(zc_k))
                {
                    const float u2_k = fx * (Xcur_k(0) / zc_k) + cx;
                    const float v2_k = fy * (Xcur_k(1) / zc_k) + cy;
                    if (std::isfinite(u2_k) && std::isfinite(v2_k) &&
                        u2_k >= 0.0f && u2_k < cols && v2_k >= 0.0f && v2_k < rows)
                    {
                        ek_ptr[x][0] = u2_k - static_cast<float>(x);
                        ek_ptr[x][1] = v2_k - static_cast<float>(y);
                    }
                }
                
                // Transform with T_acc_km1
                const Eigen::Vector3f Xcur_km1 = T_acc_km1 * Xprev;
                const float zc_km1 = Xcur_km1(2);
                if (zc_km1 > 1e-6f && std::isfinite(zc_km1))
                {
                    const float u2_km1 = fx * (Xcur_km1(0) / zc_km1) + cx;
                    const float v2_km1 = fy * (Xcur_km1(1) / zc_km1) + cy;
                    if (std::isfinite(u2_km1) && std::isfinite(v2_km1) &&
                        u2_km1 >= 0.0f && u2_km1 < cols && v2_km1 >= 0.0f && v2_km1 < rows)
                    {
                        ekm1_ptr[x][0] = u2_km1 - static_cast<float>(x);
                        ekm1_ptr[x][1] = v2_km1 - static_cast<float>(y);
                    }
                }
            }
        }
        
        // ✅ FIX 7: TRUE PEELING - Only update background pixels!
        // This is the KEY difference from "global compensation"
        #pragma omp parallel for if(rows * cols > 640 * 480)
        for (int y = 0; y < rows; ++y)
        {
            const uchar *vptr = validMask.ptr<uchar>(y);
            const uchar *bptr = bgMask.ptr<uchar>(y);
            cv::Vec2f *fptr = flowRem.ptr<cv::Vec2f>(y);
            const cv::Vec2f *ek_ptr = egoFlow_k.ptr<cv::Vec2f>(y);
            const cv::Vec2f *ekm1_ptr = egoFlow_km1.ptr<cv::Vec2f>(y);
            float *rptr = residual.ptr<float>(y);
            
            for (int x = 0; x < cols; ++x)
            {
                if (!vptr[x])
                {
                    rptr[x] = 0.0f;
                    continue;
                }
                
                // Incremental delta
                const float delta_x = ek_ptr[x][0] - ekm1_ptr[x][0];
                const float delta_y = ek_ptr[x][1] - ekm1_ptr[x][1];
                
                // Only update if currently classified as background
                if (bptr[x])
                {
                    fptr[x][0] -= delta_x;
                    fptr[x][1] -= delta_y;
                }
                
                // Compute residual
                const float dx = fptr[x][0];
                const float dy = fptr[x][1];
                rptr[x] = std::sqrt(dx * dx + dy * dy);
            }
        }

        std::vector<float> samples;
        samples.reserve((rows / stride + 1) * (cols / stride + 1));
        for (int y = 0; y < rows; y += stride)
        {
            const float *rptr = residual.ptr<float>(y);
            const uchar *vptr = validMask.ptr<uchar>(y);
            const uchar *bptr = bgMask.ptr<uchar>(y);
            for (int x = 0; x < cols; x += stride)
            {
                if (vptr[x] && bptr[x])
                {
                    const float r = rptr[x];
                    if (std::isfinite(r))
                        samples.push_back(r);
                }
            }
        }

        thr = minPx;
        if (!samples.empty())
        {
            const float med = Median(samples);
            const float mad = MedianAbsDev(samples, med);
            const float sigma = 1.4826f * mad;
            thr = std::max(minPx, med + sigmaScale * sigma);
            lastMed = med;
            lastMad = mad;
            lastSampleCount = samples.size();
        }

        int bgCount = 0;
        int diffCount = 0;
        #pragma omp parallel for reduction(+:bgCount,diffCount) if(rows * cols > 640 * 480)
        for (int y = 0; y < rows; ++y)
        {
            const float *rptr = residual.ptr<float>(y);
            const uchar *vptr = validMask.ptr<uchar>(y);
            uchar *bptr = bgMask.ptr<uchar>(y);
            for (int x = 0; x < cols; ++x)
            {
                if (!vptr[x])
                {
                    if (bptr[x])
                        diffCount++;
                    bptr[x] = 0;
                    continue;
                }
                if (!bptr[x])
                    continue;

                const float r = rptr[x];
                const bool keepBg = (std::isfinite(r) && r <= thr);
                if (!keepBg)
                {
                    bptr[x] = 0;
                    diffCount++;
                }
                else
                {
                    bgCount++;
                }
            }
        }

        const float bgRatio = static_cast<float>(bgCount) / static_cast<float>(totalValid);
        const int motionPixels = totalValid - bgCount;

        if (ydm_debug)
        {
            std::cout << "[YDM][BG_PEEL] iter=" << (iter + 1) << "/" << iters
                      << " thr=" << thr << "px bgInlierRatio=" << bgRatio
                      << " motionPixels=" << motionPixels << std::endl;
        }

        const float changeRatio = static_cast<float>(diffCount) / static_cast<float>(totalValid);
        const bool tooFewInliers = (minInlierRatio > 0.0f && bgRatio < minInlierRatio);
        const bool littleChange = (changeRatio < 0.005f);
        if (tooFewInliers || littleChange)
        {
            if (ydm_debug)
            {
                std::cout << "[YDM][BG_PEEL] early_stop"
                          << " inlierRatio=" << bgRatio
                          << " changeRatio=" << changeRatio
                          << " minInlierRatio=" << minInlierRatio
                          << std::endl;
            }
            break;
        }
    }

    outResidual = residual;
    outValidMask = validMask;
    outMotionMask = cv::Mat::zeros(currGray.size(), CV_8UC1);
    #pragma omp parallel for if(rows * cols > 640 * 480)
    for (int y = 0; y < rows; ++y)
    {
        const uchar *vptr = validMask.ptr<uchar>(y);
        const uchar *bptr = bgMask.ptr<uchar>(y);
        uchar *mptr = outMotionMask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x)
        {
            if (!vptr[x])
            {
                mptr[x] = 0;
                continue;
            }
            mptr[x] = bptr[x] ? 0 : 255;
        }
    }
    outThr = thr;

    if (ydm_debug)
    {
        const int motionPixels = cv::countNonZero(outMotionMask);
        if (motionPixels == 0)
        {
            float staticCoverage = 1.0f;
            if (!staticPriorMask.empty() && staticPriorMask.type() == CV_8UC1 && staticPriorMask.size() == validMask.size())
            {
                const int staticCount = cv::countNonZero(staticPriorMask);
                staticCoverage = static_cast<float>(staticCount) / static_cast<float>(totalValid);
                if (staticCoverage > 1.0f)
                    staticCoverage = 1.0f;
            }
            std::cout << "[YDM][BG_PEEL] motionPixels=0 med=" << lastMed
                      << " mad=" << lastMad << " minPx=" << minPx
                      << " sigmaScale=" << sigmaScale
                      << " samples=" << lastSampleCount
                      << " staticCoverage=" << staticCoverage
                      << std::endl;
        }
    }

    // Note: Pose is NOT updated, we keep ORB-SLAM's pose and only refine mask iteratively
    if (ydm_verbose || ydm_debug)
    {
        std::cout << "[YDM][BG_PEEL] Completed " << performedIters << " iterations (pose unchanged)" << std::endl;
    }

    return true;
}

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

    cv::Mat flowPrevToCurr, residual, valid, motionMask;
    float thr = mParams.residualMinPx;
    if (!ComputeFlowAndResidualRGBD(prevGray, prevDepthMeters, TcwPrev,
                                    currGray, TcwCurr, K,
                                    flowPrevToCurr, residual, valid, motionMask, thr))
        return false;

    return CleanRGBDFromPrecomputed(residual, valid, motionMask, thr,
                                    instanceIdMask, instanceClassNames,
                                    prevDynamicRigidMask, flowPrevToCurr,
                                    currKeypoints, outlierFlags,
                                    outRefinedDynamicMask, outDynamicRigidMask);
}

bool DynamicObjectCleaner::ComputeFlowAndResidualRGBD(const cv::Mat &prevGray,
                                                     const cv::Mat &prevDepthMeters,
                                                     const Sophus::SE3f &TcwPrev,
                                                     const cv::Mat &currGray,
                                                     const Sophus::SE3f &TcwCurr,
                                                     const Eigen::Matrix3f &K,
                                                     cv::Mat &outFlowPrevToCurr,
                                                     cv::Mat &outResidual,
                                                     cv::Mat &outValidMask,
                                                     cv::Mat &outMotionMask,
                                                     float &outThr)
{
    if (prevGray.empty() || currGray.empty() || prevDepthMeters.empty())
        return false;
    if (prevGray.size() != currGray.size() || prevGray.size() != prevDepthMeters.size())
        return false;
    if (prevGray.type() != CV_8UC1 || currGray.type() != CV_8UC1)
        return false;
    if (prevDepthMeters.type() != CV_32FC1)
        return false;

    if (!EnsureFlowNetLoaded())
        return false;

    outFlowPrevToCurr.release();
    if (!mFlowNet.InferFlow(prevGray, currGray, outFlowPrevToCurr) || outFlowPrevToCurr.empty())
        return false;

    // ✅ FORWARD-BACKWARD CONSISTENCY CHECK
    const bool enable_consistency = (std::getenv("YDM_FLOW_CONSISTENCY") != nullptr && 
                                     std::string(std::getenv("YDM_FLOW_CONSISTENCY")) == "1");
    
    if (enable_consistency)
    {
        cv::Mat backwardFlow;
        if (mFlowNet.InferFlow(currGray, prevGray, backwardFlow) && !backwardFlow.empty())
        {
            cv::Mat consistencyMask;
            if (CheckFlowConsistency(outFlowPrevToCurr, backwardFlow, consistencyMask))
            {
                const int rows = outFlowPrevToCurr.rows;
                const int cols = outFlowPrevToCurr.cols;
                
                #pragma omp parallel for if(rows * cols > 640 * 480)
                for (int y = 0; y < rows; ++y)
                {
                    cv::Vec2f *fptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
                    const uchar *cptr = consistencyMask.ptr<uchar>(y);
                    for (int x = 0; x < cols; ++x)
                    {
                        if (!cptr[x])
                            fptr[x] = cv::Vec2f(0.0f, 0.0f);
                    }
                }
            }
        }
    }

    // ✅ TEMPORAL SMOOTHING: Reduce flicker (same as RefineMotionWithFlowPeeling)
    const bool enable_temporal_smooth = (std::getenv("YDM_FLOW_SMOOTH") == nullptr || 
                                         std::string(std::getenv("YDM_FLOW_SMOOTH")) != "0");
    
    if (enable_temporal_smooth && !mPrevFlow.empty() && 
        mPrevFlow.size() == outFlowPrevToCurr.size() && 
        mPrevFlow.type() == CV_32FC2)
    {
        const float alpha = mFlowSmoothAlpha;
        const int rows = outFlowPrevToCurr.rows;
        const int cols = outFlowPrevToCurr.cols;
        
        #pragma omp parallel for if(rows * cols > 640 * 480)
        for (int y = 0; y < rows; ++y)
        {
            cv::Vec2f *curr_ptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
            const cv::Vec2f *prev_ptr = mPrevFlow.ptr<cv::Vec2f>(y);
            
            for (int x = 0; x < cols; ++x)
            {
                curr_ptr[x][0] = alpha * curr_ptr[x][0] + (1.0f - alpha) * prev_ptr[x][0];
                curr_ptr[x][1] = alpha * curr_ptr[x][1] + (1.0f - alpha) * prev_ptr[x][1];
            }
        }
    }
    
    // Cache for next frame
    mPrevFlow = outFlowPrevToCurr.clone();
    mPrevGray = currGray.clone();

    cv::Mat egoFlow;
    ComputeEgoFlow(prevDepthMeters, TcwPrev, TcwCurr, K, egoFlow, outValidMask);

    // Depth Edge Suppression + validMask erosion
    // - YDM_DEPTH_EDGE_TH: threshold on |dZ/dx|+|dZ/dy| (in meters/pixel). <=0 disables.
    // - YDM_VALID_ERODE: erosion radius (pixels). 0 disables.
    {
        float depth_edge_th = 0.0f;
        int valid_erode_r = 0;
        const bool ydm_debug_dynamic = (std::getenv("YDM_DEBUG_DYNAMIC") && std::string(std::getenv("YDM_DEBUG_DYNAMIC")) == "1");
        
        if (const char* env = std::getenv("YDM_DEPTH_EDGE_TH"))
        {
            try { depth_edge_th = std::stof(std::string(env)); } catch (...) {}
        }
        if (const char* env = std::getenv("YDM_VALID_ERODE"))
        {
            try { valid_erode_r = std::stoi(std::string(env)); } catch (...) {}
        }
        
        if (depth_edge_th > 0.0f && !prevDepthMeters.empty() && prevDepthMeters.type() == CV_32FC1)
        {
            const int before = cv::countNonZero(outValidMask);

            cv::Mat dx, dy;
            cv::Sobel(prevDepthMeters, dx, CV_32F, 1, 0, 3);
            cv::Sobel(prevDepthMeters, dy, CV_32F, 0, 1, 3);
            cv::Mat mag = cv::abs(dx) + cv::abs(dy);

            cv::Mat edgeMaskF;
            cv::threshold(mag, edgeMaskF, depth_edge_th, 255.0, cv::THRESH_BINARY);
            cv::Mat edgeMask;
            edgeMaskF.convertTo(edgeMask, CV_8U);

            // Remove edge pixels from residual computation.
            outValidMask.setTo(0, edgeMask);

            if (ydm_debug_dynamic)
            {
                const int after = cv::countNonZero(outValidMask);
                const int dropped = std::max(0, before - after);
                const float drop_pct = (before > 0) ? (100.0f * static_cast<float>(dropped) / static_cast<float>(before)) : 0.0f;
                std::cout << "[YDM][DEPTH] edge_th=" << depth_edge_th << " drop=" << drop_pct << "%" << std::endl;
            }
        }

        if (valid_erode_r > 0 && !outValidMask.empty())
        {
            const int k = 2 * valid_erode_r + 1;
            cv::erode(outValidMask, outValidMask,
                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
        }
    }

    outResidual = cv::Mat::zeros(currGray.size(), CV_32FC1);
#pragma omp parallel for if(currGray.rows * currGray.cols > 640 * 480)
    for (int y = 0; y < currGray.rows; ++y)
    {
        const uchar *vptr = outValidMask.ptr<uchar>(y);
        const cv::Vec2f *fptr = outFlowPrevToCurr.ptr<cv::Vec2f>(y);
        const cv::Vec2f *eptr = egoFlow.ptr<cv::Vec2f>(y);
        float *rptr = outResidual.ptr<float>(y);
        for (int x = 0; x < currGray.cols; ++x)
        {
            if (!vptr[x])
                continue;
            const cv::Vec2f df = fptr[x] - eptr[x];
            rptr[x] = std::sqrt(df[0] * df[0] + df[1] * df[1]);
        }
    }

    outThr = mParams.residualMinPx;
    ComputeMotionMaskIterative(outResidual, outValidMask, mParams.residualIterations,
                               mParams.residualSigmaScale, mParams.residualMinPx,
                               outMotionMask, outThr);
    
    
    return true;
}

bool DynamicObjectCleaner::CleanRGBDFromPrecomputed(const cv::Mat &residual,
                                                   const cv::Mat &validMask,
                                                   const cv::Mat &motionMask,
                                                   float residualThr,
                                                   const cv::Mat &instanceIdMask,
                                                   const std::vector<std::string> &instanceClassNames,
                                                   const cv::Mat &prevDynamicRigidMask,
                                                   const cv::Mat &flowPrevToCurr,
                                                   const std::vector<cv::KeyPoint> &currKeypoints,
                                                   std::vector<bool> &outlierFlags,
                                                   cv::Mat &outRefinedDynamicMask,
                                                   cv::Mat &outDynamicRigidMask)
{
    outlierFlags.assign(currKeypoints.size(), false);
    outRefinedDynamicMask = cv::Mat::zeros(residual.size(), CV_8UC1);
    outDynamicRigidMask = cv::Mat::zeros(residual.size(), CV_8UC1);

    if (residual.empty() || residual.type() != CV_32FC1)
        return false;
    if (validMask.empty() || validMask.type() != CV_8UC1)
        return false;
    if (motionMask.empty() || motionMask.type() != CV_8UC1)
        return false;
    if (flowPrevToCurr.empty() || flowPrevToCurr.type() != CV_32FC2)
        return false;

    // If no instance segmentation, we can still prune by residual threshold if configured.
    const bool hasInstances = (!instanceIdMask.empty() && instanceIdMask.type() == CV_32SC1);
    if (!hasInstances)
    {
        if (mParams.pruneByResidualOutsideInstances)
        {
            for (size_t i = 0; i < currKeypoints.size(); ++i)
            {
                const int x = cvRound(currKeypoints[i].pt.x);
                const int y = cvRound(currKeypoints[i].pt.y);
                if (x < 0 || y < 0 || x >= residual.cols || y >= residual.rows)
                    continue;
                if (!validMask.at<uchar>(y, x))
                    continue;
                const float r = residual.at<float>(y, x);
                if (std::isfinite(r) && r > residualThr)
                    outlierFlags[i] = true;
            }
        }
        return true;
    }

    // Semantic prefilter: non-rigid instances are always dynamic
    cv::Mat nonRigidMask = BuildNonRigidMaskFromInstances(instanceIdMask, instanceClassNames);

    // Optional propagation of previous dynamic rigid mask via flow (rigid only)
    cv::Mat propagatedRigid = cv::Mat::zeros(residual.size(), CV_8UC1);
    if (mParams.enableMaskPropagation && !prevDynamicRigidMask.empty())
    {
        propagatedRigid = PropagateMaskWithFlow(prevDynamicRigidMask, flowPrevToCurr, residual.size());
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
        const int rows = residual.rows;
        const int cols = residual.cols;
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
            cv::Mat dynamicRigidMask = cv::Mat::zeros(residual.size(), CV_8UC1);
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
            if (validMask.at<uchar>(y, x))
            {
                const float r = residual.at<float>(y, x);
                if (std::isfinite(r) && r > residualThr)
                    out = true;
            }
        }

        outlierFlags[i] = out;
    }

    return true;
}

void DynamicObjectCleaner::WarpProbability(const cv::Mat &prevProb,
                                           const cv::Mat &flowPrevToCurr,
                                           cv::Mat &outWarpedProb)
{
    if (prevProb.empty() || flowPrevToCurr.empty())
    {
        outWarpedProb = cv::Mat::zeros(flowPrevToCurr.size(), CV_32FC1);
        return;
    }

    outWarpedProb = cv::Mat::zeros(prevProb.size(), CV_32FC1);
    const int rows = prevProb.rows;
    const int cols = prevProb.cols;
    
    // Simple Forward Splatting
    // For proper probabilistic warping we might want bilinear splatting, 
    // but for mask smoothing NN splatting + max is robust enough.
    
    #pragma omp parallel for
    for(int y=0; y<rows; y++)
    {
        const float* pPtr = prevProb.ptr<float>(y);
        const cv::Vec2f* fPtr = flowPrevToCurr.ptr<cv::Vec2f>(y);
        
        for(int x=0; x<cols; x++)
        {
            float prob = pPtr[x];
            // Skip low probability pixels to save time/atomic collisions
            if(prob < 0.01f) continue;
            
            float dx = fPtr[x][0];
            float dy = fPtr[x][1];
            
            int nx = cvRound(x + dx);
            int ny = cvRound(y + dy);
            
            if(nx >= 0 && nx < cols && ny >= 0 && ny < rows)
            {
                // We use a simple check-and-set. Not thread-safe for exact value,
                // but good enough for visual mask smoothing.
                float currVal = outWarpedProb.at<float>(ny, nx);
                if (prob > currVal)
                    outWarpedProb.at<float>(ny, nx) = prob;
            }
        }
    }
}

bool DynamicObjectCleaner::CheckFlowConsistency(const cv::Mat &forwardFlow,
                                                  const cv::Mat &backwardFlow,
                                                  cv::Mat &outConsistencyMask)
{
    if (forwardFlow.empty() || backwardFlow.empty()) return false;
    
    outConsistencyMask = cv::Mat::zeros(forwardFlow.size(), CV_8UC1);
    const float defaultTh = 1.0f;  // Default flow consistency threshold (pixels)
    const float thSq = defaultTh * defaultTh;
    const int rows = forwardFlow.rows;
    const int cols = forwardFlow.cols;

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y)
    {
        const cv::Vec2f *fwdPtr = forwardFlow.ptr<cv::Vec2f>(y);
        uchar *maskPtr = outConsistencyMask.ptr<uchar>(y);
        
        for (int x = 0; x < cols; ++x)
        {
            cv::Vec2f u = fwdPtr[x];
            
            // Map to next frame
            float nextX = x + u[0];
            float nextY = y + u[1];
            
            int ix = cvRound(nextX);
            int iy = cvRound(nextY);
            
            if (ix >= 0 && ix < cols && iy >= 0 && iy < rows)
            {
                // Sample backward flow at mapped position
                cv::Vec2f v = backwardFlow.at<cv::Vec2f>(iy, ix);
                
                // Consistency check: u + v should be close to 0
                // Magnitude of sum vector
                cv::Vec2f diff = u + v;
                float errSq = diff.dot(diff);
                
                if (errSq <= thSq)
                {
                    maskPtr[x] = 255;
                }
            }
        }
    }
    return true;
}
} // namespace ORB_SLAM3
