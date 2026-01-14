#ifndef DYNAMIC_OBJECT_CLEANER_H
#define DYNAMIC_OBJECT_CLEANER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>

#include <Eigen/Core>

#include "FlowNetOnnx.h"
#include "sophus/se3.hpp"

namespace ORB_SLAM3
{

class DynamicObjectCleaner
{
public:
    struct Params
    {
        // Residual threshold: thr = max(minPx, median + sigmaScale * 1.4826 * MAD)
        float residualSigmaScale = 2.5f;
        float residualMinPx = 1.0f;

        // Iterative robust refinement steps (DIO-SLAM style)
        int residualIterations = 7;

        // Instance decision based on overlap ratio (DIO-SLAM prompt):
        // Let N=|Inst|, M=|Motion|, L=|Inst ∩ Motion|; dynamic if L/min(N,M) > overlapDynamicTh
        float overlapDynamicTh = 0.70f;

        // Morphology for refined mask
        int refineDilate = 3; // radius in pixels (approx)

        // Optional: propagate previous dynamic rigid mask using dense flow
        bool enableMaskPropagation = true;
        int propagationDilate = 1;

        // Also prune keypoints by residual even if outside refined instance masks
        bool pruneByResidualOutsideInstances = false;

        // Sampling stride for robust stats (outside semantic region)
        int statsStride = 4;

        // Background peeling parameters (Option B)
        bool enableBgPeeling = false;        // Enable iterative BG peeling with pose refinement
        int bgPeelIterations = 7;            // Number of peeling iterations
        int bgPeelStride = 4;                // Sampling stride for GN (speed/accuracy tradeoff)
        float bgPeelSigmaScale = 2.5f;       // Threshold scale for BG candidate selection
        float bgPeelMinPx = 1.0f;            // Minimum threshold for BG candidate
        float bgPeelHuberDelta = 5.0f;       // Huber loss delta (pixels) for robust GN
        float bgPeelMinInlierRatio = 0.3f;   // Minimum BG inlier ratio to accept pose update
    };

public:
    DynamicObjectCleaner();

    bool EnsureFlowNetLoaded();

    // Iterative background peeling over dense flow (pose unchanged):
    // - Computes ego-flow increments and progressively peels them from dense flow.
    // - outResidual/outMotionMask: final residual and motion evidence after K iterations.
    // - staticPriorMask (optional): CV_8UC1, 255=eligible static candidates for stats.
    bool RefineMotionWithFlowPeeling(const cv::Mat &prevGray,
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
                                     const cv::Mat &staticPriorMask = cv::Mat());

    // Compute dense FlowNet flow + ego-motion compensated residual once, so callers can reuse.
    // - outFlowPrevToCurr: CV_32FC2
    // - outResidual: CV_32FC1 (pixel magnitude)
    // - outValidMask: CV_8UC1 (255 valid)
    // - outMotionMask: CV_8UC1 (255 moving), derived from robust thresholding on residual
    // - outThr: robust residual threshold used to build outMotionMask
    bool ComputeFlowAndResidualRGBD(const cv::Mat &prevGray,
                                   const cv::Mat &prevDepthMeters,
                                   const Sophus::SE3f &TcwPrev,
                                   const cv::Mat &currGray,
                                   const Sophus::SE3f &TcwCurr,
                                   const Eigen::Matrix3f &K,
                                   cv::Mat &outFlowPrevToCurr,
                                   cv::Mat &outResidual,
                                   cv::Mat &outValidMask,
                                   cv::Mat &outMotionMask,
                                   float &outThr);

    // Same as CleanRGBD, but reuses precomputed flow/residual to avoid running FlowNet twice.
    bool CleanRGBDFromPrecomputed(const cv::Mat &residual,
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
                                  cv::Mat &outDynamicRigidMask);

    // Compute refined dynamic mask and mark outlier keypoints.
    // - prev/curr gray: CV_8U (1 channel)
    // - prevDepthMeters: CV_32F depth in meters (same size as images)
    // - TcwPrev/TcwCurr: camera pose (world -> camera)
    // - instanceIdMask: CV_32SC1 (0=bg, 1..N=instance id)
    // - instanceClassNames: size N, index (id-1) -> class name
    // - prevDynamicRigidMask: optional CV_8U (0/255), previous-frame dynamic rigid mask for propagation
    bool CleanRGBD(const cv::Mat &prevGray,
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
                  cv::Mat &outDynamicRigidMask);

    void SetParams(const Params &p) { mParams = p; }
    const Params &GetParams() const { return mParams; }

    void WarpProbability(const cv::Mat &prevProb,
                         const cv::Mat &flowPrevToCurr,
                         cv::Mat &outWarpedProb);

    bool CheckFlowConsistency(const cv::Mat &forwardFlow,
                              const cv::Mat &backwardFlow,
                              cv::Mat &outConsistencyMask);

private:
    static float Median(std::vector<float> &v);
    static float MedianAbsDev(std::vector<float> &v, float med);

    static bool IsNonRigidClass(const std::string &cls);
    static bool IsRigidClass(const std::string &cls);

    static cv::Mat BuildNonRigidMaskFromInstances(const cv::Mat &instanceIdMask,
                                                  const std::vector<std::string> &instanceClassNames);

    static void ComputeMotionMaskIterative(const cv::Mat &residual,
                                           const cv::Mat &validMask,
                                           int iterations,
                                           float sigmaScale,
                                           float minPx,
                                           cv::Mat &outMotionMask,
                                           float &outThr);

    static cv::Mat PropagateMaskWithFlow(const cv::Mat &prevMask,
                                         const cv::Mat &flowPrevToCurr,
                                         const cv::Size &size);

    void ComputeEgoFlow(const cv::Mat &prevDepthMeters,
                        const Sophus::SE3f &TcwPrev,
                        const Sophus::SE3f &TcwCurr,
                        const Eigen::Matrix3f &K,
                        cv::Mat &egoFlow,
                        cv::Mat &validMask) const;

private:
    Params mParams;

    FlowNetOnnx mFlowNet;
    bool mFlowNetLoaded = false;
};

} // namespace ORB_SLAM3

#endif
