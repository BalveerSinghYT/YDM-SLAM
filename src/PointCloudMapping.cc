/*
 * Point Cloud Mapping for ORB-SLAM3
 * Based on ORB_SLAM3_PointCloud implementation
 */

#include "PointCloudMapping.h"
#include "KeyFrame.h"
#include "Converter.h"
#include "Map.h"

#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <sys/stat.h>

namespace ORB_SLAM3
{

PointCloudMapping::PointCloudMapping(double resolution_, double meank_, double thresh_, double depthMapFactor_)
    : mabIsUpdating(false), mDepthMapFactor(depthMapFactor_)
{
    this->resolution = resolution_;
    this->meank = meank_;
    this->thresh = thresh_;
    
    std::cout << "点云建图参数: resolution=" << resolution 
              << " meank=" << meank 
              << " thresh=" << thresh 
              << " depthMapFactor=" << mDepthMapFactor << std::endl;
    
    // 初始化统计滤波器
    statistical_filter = new pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA>(true);
    statistical_filter->setMeanK(meank);
    statistical_filter->setStddevMulThresh(thresh);
    
    // 初始化体素滤波器
    voxel = new pcl::VoxelGrid<pcl::PointXYZRGBA>();
    voxel->setLeafSize(resolution, resolution, resolution);
    
    // 初始化全局点云地图
    globalMap = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);

    // 启动可视化线程
    viewerThread = std::make_shared<std::thread>(std::bind(&PointCloudMapping::viewer, this));
}

void PointCloudMapping::shutdown()
{
    bool doJoin = false;
    {
        std::unique_lock<std::mutex> lck(shutDownMutex);
        if(!shutDownFlag)
        {
            shutDownFlag = true;
            keyFrameUpdated.notify_one();
        }
        doJoin = (viewerThread && viewerThread->joinable());
    }

    if(doJoin)
        viewerThread->join();

    // Saving is safe to call multiple times due to empty-cloud guard.
    save();

    if(!(std::getenv("YDM_PCD_GLOBAL") && std::string(std::getenv("YDM_PCD_GLOBAL")) == "1"))
        std::cout << "PointCloudMapping: 已输出关键帧点云到 ./pcd/kf_*.pcd（默认模式）" << std::endl;
}

void PointCloudMapping::Clear()
{
    std::cout << "清除稠密地图" << std::endl;
    std::unique_lock<std::mutex> lck(mMutexGlobalMap);
    globalMap.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, int idk, std::vector<KeyFrame *> vpKFs)
{
    std::cout << "接收关键帧，第 " << kf->mnId << " 个" << std::endl;
    
    if (color.empty())
        return;
    
    std::unique_lock<std::mutex> lck(keyframeMutex);
    keyframes.push_back(kf);
    currentvpKFs = vpKFs;
    
    keyFrameUpdated.notify_one();
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf)
{
    // 检查关键帧是否包含RGB图像
    if (kf->imLeftRgb.empty())
        return;
    
    std::unique_lock<std::mutex> lck(keyframeMutex);
    mlNewKeyFrames.emplace_back(kf);
    
    // 限制队列大小，避免内存过大
    if(mlNewKeyFrames.size() > 35)
        mlNewKeyFrames.pop_front();
}

void PointCloudMapping::generatePointCloud(KeyFrame *kf)
{
    // 检查图像数据有效性
    if (kf->imLeftRgb.empty() || kf->imDepth.empty())
        return;

    const int rgbChannels = kf->imLeftRgb.channels();
    if (!(rgbChannels == 1 || rgbChannels == 3 || rgbChannels == 4))
        return;
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    
    // DEBUG: Check mask propagation
    // NOTE: Printing and especially cv::imwrite() are very expensive; keep them off by default.
    const bool ydm_pcd_debug = (std::getenv("YDM_PCD_DEBUG") && std::string(std::getenv("YDM_PCD_DEBUG")) == "1");
    static int debug_print_count = 0;
    if (ydm_pcd_debug && debug_print_count < 15 && kf->mnId % 5 == 0)
    {
        std::cout << "Debug PointCloud: Frame " << kf->mnId << " has " << kf->mvDynamicMask.size() << " masks." << std::endl;
        std::cout << "Image Depth Size: " << kf->imDepth.size() << std::endl;
        for (size_t k = 0; k < kf->mvDynamicMask.size(); k++)
        {
            std::cout << "Mask " << k << " Size: " << kf->mvDynamicMask[k].size() << std::endl;
            std::string fname = "mask/mask_" + std::to_string(kf->mnId) + "_" + std::to_string(k) + ".png";
            cv::imwrite(fname, kf->mvDynamicMask[k]);
        }
        debug_print_count++;
    }

    // 从深度图生成点云，采样间隔为2像素
    for (int m = 0; m < kf->imDepth.rows; m += 2)
    {
            
        for (int n = 0; n < kf->imDepth.cols; n += 2)
        {
            // Dynamic object filtering
            bool isDynamic = false;

            for (const cv::Mat& mask : kf->mvDynamicMask) {
                if (mask.empty()) continue; 
                // Ensure mask size matches? 
                // if(mask.rows != kf->imDepth.rows) { resize... } -> YOLO mask should be resized. Flow mask is mImGray size.
                
                // Safe check bounds
                if (m < mask.rows && n < mask.cols) {
                    if(mask.at<uchar>(m, n) == 255) {
                        isDynamic = true;
                        break;
                    }
                }
            }
            if (isDynamic) continue;
            float d = 0;
            if (kf->imDepth.type() == CV_32F)
            {
                d = kf->imDepth.ptr<float>(m)[n];
            }
            else if (kf->imDepth.type() == CV_16U)
            {
                d = (float)kf->imDepth.ptr<unsigned short>(m)[n] / mDepthMapFactor;
            }
            else
            {
                continue;
            }
            
            // 过滤无效深度值
            if (d < 0.05 || d > 6)
                continue;

            // Simple edge filtering (depth jump)
            if (n > 4) {
                float d_left = 0;
                if (kf->imDepth.type() == CV_32F) d_left = kf->imDepth.ptr<float>(m)[n-4];
                else d_left = (float)kf->imDepth.ptr<unsigned short>(m)[n-4] / mDepthMapFactor;
                if (std::abs(d - d_left) > 0.1 * d) continue; 
            }
            
            pcl::PointXYZRGBA p;
            
            // 反投影到3D空间
            p.z = d;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy; // Flip Y-axis

            // 获取颜色信息
            if (rgbChannels == 3)
            {
                const uchar* row = kf->imLeftRgb.ptr<uchar>(m);
                p.b = row[n * 3 + 0];
                p.g = row[n * 3 + 1];
                p.r = row[n * 3 + 2];
            }
            else if (rgbChannels == 4)
            {
                const uchar* row = kf->imLeftRgb.ptr<uchar>(m);
                p.b = row[n * 4 + 0];
                p.g = row[n * 4 + 1];
                p.r = row[n * 4 + 2];
            }
            else
            {
                const uchar v = kf->imLeftRgb.ptr<uchar>(m)[n];
                p.r = v;
                p.g = v;
                p.b = v;
            }

            pPointCloud->points.push_back(p);
        }
    }
    
    pPointCloud->height = 1;
    pPointCloud->width = pPointCloud->points.size();
    pPointCloud->is_dense = true;
    
    // 保存到关键帧
    kf->mptrPointCloud = pPointCloud;
}

void PointCloudMapping::viewer()
{
    // pcl::visualization::CloudViewer viewer("PointCloud"); // Optional: Enable if PCL window is desired
    const bool enableGlobalMerge = (std::getenv("YDM_PCD_GLOBAL") && std::string(std::getenv("YDM_PCD_GLOBAL")) == "1");
    // Default: dump one PCD every N keyframes (to reduce IO)
    const int dumpStride = 5;
    bool ensuredDir = false;
    while (1)
    {
        {
            std::unique_lock<std::mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
                break;
        }
        
        // 如果正在停止或更新，跳过本次循环
        if (bStop || mabIsUpdating)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        int N;
        std::list<KeyFrame *> lNewKeyFrames;
        {
            std::unique_lock<std::mutex> lck(keyframeMutex);
            N = mlNewKeyFrames.size();
            lNewKeyFrames = mlNewKeyFrames;
            
            if(N == 0)
            {
                // 如果没有新关键帧，释放锁并等待或sleep
                lck.unlock(); 
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            else
            {
                mlNewKeyFrames.clear();
            }
        }
        
        // 处理新的关键帧
        for (auto pKF : lNewKeyFrames)
        {
            if (pKF->isBad())
                continue;
            
            // 生成点云 (Note: Uses CheckDynamicMask and mask+=2 internally)
            generatePointCloud(pKF);

            if(!pKF->mptrPointCloud || pKF->mptrPointCloud->empty())
                continue;
            
            // 按照 SiyuanYue 逻辑，直接变换并叠加到全局地图
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p(new pcl::PointCloud<pcl::PointXYZRGBA>);
            
            // Use the canonical float overload to avoid alignment/ABI pitfalls
            Eigen::Matrix4f Twc_f = pKF->GetPoseInverse().matrix();
            pcl::transformPointCloud(*(pKF->mptrPointCloud), *p, Twc_f);

            // Default mode: write per-keyframe PCDs (much safer than merging into a huge aligned vector)
            if(!enableGlobalMerge)
            {
                if (dumpStride > 1 && (static_cast<int>(pKF->mnId) % dumpStride) != 0)
                    continue;

                if(!ensuredDir)
                {
                    if(::mkdir("pcd", 0755) != 0 && errno != EEXIST)
                        std::cerr << "WARNING: Failed to create ./pcd directory (errno=" << errno << ")" << std::endl;
                    ensuredDir = true;
                }

                std::ostringstream oss;
                oss << "pcd/kf_" << std::setw(6) << std::setfill('0') << pKF->mnId << ".pcd";
                try
                {
                    pcl::io::savePCDFileBinaryCompressed(oss.str(), *p);
                }
                catch(const pcl::IOException &e)
                {
                    std::cerr << "WARNING: Failed to save " << oss.str() << ": " << e.what() << std::endl;
                }
                continue;
            }

            // Merge both operations under a single lock to avoid race conditions
            {
                std::unique_lock<std::mutex> lck(mMutexGlobalMap);
                
                // Add new points to global map
                *globalMap += *p;
                
                // Voxel Grid Filter on ENTIRE Global Map
                // IMPORTANT: Must use a temporary cloud, cannot filter in-place
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
                voxel->setInputCloud(globalMap);
                voxel->filter(*tmp);
                globalMap->swap(*tmp);
            }
        }
        
        // viewer.showCloud(globalMap); // Optional visualization
        
        // 保存点云到文件（已禁用以提升性能）
        // save();
        
        // std::cout << "点云地图已更新，当前点数: " << globalMap->size() << std::endl;
    }
}

void PointCloudMapping::save()
{
    // Always write result.pcd to current working directory.
    // If the point cloud is empty, write a minimal valid PCD file so downstream scripts don't fail.
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudSnapshot;
    {
        std::unique_lock<std::mutex> lck(mMutexGlobalMap);
        cloudSnapshot = globalMap;
    }

    if(!cloudSnapshot || cloudSnapshot->empty())
    {
        std::ofstream out("result.pcd", std::ios::out | std::ios::trunc);
        if(!out.is_open())
        {
            std::cerr << "WARNING: Failed to create result.pcd (cannot open for writing)" << std::endl;
            return;
        }
        out << "# .PCD v0.7 - Point Cloud Data file format\n";
        out << "VERSION 0.7\n";
        out << "FIELDS x y z rgb\n";
        out << "SIZE 4 4 4 4\n";
        out << "TYPE F F F F\n";
        out << "COUNT 1 1 1 1\n";
        out << "WIDTH 0\n";
        out << "HEIGHT 1\n";
        out << "VIEWPOINT 0 0 0 1 0 0 0\n";
        out << "POINTS 0\n";
        out << "DATA ascii\n";
        return;
    }

    try
    {
        pcl::io::savePCDFile("result.pcd", *cloudSnapshot);
    }
    catch(const pcl::IOException &e)
    {
        std::cerr << "WARNING: Failed to save result.pcd: " << e.what() << std::endl;
    }
}

void PointCloudMapping::updatecloud(Map &curMap)
{
    std::unique_lock<std::mutex> lck(updateMutex);

    mabIsUpdating = true;
    currentvpKFs = curMap.GetAllKeyFrames();
    
    std::cout << "开始点云更新" << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmpGlobalMap(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr curPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmpGlobalMapFilter(new pcl::PointCloud<pcl::PointXYZRGBA>());
    
    // 遍历所有关键帧，重新生成全局点云
    for (size_t i = 0; i < currentvpKFs.size(); i++)
    {
        if (!mabIsUpdating)
        {
            std::cout << "中断点云更新" << std::endl;
            return;
        }
        
        if (!currentvpKFs[i]->isBad() && currentvpKFs[i]->mptrPointCloud)
        {
            // 转换到世界坐标系
            Eigen::Matrix4f Twc_f = currentvpKFs[i]->GetPoseInverse().matrix();
            Eigen::Matrix4d Twc_d = Twc_f.cast<double>();
            pcl::transformPointCloud(
                *(currentvpKFs[i]->mptrPointCloud), *curPointCloud,
                Twc_d);
            
            *tmpGlobalMap += *curPointCloud;

            // 定期进行体素滤波，避免点云过大
            voxel->setInputCloud(tmpGlobalMap);
            voxel->filter(*tmpGlobalMapFilter);
            tmpGlobalMap->swap(*tmpGlobalMapFilter);
        }
    }
    
    std::cout << "点云更新完成" << std::endl;
    
    // 更新全局地图
    {
        std::unique_lock<std::mutex> lck(mMutexGlobalMap);
        globalMap = tmpGlobalMap;
    }
    
    mabIsUpdating = false;
}

} // namespace ORB_SLAM3
