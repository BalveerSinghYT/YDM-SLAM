/*
 * Point Cloud Mapping for ORB-SLAM3
 * Based on ORB_SLAM3_PointCloud implementation
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <list>
#include <vector>

#include "System.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>

namespace ORB_SLAM3
{

class KeyFrame;
class Map;

/**
 * @brief 点云数据结构
 * 存储单个关键帧对应的点云及其位姿
 */
class PointCloude
{
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZRGB>;
    
    PointCloud::Ptr pcE;        // 点云数据
    Eigen::Isometry3d T;        // 位姿变换
    int pcID;                   // 点云ID
};

/**
 * @brief 点云建图类
 * 负责从RGB-D图像生成稠密点云地图，并提供可视化功能
 */
class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    /**
     * @brief 构造函数
     * @param resolution_ 体素滤波分辨率
     * @param meank_ 统计滤波的邻域点数
     * @param thresh_ 统计滤波的标准差倍数阈值
     */
    /**
     * @brief 构造函数
     * @param resolution_ 体素滤波分辨率
     * @param meank_ 统计滤波的邻域点数
     * @param thresh_ 统计滤波的标准差倍数阈值
     * @param depthMapFactor_ 深度图因子 (默认1000.0)
     */
    PointCloudMapping(double resolution_, double meank_, double thresh_, double depthMapFactor_ = 1000.0);    
    ~PointCloudMapping();

    /**
     * @brief 保存点云地图到文件
     */
    void save();

    /**
     * @brief 插入关键帧（带完整参数）
     * @param kf 关键帧指针
     * @param color RGB图像
     * @param depth 深度图
     * @param idk 关键帧ID
     * @param vpKFs 所有关键帧列表
     */
    void insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, int idk, std::vector<KeyFrame *> vpKFs);

    /**
     * @brief 插入关键帧（简化版本）
     * @param kf 关键帧指针（已包含RGB和深度图）
     */
    void insertKeyFrame(KeyFrame *kf);

    /**
     * @brief 关闭点云建图线程
     */
    void shutdown();

    /**
     * @brief 点云可视化线程主循环
     */
    void viewer();

    /**
     * @brief 回环后更新全局点云地图
     * @param curMap 当前地图
     */
    void updatecloud(Map &curMap);

    /**
     * @brief 清除点云地图
     */
    void Clear();

    // 公共成员变量
    int mnloopcount = 0;                        // 回环计数
    std::vector<KeyFrame *> currentvpKFs;       // 当前关键帧列表
    bool cloudbusy = false;                     // 点云处理忙标志
    bool loopbusy = false;                      // 回环处理忙标志
    bool bStop = false;                         // 停止标志
    std::atomic<bool> mabIsUpdating;            // 是否正在更新点云

protected:
    /**
     * @brief 从关键帧生成点云
     * @param kf 关键帧指针
     */
    void generatePointCloud(KeyFrame *kf);

    // 新关键帧队列
    std::list<KeyFrame *> mlNewKeyFrames;
    
    // 全局点云地图
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr globalMap;
    
    // 可视化线程
    std::shared_ptr<std::thread> viewerThread;

    // 关闭标志和互斥锁
    bool shutDownFlag = false;
    std::mutex shutDownMutex;

    // 条件变量：关键帧更新通知
    std::condition_variable keyFrameUpdated;
    
    // 互斥锁
    std::mutex mMutexGlobalMap;     // 全局地图互斥锁
    std::mutex keyframeMutex;       // 关键帧队列互斥锁
    std::mutex updateMutex;         // 更新互斥锁

    // 点云数据存储
    std::vector<PointCloude> pointcloud;
    std::vector<KeyFrame *> keyframes;
    std::vector<cv::Mat> colorImgs;
    std::vector<cv::Mat> depthImgs;
    std::vector<cv::Mat> colorImgks;
    std::vector<cv::Mat> depthImgks;
    std::vector<int> ids;

    uint16_t lastKeyframeSize = 0;

    // 滤波参数
    double resolution = 0.04;   // 体素滤波分辨率
    double meank = 50;          // 统计滤波邻域点数
    double thresh = 1;          // 统计滤波标准差倍数
    double mDepthMapFactor = 1000.0; // 深度图因子

    // PCL滤波器
    pcl::VoxelGrid<pcl::PointXYZRGBA> *voxel;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> *statistical_filter;
};

} // namespace ORB_SLAM3

#endif // POINTCLOUDMAPPING_H
