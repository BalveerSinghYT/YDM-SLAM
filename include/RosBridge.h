#pragma once

#ifdef YDM_WITH_ROS

#include <memory>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sophus/se3.hpp>
#include <Eigen/Core>

namespace ORB_SLAM3 {

class RosBridge {
public:
    static RosBridge& Instance();

    // Safe to call even if ros::init() was not called; it will no-op.
    bool IsReady() const;

    void Publish(
        const pcl::PointCloud<pcl::PointXYZRGBA>& cloud_orb,
        const pcl::PointCloud<pcl::PointXYZRGBA>* cloud_dyn_orb,
        const Sophus::SE3f& Twc_orb,
        double timestamp_sec);

private:
    RosBridge();

    void EnsureInit();
    void PublishTF(const Sophus::SE3f& Twc_orb, const ros::Time& stamp);

    // ORB(optical-like) -> ROS(map ENU-like): [x y z] -> [z, -x, -y]
    static Eigen::Matrix3f R_orb_to_ros();

private:
    bool initialized_ = false;
    std::unique_ptr<ros::NodeHandle> nh_;

    ros::Publisher pub_static_;
    ros::Publisher pub_dynamic_;
    ros::Publisher pub_pose_;

    tf2_ros::TransformBroadcaster tf_br_;

    std::string frame_id_map_;
    std::string frame_id_camera_;
};

} // namespace ORB_SLAM3

#endif // YDM_WITH_ROS
