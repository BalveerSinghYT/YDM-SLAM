#ifdef YDM_WITH_ROS

#include "RosBridge.h"

#include <cstdlib>

#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/TransformStamped.h>

namespace ORB_SLAM3 {

RosBridge& RosBridge::Instance() {
    static RosBridge instance;
    return instance;
}

RosBridge::RosBridge() {
    // Lazy init
}

bool RosBridge::IsReady() const {
    return initialized_ && ros::ok();
}

void RosBridge::EnsureInit() {
    if (initialized_) return;
    if (!ros::isInitialized()) return;

    nh_ = std::make_unique<ros::NodeHandle>();

    frame_id_map_ = "map";
    frame_id_camera_ = "camera";
    if (const char* e = std::getenv("YDM_ROS_FRAME_ID")) frame_id_map_ = std::string(e);
    if (const char* e = std::getenv("YDM_ROS_CAMERA_FRAME")) frame_id_camera_ = std::string(e);

    // Latch point clouds so late subscribers can still get the last frame.
    pub_static_ = nh_->advertise<sensor_msgs::PointCloud2>("/ydm/static_cloud", 1, true);
    pub_dynamic_ = nh_->advertise<sensor_msgs::PointCloud2>("/ydm/dynamic_cloud", 1, true);
    pub_pose_ = nh_->advertise<geometry_msgs::PoseStamped>("/ydm/camera_pose", 1);

    initialized_ = true;
}

Eigen::Matrix3f RosBridge::R_orb_to_ros() {
    Eigen::Matrix3f R;
    R << 0.f, 0.f, 1.f,
        -1.f, 0.f, 0.f,
         0.f,-1.f, 0.f;
    return R;
}

void RosBridge::PublishTF(const Sophus::SE3f& Twc_orb, const ros::Time& stamp) {
    const Eigen::Matrix3f R = R_orb_to_ros();

    const Eigen::Matrix3f Rwc_orb = Twc_orb.rotationMatrix();
    const Eigen::Vector3f twc_orb = Twc_orb.translation();

    const Eigen::Matrix3f Rwc_ros = R * Rwc_orb * R.transpose();
    const Eigen::Vector3f twc_ros = R * twc_orb;

    Eigen::Quaternionf q(Rwc_ros);
    q.normalize();

    geometry_msgs::TransformStamped tfm;
    tfm.header.stamp = stamp;
    tfm.header.frame_id = frame_id_map_;
    tfm.child_frame_id = frame_id_camera_;
    tfm.transform.translation.x = twc_ros.x();
    tfm.transform.translation.y = twc_ros.y();
    tfm.transform.translation.z = twc_ros.z();
    tfm.transform.rotation.x = q.x();
    tfm.transform.rotation.y = q.y();
    tfm.transform.rotation.z = q.z();
    tfm.transform.rotation.w = q.w();

    tf_br_.sendTransform(tfm);

    geometry_msgs::PoseStamped pose;
    pose.header = tfm.header;
    pose.pose.position.x = tfm.transform.translation.x;
    pose.pose.position.y = tfm.transform.translation.y;
    pose.pose.position.z = tfm.transform.translation.z;
    pose.pose.orientation = tfm.transform.rotation;
    pub_pose_.publish(pose);
}

void RosBridge::Publish(
    const pcl::PointCloud<pcl::PointXYZRGBA>& cloud_orb,
    const pcl::PointCloud<pcl::PointXYZRGBA>* cloud_dyn_orb,
    const Sophus::SE3f& Twc_orb,
    double timestamp_sec)
{
    EnsureInit();
    if (!IsReady()) return;

    const ros::Time stamp = (timestamp_sec > 0.0) ? ros::Time(timestamp_sec) : ros::Time::now();

    // Publish TF/pose
    PublishTF(Twc_orb, stamp);

    // Convert clouds into ROS map frame (axis fix)
    const Eigen::Matrix3f R = R_orb_to_ros();

    pcl::PointCloud<pcl::PointXYZRGBA> cloud_ros;
    cloud_ros.reserve(cloud_orb.size());
    for (const auto& p : cloud_orb.points) {
        pcl::PointXYZRGBA q = p;
        const Eigen::Vector3f v(p.x, p.y, p.z);
        const Eigen::Vector3f vr = R * v;
        q.x = vr.x();
        q.y = vr.y();
        q.z = vr.z();
        cloud_ros.points.push_back(q);
    }
    cloud_ros.width = cloud_ros.points.size();
    cloud_ros.height = 1;
    cloud_ros.is_dense = true;

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud_ros, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id_map_;
    pub_static_.publish(msg);

    if (cloud_dyn_orb && !cloud_dyn_orb->empty()) {
        pcl::PointCloud<pcl::PointXYZRGBA> cloud_dyn_ros;
        cloud_dyn_ros.reserve(cloud_dyn_orb->size());
        for (const auto& p : cloud_dyn_orb->points) {
            pcl::PointXYZRGBA q = p;
            const Eigen::Vector3f v(p.x, p.y, p.z);
            const Eigen::Vector3f vr = R * v;
            q.x = vr.x();
            q.y = vr.y();
            q.z = vr.z();
            cloud_dyn_ros.points.push_back(q);
        }
        cloud_dyn_ros.width = cloud_dyn_ros.points.size();
        cloud_dyn_ros.height = 1;
        cloud_dyn_ros.is_dense = true;

        sensor_msgs::PointCloud2 msgd;
        pcl::toROSMsg(cloud_dyn_ros, msgd);
        msgd.header.stamp = stamp;
        msgd.header.frame_id = frame_id_map_;
        pub_dynamic_.publish(msgd);
    }
}

} // namespace ORB_SLAM3

#endif // YDM_WITH_ROS
