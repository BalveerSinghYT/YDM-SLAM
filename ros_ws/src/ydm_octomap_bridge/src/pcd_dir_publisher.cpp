#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cctype>
#include <dirent.h>
#include <sys/stat.h>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::vector<std::string> ListPcdFiles(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) return out;

  while (dirent* ent = readdir(d)) {
    const std::string name(ent->d_name);
    if (name == "." || name == "..") continue;
    if (!EndsWith(name, ".pcd")) continue;
    out.push_back(name);
  }
  closedir(d);

  std::sort(out.begin(), out.end());
  return out;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

bool FileExists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcd_dir_publisher");
  ros::NodeHandle nh("~");

  std::string static_dir;
  std::string dynamic_dir;
  std::string frame_id;
  double rate_hz = 1.0;
  bool loop = true;
  bool follow = true;
  bool publish_dynamic = true;

  nh.param<std::string>("static_dir", static_dir, std::string("pcd"));
  nh.param<std::string>("dynamic_dir", dynamic_dir, std::string("pcd_dyn"));
  nh.param<std::string>("frame_id", frame_id, std::string("map"));
  nh.param<double>("rate_hz", rate_hz, 1.0);
  nh.param<bool>("loop", loop, true);
  nh.param<bool>("follow", follow, true);
  nh.param<bool>("publish_dynamic", publish_dynamic, true);

  ros::Publisher pub_static = nh.advertise<sensor_msgs::PointCloud2>("/ydm/static_cloud", 1, true);
  ros::Publisher pub_dynamic = nh.advertise<sensor_msgs::PointCloud2>("/ydm/dynamic_cloud", 1, true);

  ROS_INFO_STREAM("[ydm_octomap_bridge] static_dir=" << static_dir << " dynamic_dir=" << dynamic_dir
                                                    << " frame_id=" << frame_id << " rate_hz=" << rate_hz
                                                    << " follow=" << (follow ? "true" : "false")
                                                    << " loop=" << (loop ? "true" : "false")
                                                    << " publish_dynamic=" << (publish_dynamic ? "true" : "false"));

  ros::Rate rate(std::max(0.1, rate_hz));

  std::set<std::string> published;
  size_t idx = 0;  // used only when follow=false

  while (ros::ok()) {
    ros::spinOnce();

    std::vector<std::string> static_files = ListPcdFiles(static_dir);
    if (static_files.empty()) {
      rate.sleep();
      continue;
    }

    std::vector<std::string> to_publish;
    if (follow) {
      // Publish only new files (incremental, near-real-time while SLAM writes PCDs).
      for (const auto& f : static_files) {
        if (published.find(f) == published.end()) {
          to_publish.push_back(f);
        }
      }
    } else {
      // Old behavior: sequential playback.
      if (idx >= static_files.size()) {
        if (loop) idx = 0;
        else break;
      }
      to_publish.push_back(static_files[idx]);
      ++idx;
    }

    if (to_publish.empty()) {
      rate.sleep();
      continue;
    }

    // Publish at most one cloud per tick to avoid blocking the ROS thread.
    const std::string file = to_publish.front();
    const std::string static_path = JoinPath(static_dir, file);
    const std::string dynamic_path = JoinPath(dynamic_dir, file);

    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    if (pcl::io::loadPCDFile(static_path, cloud) != 0) {
      ROS_WARN_STREAM("Failed to load " << static_path);
      rate.sleep();
      continue;
    }

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;
    pub_static.publish(msg);
    published.insert(file);

    // Dynamic cloud is optional. Avoid spamming warnings when files don't exist.
    if (publish_dynamic && FileExists(dynamic_path)) {
      pcl::PointCloud<pcl::PointXYZRGBA> cloud_dyn;
      if (pcl::io::loadPCDFile(dynamic_path, cloud_dyn) == 0) {
        sensor_msgs::PointCloud2 msg_dyn;
        pcl::toROSMsg(cloud_dyn, msg_dyn);
        msg_dyn.header.stamp = msg.header.stamp;
        msg_dyn.header.frame_id = frame_id;
        pub_dynamic.publish(msg_dyn);
      }
    }
    rate.sleep();
  }

  return 0;
}
