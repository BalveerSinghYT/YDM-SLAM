cmake_minimum_required(VERSION 3.0.2)
project(ydm_octomap_bridge)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(pcd_dir_publisher src/pcd_dir_publisher.cpp)

target_link_libraries(pcd_dir_publisher
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
