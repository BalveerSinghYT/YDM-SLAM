# ydm_move_base_demo

Minimal `move_base` demo configuration that consumes YDM-SLAM PointCloud2 topics as 3D obstacle sources via `costmap_2d` `VoxelLayer`.

## Expected inputs
- TF chain: dataset playback often only provides `map -> camera` (no `base_link/odom`).
- PointCloud2 topics:
  - `/ydm/static_cloud`
  - `/ydm/dynamic_cloud`

## Build
From `ros_ws/`:
- `catkin_make`

## Run (in separate terminals)
1) Start roscore (if not already running):
- `roscore`

2) Start YDM-SLAM (must publish `/ydm/*` topics)

3) Start move_base demo:
- `source devel/setup.bash`
- `roslaunch ydm_move_base_demo move_base_ydm_demo.launch`

## Visualize
- Use RViz to view costmaps and voxel grid topics under `/move_base/*`.

Note: This demo defaults to `base_local_planner/TrajectoryPlannerROS` because `dwa_local_planner/DWAPlannerROS` may not be installed.

Note: This demo does not require `/map`. Both global and local costmaps use rolling windows.
