# 模块文档：仿真基础设施

**源文件**：
- ROS2 节点：[src/tb4_drl_navigation/tb4_drl_navigation/envs/utils/ros_gz.py](../src/tb4_drl_navigation/tb4_drl_navigation/envs/utils/ros_gz.py)
- 仿真启动：[src/turtlebot4/tb4_gz_sim/launch/](../src/turtlebot4/tb4_gz_sim/launch/)
- 机器人描述：[src/turtlebot4/tb4_description/](../src/turtlebot4/tb4_description/)

---

## 1. 概述

仿真基础设施由两层组成：
- **ROS2 节点层**（`ros_gz.py`）：Python 类，封装传感器订阅、速度发布、Gazebo 服务调用
- **仿真启动层**（`tb4_gz_sim`）：ROS2 launch 文件，负责启动 Gazebo、生成机器人、配置 ROS-Gz 桥接

---

## 2. ROS2 节点（ros_gz.py）

### 2.1 Sensors 节点

订阅传感器话题，缓存最新数据供环境读取。

| 订阅话题 | 消息类型 | 用途 |
|----------|----------|------|
| `/scan` | `sensor_msgs/LaserScan` | 激光雷达数据 |
| `/odometry/filtered` | `nav_msgs/Odometry` | EKF 融合后的里程计 |

主要方法：
- `get_latest_scan()` → 原始激光距离列表
- `get_latest_pose_stamped()` → 当前位姿（PoseStamped）
- `get_range_min_max()` / `get_angle_min_max()` → 激光参数

### 2.2 Publisher 节点

发布控制指令和可视化标记。

| 发布话题 | 消息类型 | 用途 |
|----------|----------|------|
| `/cmd_vel` | `geometry_msgs/TwistStamped` | 速度指令 |
| `/visualization_marker` | `visualization_msgs/Marker` | 目标点标记（蓝色箭头） |
| `/visualization_marker_array` | `MarkerArray` | 调试可视化（激光射线、朝向弧） |
| `/path` | `nav_msgs/Path` | 机器人轨迹 |

调试可视化（`publish_observation`）：
- 激光扇区最近点射线（红色线段）
- 机器人到目标的方向箭头（绿色）
- 朝向偏差弧度标注

### 2.3 SimulationControl 节点

通过 ROS2 服务控制 Gazebo 仿真状态。

| 服务 | 类型 | 用途 |
|------|------|------|
| `/world/{name}/control` | `ControlWorld` | 暂停/恢复/重置仿真 |
| `world/{name}/set_pose` | `SetEntityPose` | 设置 Gazebo 实体位姿 |
| `set_pose` | `SetPose` | 重置 EKF 里程计原点 |
| `world/{name}/create` | `SpawnEntity` | 生成实体 |
| `world/{name}/remove` | `DeleteEntity` | 删除实体 |

关键方法：
- `reset_world()` → 重置模型（Gazebo 限制：仅重置到初始状态）
- `pause_unpause(pause)` → 暂停/恢复仿真
- `set_entity_pose(name, pose)` → 传送实体到指定位姿
- `set_pose(pose)` → 重置 robot_localization EKF 状态
- `get_obstacles(starts_with)` → 获取 Gazebo 中以指定前缀命名的模型列表

---

## 3. 仿真启动（tb4_gz_sim）

### 3.1 主启动文件：simulation.launch.py

启动完整仿真栈，包含以下组件：

```
simulation.launch.py
├── gz_sim.launch.py          # 启动 Gazebo 仿真器
├── spawn_tb4.launch.py       # 在 Gazebo 中生成 TurtleBot4
├── bridge.launch.py          # ROS-Gz 话题桥接
├── robot_state_publisher     # 发布 TF 树（基于 URDF）
├── ekf_filter_node           # robot_localization EKF（默认启用）
├── slam_toolbox              # 在线建图（可选，do_slam=true）
└── rviz2                     # 可视化（默认启用）
```

**常用启动参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `headless` | False | 无头模式（无 GUI） |
| `use_rviz` | True | 是否启动 RViz2 |
| `do_slam` | false | 是否启动 SLAM 建图 |
| `do_localization` | true | 是否启动 EKF 定位 |
| `world` | static_world.sdf | 仿真世界文件 |

### 3.2 仿真世界

- `worlds/static_world.sdf`：静态障碍物世界，包含预置障碍物（命名为 `obstacle_*`）
- `worlds/empty_world.sdf`：空世界
- `models/obstacle/`：可移动障碍物模型（圆柱体）
- `models/td7_world/`：世界环境模型

### 3.3 配置文件

| 文件 | 用途 |
|------|------|
| `configs/ekf.yaml` | EKF 参数（融合 IMU + 轮式里程计） |
| `configs/tb4_bridge.yaml` | ROS-Gz 话题桥接映射 |
| `configs/mapper_params_online_async.yaml` | slam_toolbox 参数 |

---

## 4. 机器人描述（tb4_description）

- `urdf/standard/turtlebot4.urdf.xacro`：完整机器人 URDF（含传感器）
- `urdf/icreate/create3.urdf.xacro`：iCreate3 底盘
- `urdf/sensors/rplidar.urdf.xacro`：RPLidar 激光雷达
- `urdf/sensors/oakd.urdf.xacro`：OAK-D 深度相机
- `meshes/`：视觉和碰撞网格文件（.dae）
- `rviz/config_drl.rviz`：DRL 训练专用 RViz 配置

---

## 5. 工具类

### Launcher（utils/launch.py）

封装 ROS2 工作空间构建和 launch 文件执行：
- `find_workspace(start_dir)` → 向上搜索含 `src/` 目录的工作空间根目录
- `build()` → 执行 `build.sh`
- `launch(package, launch_file, *args)` → source 工作空间后执行 `ros2 launch`

### dtype_convertor（utils/dtype_convertor.py）

- `PoseConverter.from_dict(...)` → 从字典构建 `geometry_msgs/Pose`
- `TwistConverter.from_dict(...)` → 从字典构建 `geometry_msgs/Twist`

---

## 6. ROS2 话题/服务总览

```
订阅：
  /scan                    ← Gazebo RPLidar 插件
  /odometry/filtered       ← robot_localization EKF

发布：
  /cmd_vel                 → Gazebo 差速驱动插件
  /visualization_marker    → RViz2
  /visualization_marker_array → RViz2
  /path                    → RViz2

服务调用：
  /world/static_world/control    (ControlWorld)
  world/static_world/set_pose    (SetEntityPose)
  /set_pose                      (robot_localization SetPose)
```
