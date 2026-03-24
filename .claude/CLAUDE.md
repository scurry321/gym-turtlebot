# gym-turtlebot 项目概述

> 本文档为 Claude Code 工作上下文文件，描述项目架构、模块职责和关键设计决策。
> 模块详细文档见 [modules/](modules/) 目录。

---

## 1. 项目定位

`gym-turtlebot` 是一个基于 **ROS 2 Jazzy + Gazebo Harmonic** 的深度强化学习 (DRL) 导航训练框架。
它将 TurtleBot4 机器人仿真封装为标准 [Gymnasium](https://gymnasium.farama.org/) 接口，
使研究者可以直接用 stable-baselines3 等库训练导航策略。

**核心价值**：把 ROS2/Gazebo 仿真的复杂性隐藏在 `gym.Env` 接口后面，让 DRL 训练代码与机器人平台解耦。

---

## 2. 技术栈

| 层次 | 技术 |
|------|------|
| 机器人仿真 | ROS 2 Jazzy, Gazebo Harmonic |
| RL 框架 | Gymnasium, stable-baselines3 (SAC) |
| 传感器融合 | robot_localization (EKF) |
| 建图 | slam_toolbox (可选) |
| 语言 | Python 3.12 |
| 包管理 | uv (pyproject.toml) |

---

## 3. 项目结构

```
gym-turtlebot/
├── src/
│   ├── tb4_drl_navigation/          # 主 DRL 环境包（ROS2 Python 包）
│   │   └── tb4_drl_navigation/
│   │       ├── envs/diffdrive/      # Gymnasium 环境实现
│   │       ├── envs/utils/          # ROS2 节点工具
│   │       ├── utils/               # 通用工具（启动器、类型转换）
│   │       ├── wrappers/            # Gymnasium 包装器
│   │       └── examples/            # 训练示例（SAC）
│   └── turtlebot4/                  # 机器人描述与仿真包
│       ├── tb4_description/         # URDF/xacro 机器人模型
│       └── tb4_gz_sim/              # Gazebo 仿真启动与配置
├── docs/                            # 用户文档（教程、备忘录）
├── build.sh / setup.sh / test.sh    # 工作空间脚本
└── pyproject.toml                   # Python 依赖
```

---

## 4. 模块说明

| 模块 | 职责 | 详细文档 |
|------|------|----------|
| Gymnasium 环境 | 定义观测/动作空间、奖励函数、回合逻辑 | [modules/env.md](modules/env.md) |
| 仿真基础设施 | ROS2 节点、Gazebo 控制、传感器订阅 | [modules/simulation.md](modules/simulation.md) |
| DRL 训练 | SAC 算法配置、训练/推理流程 | [modules/drl.md](modules/drl.md) |

---

## 5. 快速开始

```bash
# 1. 安装依赖
./setup.sh

# 2. 构建工作空间
./build.sh && source install/local_setup.bash

# 3. 启动仿真
ros2 launch tb4_gz_sim simulation.launch.py

# 4. 训练（另开终端）
python3 src/tb4_drl_navigation/tb4_drl_navigation/examples/sac.py train

# 5. 推理评估
python3 src/tb4_drl_navigation/tb4_drl_navigation/examples/sac.py eval <model_path>
```

---

## 6. 关键设计决策

### 6.1 Gazebo 不支持模型重置的绕过方案
Gazebo Harmonic 暂不支持完整的模型状态重置。项目通过两步组合实现回合重置：
1. `robot_localization.SetPose` 服务 → 重置 EKF 里程计
2. `ros_gz_interfaces.SetEntityPose` 服务 → 重置 Gazebo 中实体位姿

### 6.2 观测空间设计
激光雷达原始数据（通常 360 个点）被分箱压缩为 `num_bins`（默认 20~30）个扇区的最小距离，
同时记录每个扇区最近点的方位角，大幅降低观测维度。

### 6.3 奖励函数
奖励由四部分构成，详见 [modules/env.md](modules/env.md#奖励函数)。
关键设计：障碍物惩罚按 `|cos(angle)|` 加权，避免机器人横穿两柱间时受到不合理惩罚。

### 6.4 场景随机化
`ScenarioGenerator` 基于占用栅格地图（PGM + YAML）生成随机起点/终点对和障碍物位置，
每次 `reset()` 时可选择重新洗牌障碍物（`shuffle_on_reset=True`）。
