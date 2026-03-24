# 模块文档：Gymnasium 环境

**源文件**：[src/tb4_drl_navigation/tb4_drl_navigation/envs/diffdrive/](../src/tb4_drl_navigation/tb4_drl_navigation/envs/diffdrive/)

---

## 1. 概述

`Turtlebot4Env` 是项目的核心类，继承自 `gymnasium.Env`，将 TurtleBot4 在 Gazebo 中的导航任务
封装为标准 RL 接口。注册 ID 为 `Turtlebot4Env-v0`。

```python
env = gym.make('Turtlebot4Env-v0', world_name='static_world')
env = gym.wrappers.FlattenObservation(env)  # 推荐：展平字典观测
```

---

## 2. 观测空间

观测为 `spaces.Dict`，包含 5 个字段：

| 字段 | 形状 | 范围 | 含义 |
|------|------|------|------|
| `min_ranges` | `(num_bins,)` | `[range_min, range_max]` | 各激光扇区最小距离 |
| `min_ranges_angle` | `(num_bins,)` | `[angle_min, angle_max]` | 各扇区最近点方位角 |
| `dist_to_goal` | `(1,)` | `[0, 100]` m | 到目标的欧氏距离 |
| `orient_to_goal` | `(1,)` | `[-π, π]` rad | 机器人朝向与目标方向的夹角 |
| `action` | `(2,)` | `[0,-1]~[1,1]` | 上一步执行的动作 |

**激光分箱处理**（`_process_lidar`）：
- 将 360 个原始激光点按角度均匀分为 `num_bins` 个扇区
- 每个扇区取最小距离值及其对应角度
- 无效读数（NaN/Inf/超范围）用 `range_max` 填充

---

## 3. 动作空间

连续二维动作：
```
Box(low=[0.0, -1.0], high=[1.0, 1.0], shape=(2,), dtype=float32)
action[0]: 线速度（前进，归一化）
action[1]: 角速度（转向，归一化）
```

动作通过 `TwistConverter` 转为 ROS2 `TwistStamped` 消息发布到 `/cmd_vel`。

---

## 4. 奖励函数

奖励由四部分构成（`_get_reward` 方法）：

| 情形 | 奖励值 |
|------|--------|
| 到达目标（dist < goal_threshold） | **+200** |
| 碰撞（min_range < collision_threshold） | **-100** |
| 障碍物惩罚（前向扇区，dist < 1.0m） | `(min_front_range - 1) / 2 * \|cos(angle)\|` |
| 动作奖励（每步） | `linear * cos(orient_to_goal) / 2 - \|angular\| / 2 - 0.001` |

**设计要点**：
- 动作奖励用 `cos(orient_to_goal)` 替代纯线速度，背离目标时给负奖励
- 障碍物惩罚按 `|cos(angle)|` 加权：正前方权重最大，垂直方向权重为 0
  （避免机器人横穿两柱间时受到不合理惩罚）
- 微小时间惩罚 `-0.001` 鼓励尽快到达目标

---

## 5. 回合流程

### reset()
1. 调用 `SimulationControl.reset_world()`（仅重置模型初始状态，Gazebo 限制）
2. 通过 `ScenarioGenerator` 随机采样起点/终点（或接受 `options` 中的指定值）
3. 用 `SetEntityPose` + `SetPose` 双重重置机器人位姿和里程计
4. 若 `shuffle_on_reset=True`，重新随机化障碍物位置
5. 传播仿真 `time_delta` 秒，返回初始观测

### step(action)
1. 发布速度指令
2. 解除仿真暂停，等待 `time_delta` 秒，再暂停
3. 读取传感器数据，构建观测字典
4. 判断终止条件（到达目标 / 碰撞）
5. 计算奖励，返回 `(obs, reward, terminated, truncated, info)`

---

## 6. 场景生成器（ScenarioGenerator）

**文件**：`envs/diffdrive/scenario_generator.py`

基于占用栅格地图生成随机导航场景：

1. 加载 PGM 地图 + YAML 元数据（分辨率、原点、阈值）
2. 用形态学腐蚀（`cv2.erode`）生成"缓冲自由空间"（考虑机器人半径和障碍物间距）
3. 构建 KD-Tree 加速空间查询
4. `generate_start_goal()`：采样满足最小间距约束的起点/终点对，支持 `uniform/close/far` 偏置
5. `generate_obstacles()`：在自由空间中放置 N 个障碍物，保证与起点/终点的间距

**坐标转换**：
- `map_to_world(row, col)` → 世界坐标 (x, y)，考虑地图原点旋转
- `world_to_map(x, y)` → 地图像素坐标 (row, col)

---

## 7. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_bins` | 30 | 激光分箱数量 |
| `goal_threshold` | 0.35 m | 到达目标的距离阈值 |
| `collision_threshold` | 0.4 m | 碰撞检测距离阈值 |
| `time_delta` | 0.4 s | 每步仿真时长 |
| `robot_radius` | 0.3 m | 用于地图缓冲的机器人半径 |
| `obstacle_clearance` | 2.0 m | 障碍物与起点/终点的最小间距 |
| `shuffle_on_reset` | True | 每次 reset 是否随机化障碍物 |
