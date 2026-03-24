# 模块文档：DRL 训练

**源文件**：[src/tb4_drl_navigation/tb4_drl_navigation/examples/sac.py](../src/tb4_drl_navigation/tb4_drl_navigation/examples/sac.py)

---
?
## 1. 概述

`sac.py` 是基于 **Soft Actor-Critic (SAC)** 算法的导航训练示例，使用 stable-baselines3 实现。
提供训练（`train`）和推理（`eval`）两种模式，支持断点续训。

---

## 2. 配置结构

使用 Python `dataclass` 管理配置，分三层：

```
ExperimentConfig
├── EnvConfig       # 环境参数
├── SACConfig       # SAC 算法超参数
└── 实验级参数      # 总步数、保存频率、日志路径等
```

### EnvConfig（环境配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `env_id` | `Turtlebot4Env-v0` | Gymnasium 环境 ID |
| `world_name` | `static_world` | Gazebo 世界名称 |
| `num_bins` | 20 | 激光分箱数 |
| `goal_threshold` | 0.35 m | 到达目标阈值 |
| `collision_threshold` | 0.4 m | 碰撞检测阈值 |
| `time_delta` | 0.1 s | 每步仿真时长（训练时比默认值更短） |
| `goal_sampling_bias` | `close` | 目标采样偏置（训练时偏近） |
| `shuffle_on_reset` | True | 每回合随机化障碍物 |

### SACConfig（算法配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `policy_type` | `MlpPolicy` | 策略网络类型 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `batch_size` | 1024 | 训练批大小 |
| `gamma` | 0.99 | 折扣因子 |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `net_arch` | pi:[512,512,256], qf:[1024,1024,512] | 策略/Q网络结构 |

### ExperimentConfig（实验配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `total_timesteps` | 1,000,000 | 总训练步数 |
| `save_freq` | 5000 | 检查点保存频率 |
| `eval_freq` | 5000 | 评估频率 |
| `n_eval_episodes` | 5 | 每次评估的回合数 |
| `seed` | 42 | 随机种子 |
| `log_dir` | `experiments/` | 实验输出目录 |

---

## 3. 训练流程（SACExperiment）

```
SACExperiment.train()
├── 初始化目录结构
│   ├── experiments/sac_navigation/checkpoints/
│   ├── experiments/sac_navigation/best_model/
│   ├── experiments/sac_navigation/logs/          ← TensorBoard
│   └── experiments/sac_navigation/eval_logs/
├── 设置随机种子
├── 创建/加载 SAC 模型
├── 注册回调
│   ├── CheckpointCallback（每 5000 步保存，含 replay buffer）
│   └── EvalCallback（每 5000 步评估，保存最优模型）
└── model.learn(total_timesteps, progress_bar=True)
    └── 训练结束/中断后保存 final_model + replay_buffer + config.yaml
```

### 断点续训

```bash
python3 sac.py train --resume experiments/sac_navigation/checkpoints/sac_model_40000_steps.zip
```

续训逻辑：
1. 从 checkpoint `.zip` 加载模型
2. 从文件名解析已完成步数（`sac_model_{steps}_steps.zip`）
3. 自动加载同目录下的 `*_replay_buffer.pkl`（若存在）
4. 计算剩余步数继续训练

---

## 4. 推理流程（SACInference）

```bash
python3 sac.py eval experiments/sac_navigation/best_model/best_model.zip --episodes 10
```

- 加载模型，确定性推理（`deterministic=True`）
- 循环执行 `num_episodes` 个回合直到终止或截断

---

## 5. 环境包装链

```python
Turtlebot4Env                    # 原始环境（Dict 观测）
  → TimeLimit                    # Gymnasium 自动添加（max_episode_steps）
  → OrderEnforcing               # Gymnasium 自动添加
  → PassiveEnvChecker            # Gymnasium 自动添加
  → FlattenObservation           # 展平 Dict → 一维向量（训练必需）
  → Monitor                      # stable-baselines3 监控（记录 episode reward/length）
```

---

## 6. 使用 TensorBoard 监控训练

```bash
tensorboard --logdir experiments/sac_navigation/logs
```

---

## 7. 扩展自定义算法

替换 SAC 只需修改 `SACExperiment` 中的模型初始化部分，环境接口保持不变：

```python
from stable_baselines3 import TD3
model = TD3('MlpPolicy', env, ...)
```

或使用自定义配置：

```python
config = ExperimentConfig(
    env=EnvConfig(num_bins=40, time_delta=0.2),
    sac=SACConfig(batch_size=512, learning_rate=1e-4),
    total_timesteps=2_000_000,
)
```
