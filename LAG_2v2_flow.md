# 🚀 LAG 2v2多智能体训练调用流程图

## 📋 **1. 启动入口 (Shell脚本)**
```bash
# LAG/scripts/train_share_selfplay.sh
env="MultipleCombat"                    # 多智能体战斗环境
scenario="2v2/NoWeapon/HierarchySelfplay"  # 2v2无武器层次化自对弈
algo="mappo"                            # 多智能体PPO算法
exp="v1"                                # 实验版本
seed=0                                  # 随机种子

# 调用Python训练脚本
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} \
    [其他参数...]
```

## 🔄 **2. 主训练流程调用链**
```
train_share_selfplay.sh
    ↓
train_jsbsim.py (主入口)
    ↓
main() 函数
    ↓
if all_args.env_name == "MultipleCombat":
    runner = ShareJSBSimRunner(config)  # 选择多智能体训练器
    ↓
ShareJSBSimRunner.load()  # 加载算法组件
    ↓
ShareJSBSimRunner.run()   # 开始训练循环
```

## 🏗️ **3. 环境创建流程**
```
make_train_env(all_args)
    ↓
if all_args.env_name == "MultipleCombat":
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])      # 单进程环境
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])  # 多进程环境
    ↓
get_env_fn(i)() → MultipleCombatEnv(scenario_name)
    ↓
MultipleCombatEnv.__init__("2v2/NoWeapon/HierarchySelfplay")
    ↓
BaseEnv.__init__() → parse_config() → 读取YAML配置文件
    ↓
MultipleCombatEnv.load_task()
    ↓
根据task选择Task类:
    - task: "hierarchical_multiplecombat" → HierarchicalMultipleCombatTask
    - task: "multiplecombat" → MultipleCombatTask
    - task: "hierarchical_multiplecombat_shoot" → HierarchicalMultipleCombatShootTask
```

## 📁 **4. 配置文件解析**
```yaml
# LAG/envs/JSBSim/configs/2v2/NoWeapon/HierarchySelfplay.yaml
task: hierarchical_multiplecombat        # 层次化多智能体战斗任务
sim_freq: 60                            # 仿真频率 (60Hz)
agent_interaction_steps: 12             # 智能体交互步数 (0.2秒)
max_steps: 1000                         # 最大步数 (200秒)

# 4架飞机配置 (2v2)
aircraft_configs: {
  A0100: {color: Red, model: f16, ...}     # 红方飞机1
  A0200: {color: Red, model: f16, ...}     # 红方飞机2
  B0100: {color: Blue, model: f16, ...}    # 蓝方飞机1
  B0200: {color: Blue, model: f16, ...}    # 蓝方飞机2
}

# 奖励配置
PostureReward_scale: 15.0               # 姿态奖励系数
AltitudeReward_safe_altitude: 4.0       # 安全高度
EventDrivenReward_scale: 1              # 事件驱动奖励
```

## 🎯 **5. Task类选择逻辑**
```
MultipleCombatEnv.load_task()
    ↓
taskname = self.config.task  # "hierarchical_multiplecombat"
    ↓
if taskname == 'hierarchical_multiplecombat':
    self.task = HierarchicalMultipleCombatTask(self.config)
elif taskname == 'multiplecombat':
    self.task = MultipleCombatTask(self.config)
elif taskname == 'hierarchical_multiplecombat_shoot':
    self.task = HierarchicalMultipleCombatShootTask(self.config)
```

## 🏃 **6. Runner类选择逻辑**
```
train_jsbsim.py
    ↓
if all_args.env_name == "MultipleCombat":
    runner = ShareJSBSimRunner(config)  # 多智能体共享训练器
else:
    # 单智能体环境选择其他Runner
    ↓
ShareJSBSimRunner.load()
    ↓
if self.algorithm_name == "mappo":
    from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
    from algorithms.mappo.ppo_policy import PPOPolicy as Policy
    ↓
创建Policy和Trainer
    ↓
创建SharedReplayBuffer (支持共享观察)
```

## 🔧 **7. 算法组件创建**
```
ShareJSBSimRunner.load()
    ↓
创建Policy: PPOPolicy(all_args, obs_space, share_obs_space, act_space, device)
    ↓
创建Trainer: PPOTrainer(all_args, device)
    ↓
创建SharedReplayBuffer: SharedReplayBuffer(all_args, num_agents, obs_space, share_obs_space, act_space)
    ↓
初始化自对弈算法 (FSP)
    ↓
创建对手策略池
```

## 📊 **8. 训练循环流程**
```
ShareJSBSimRunner.run()
    ↓
self.warmup()                    # 预热阶段
    ↓
训练循环 (episodes):
    ↓
for step in range(self.buffer_size):  # 收集3000步数据
    ↓
self.collect()                   # 收集训练数据 (4个智能体)
    ↓
self.envs.step(actions)          # 环境步进 (32个并行环境)
    ↓
self.insert(data)                # 插入共享经验回放缓冲区
    ↓
self.compute()                   # 计算回报和优势
    ↓
self.train()                     # 训练策略网络 (MAPPO)
    ↓
self.eval()                      # 评估当前策略
    ↓
保存模型到: results/MultipleCombat/2v2/NoWeapon/HierarchySelfplay/mappo/v1/
```

## 🎮 **9. 环境交互流程**
```
env.step(action)  # action: [4个智能体的动作]
    ↓
MultipleCombatEnv.step()
    ↓
HierarchicalMultipleCombatTask.step()
    ↓
应用动作到4架飞机
    ↓
运行仿真 (12步 × 60Hz = 0.2秒)
    ↓
计算奖励:
    - AltitudeReward: 高度奖励
    - PostureReward: 姿态优势奖励  
    - EventDrivenReward: 事件驱动奖励
    ↓
检查终止条件:
    - SafeReturn: 安全返回
    - ExtremeState: 极端状态
    - Overload: 过载
    - LowAltitude: 低高度
    - Timeout: 超时
    ↓
返回: obs, share_obs, rewards, dones, info
```

## 🧠 **10. 层次化控制架构**
```
HierarchicalMultipleCombatTask
    ↓
高层策略 (3维离散动作):
    - delta_altitude: [-0.1, 0, 0.1]      # 高度变化
    - delta_heading: [-π/6, -π/12, 0, π/12, π/6]  # 航向变化
    - delta_velocity: [-0.05, 0, 0.05]    # 速度变化
    ↓
低层策略 (BaselineActor):
    - 加载预训练的baseline_model.pt
    - 将高层动作转换为低层控制指令
    - 输出4维连续动作: [aileron, elevator, rudder, throttle]
```

## 🔄 **11. 自对弈训练机制**
```
FSP (Fictitious Self-Play) 算法
    ↓
对手策略池管理:
    - 保存历史版本的策略
    - 定期更新对手策略
    - 选择不同对手进行训练
    ↓
训练数据分配:
    - 红方智能体 (A0100, A0200) 共享策略
    - 蓝方智能体 (B0100, B0200) 共享策略
    - 通过自对弈提升策略质量
```

## 📁 **12. 关键文件调用关系**
```
train_share_selfplay.sh
    ↓
train_jsbsim.py
    ↓
multiplecombat_env.py (MultipleCombatEnv)
    ↓
env_base.py (BaseEnv)
    ↓
utils.py (parse_config)
    ↓
2v2/NoWeapon/HierarchySelfplay.yaml (配置文件)
    ↓
multiplecombat_task.py (HierarchicalMultipleCombatTask)
    ↓
share_jsbsim_runner.py (ShareJSBSimRunner)
    ↓
mappo/ppo_policy.py (PPOPolicy)
    ↓
mappo/ppo_trainer.py (PPOTrainer)
```

## 🎯 **13. 2v2场景特点**
```
场景配置: 2v2/NoWeapon/HierarchySelfplay
    ↓
4架F16战斗机:
    - 红方: A0100, A0200 (初始位置: 120.0°E, 60.0°N, 20000ft)
    - 蓝方: B0100, B0200 (初始位置: 120.0°E, 60.1°N, 20000ft)
    ↓
任务目标: 获得姿态优势 (飞向敌机尾部，保持适当距离)
    ↓
层次化控制: 高层决策 + 低层执行
    ↓
多智能体协调: 4个智能体共享观察，独立决策
```

## 📊 **14. 训练参数配置**
```
硬件配置:
    - GPU: CUDA_VISIBLE_DEVICES=0
    - 并行环境: 32个 (n-rollout-threads=32)
    - 训练线程: 1个 (n-training-threads=1)

算法参数:
    - 学习率: 3e-4
    - PPO轮数: 4
    - 缓冲区大小: 3000步
    - Mini-batch: 5个
    - 总训练步数: 1亿步

网络结构:
    - 隐藏层: [128, 128]
    - RNN层: 1层，128维
    - 数据块长度: 8
```

## 🔑 **15. 关键优势**
1. **层次化架构**: 分离高层决策和低层控制，降低学习难度
2. **多智能体协调**: 4个智能体协同作战，学习团队策略
3. **自对弈训练**: 通过FSP算法持续提升策略质量
4. **共享观察**: 支持智能体间的信息共享和协调
5. **高并行度**: 32个环境并行训练，提高训练效率 