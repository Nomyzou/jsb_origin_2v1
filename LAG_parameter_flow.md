# 🚀 LAG训练参数传递流程图

## 📋 1. 启动入口 (Shell脚本)
```bash
# LAG/scripts/train_selfplay.sh
env="SingleCombat"           # 环境类型
scenario="1v1/NoWeapon/Selfplay"  # 具体场景
algo="ppo"                   # 算法名称
exp="v1"                     # 实验名称
seed=1                       # 随机种子

# 调用Python训练脚本
python train/train_jsbsim.py \
    --env-name $env \
    --scenario-name $scenario \
    --algorithm-name $algo \
    --experiment-name $exp \
    --seed $seed \
    [其他参数...]
```

## 🔄 2. 参数解析流程
```
Shell变量 → 命令行参数 → argparse解析 → all_args对象
     ↓
env="SingleCombat" → --env-name SingleCombat
scenario="1v1/NoWeapon/Selfplay" → --scenario-name 1v1/NoWeapon/Selfplay
algo="ppo" → --algorithm-name ppo
exp="v1" → --experiment-name v1
seed=1 → --seed 1
```

## 🏗️ 3. 环境创建流程
```
train_jsbsim.py
    ↓
make_train_env(all_args)
    ↓
SingleCombatEnv(all_args.scenario_name)
    ↓
BaseEnv.__init__("1v1/NoWeapon/Selfplay")
    ↓
parse_config("1v1/NoWeapon/Selfplay")
    ↓
读取: LAG/envs/JSBSim/configs/1v1/NoWeapon/Selfplay.yaml
    ↓
返回EnvConfig对象
    ↓
BaseEnv.load()
    ↓
load_task() + load_simulator()
```

## 📁 4. 配置文件解析
```yaml
# LAG/envs/JSBSim/configs/1v1/NoWeapon/Selfplay.yaml
task: singlecombat                    # 决定使用哪个Task类
sim_freq: 60                          # 仿真频率
agent_interaction_steps: 12           # 智能体交互步数
max_steps: 1000                       # 最大步数
aircraft_configs:                     # 飞机配置
  A0100: {color: Red, model: f16, init_state: {...}}
  B0100: {color: Blue, model: f16, init_state: {...}}
battle_field_center: [120.0, 60.0, 0.0]
reward_config: {...}
```

## 🎯 5. Task类选择逻辑
```
SingleCombatEnv.load_task()
    ↓
taskname = self.config.task  # 从YAML读取: "singlecombat"
    ↓
if taskname == 'singlecombat':
    self.task = SingleCombatTask(self.config)
elif taskname == 'hierarchical_singlecombat':
    self.task = HierarchicalSingleCombatTask(self.config)
elif taskname == 'singlecombat_dodge_missile':
    self.task = SingleCombatDodgeMissileTask(self.config)
# ... 其他task类型
```

## 🏃 6. Runner类选择逻辑
```
train_jsbsim.py
    ↓
if all_args.env_name == "MultipleCombat":
    runner = ShareJSBSimRunner(config)
else:
    if all_args.use_selfplay:
        from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
    else:
        from runner.jsbsim_runner import JSBSimRunner as Runner
    runner = Runner(config)
```

## 🔧 7. 算法组件创建
```
SelfplayJSBSimRunner.load()
    ↓
创建Policy: PPOPolicy(all_args, obs_space, act_space, device)
    ↓
创建Trainer: PPOTrainer(all_args, device)
    ↓
创建ReplayBuffer: ReplayBuffer(all_args)
    ↓
初始化自对弈算法: FSP(all_args)
    ↓
创建对手策略池
```

## 📊 8. 训练循环流程
```
SelfplayJSBSimRunner.run()
    ↓
self.warmup()                    # 预热阶段
    ↓
训练循环:
    ↓
self.collect()                   # 收集训练数据
    ↓
self.insert()                    # 插入经验回放缓冲区
    ↓
self.train()                     # 训练策略网络
    ↓
self.eval()                      # 评估当前策略
    ↓
保存模型到: results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/
```

## 🎮 9. 环境交互流程
```
env.step(action)
    ↓
SingleCombatEnv.step()
    ↓
SingleCombatTask.step()
    ↓
计算奖励: [AltitudeReward, PostureReward, EventDrivenReward]
    ↓
检查终止条件: [LowAltitude, ExtremeState, Overload, SafeReturn, Timeout]
    ↓
返回: obs, rewards, dones, info
```

## 📁 10. 关键文件调用关系
```
train_selfplay.sh
    ↓
train_jsbsim.py
    ↓
singlecombat_env.py (SingleCombatEnv)
    ↓
env_base.py (BaseEnv)
    ↓
utils.py (parse_config)
    ↓
Selfplay.yaml (配置文件)
    ↓
singlecombat_task.py (SingleCombatTask)
    ↓
selfplay_jsbsim_runner.py (SelfplayJSBSimRunner)
    ↓
ppo_policy.py (PPOPolicy)
    ↓
ppo_trainer.py (PPOTrainer)
```

## 🔑 11. 核心参数映射表
| Shell变量 | 命令行参数 | 解析后变量 | 用途 | 影响对象 |
|-----------|------------|-------------|------|----------|
| `env` | `--env-name` | `all_args.env_name` | 环境类型选择 | 环境类、Runner类 |
| `scenario` | `--scenario-name` | `all_args.scenario_name` | 场景配置 | YAML文件、Task类 |
| `algo` | `--algorithm-name` | `all_args.algorithm_name` | 算法选择 | Policy类、Trainer类 |
| `exp` | `--experiment-name` | `all_args.experiment_name` | 实验标识 | 结果保存路径 |
| `seed` | `--seed` | `all_args.seed` | 随机种子 | 所有随机过程 |

## 📍 12. 结果保存路径
```
results/
└── SingleCombat/                    # env_name
    └── 1v1/NoWeapon/Selfplay/      # scenario_name
        └── ppo/                     # algorithm_name
            └── v1/                  # experiment_name
                ├── run1/
                ├── run2/
                └── ...
``` 