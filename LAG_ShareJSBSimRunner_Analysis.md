# 🔍 ShareJSBSimRunner 代码功能深度解析

## 📋 **整体功能概述**

`ShareJSBSimRunner` 是一个专门为多智能体环境设计的训练器，继承自基础 `Runner` 类。它主要负责：

1. **多智能体训练管理**: 协调多个智能体的训练过程
2. **共享观察支持**: 处理智能体间的信息共享
3. **自对弈训练**: 实现FSP (Fictitious Self-Play) 算法
4. **MAPPO算法执行**: 运行多智能体PPO训练
5. **环境交互管理**: 管理训练环境和评估环境

## 🏗️ **类结构和方法分析**

### **1. 初始化方法 `load()`**

```python
def load(self):
    # 获取环境空间信息
    self.obs_space = self.envs.observation_space          # 个体观察空间
    self.share_obs_space = self.envs.share_observation_space  # 共享观察空间
    self.act_space = self.envs.action_space               # 动作空间
    self.num_agents = self.envs.num_agents                # 智能体数量
    self.use_selfplay = self.all_args.use_selfplay        # 是否使用自对弈
```

**功能解析:**
- **环境空间获取**: 从环境中提取观察、共享观察、动作空间的定义
- **智能体数量**: 确定参与训练的智能体总数
- **自对弈标志**: 判断是否启用自对弈训练模式

```python
# 算法组件创建
if self.algorithm_name == "mappo":
    from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
    from algorithms.mappo.ppo_policy import PPOPolicy as Policy
self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
self.trainer = Trainer(self.all_args, device=self.device)
```

**功能解析:**
- **算法选择**: 目前只支持MAPPO算法
- **策略网络**: 创建包含Actor和Critic的策略网络
- **训练器**: 创建PPO训练器，负责策略更新

```python
# 经验回放缓冲区创建
if self.use_selfplay:
    self.buffer = SharedReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.share_obs_space, self.act_space)
else:
    self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.share_obs_space, self.act_space)
```

**功能解析:**
- **自对弈模式**: 只存储我方智能体的经验 (num_agents // 2)
- **正常模式**: 存储所有智能体的经验
- **共享缓冲区**: 支持智能体间的经验共享

### **2. 自对弈组件初始化**

```python
if self.use_selfplay:
    from algorithms.utils.selfplay import get_algorithm
    self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
    
    # 对手策略池管理
    self.policy_pool = {'latest': self.all_args.init_elo}  # ELO评分系统
    self.opponent_policy = [
        Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        for _ in range(self.all_args.n_choose_opponents)]
    
    # 环境分配策略
    self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
    
    # 对手数据存储
    self.opponent_obs = np.zeros_like(self.buffer.obs[0])
    self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
    self.opponent_masks = np.ones_like(self.buffer.masks[0])
```

**功能解析:**
- **自对弈算法**: 支持FSP等自对弈算法
- **策略池**: 管理历史策略版本和ELO评分
- **对手策略**: 创建多个对手策略实例
- **环境分配**: 将并行环境分配给不同对手策略
- **数据存储**: 为对手策略分配独立的存储空间

### **3. 主训练循环 `run()`**

```python
def run(self):
    self.warmup()  # 预热阶段
    
    start = time.time()
    self.total_num_steps = 0
    episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
    
    for episode in range(episodes):
        # 数据收集阶段
        for step in range(self.buffer_size):
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
            obs, share_obs, rewards, dones, infos = self.envs.step(actions)
            data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
            self.insert(data)
        
        # 训练阶段
        self.compute()           # 计算回报和优势
        train_infos = self.train()  # 训练策略网络
        
        # 后处理
        self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads
        
        # 模型保存
        if (episode % self.save_interval == 0) or (episode == episodes - 1):
            self.save(episode)
        
        # 日志记录
        if episode % self.log_interval == 0:
            self.log_info(train_infos, self.total_num_steps)
        
        # 评估
        if episode % self.eval_interval == 0 and self.use_eval:
            self.eval(self.total_num_steps)
```

**功能解析:**
- **预热阶段**: 初始化环境和缓冲区
- **数据收集**: 收集一个完整缓冲区的训练数据
- **策略训练**: 使用收集的数据训练策略网络
- **定期保存**: 按间隔保存模型检查点
- **性能评估**: 定期评估当前策略性能

### **4. 数据收集 `collect()`**

```python
@torch.no_grad()
def collect(self, step):
    self.policy.prep_rollout()  # 准备推理模式
    
    # 获取我方智能体动作
    values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
        = self.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                  np.concatenate(self.buffer.obs[step]),
                                  np.concatenate(self.buffer.rnn_states_actor[step]),
                                  np.concatenate(self.buffer.rnn_states_critic[step]),
                                  np.concatenate(self.buffer.masks[step]))
    
    # 分割并行数据 [N*M, shape] => [N, M, shape]
    values = np.array(np.split(_t2n(values), self.n_rollout_threads))
    actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
    # ... 其他数据分割
    
    # 自对弈: 获取对手策略动作
    if self.use_selfplay:
        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                             np.concatenate(self.opponent_rnn_states[env_idx]),
                             np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)
    
    return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
```

**功能解析:**
- **策略推理**: 使用当前策略生成动作
- **数据分割**: 将批量数据分割为并行环境格式
- **对手动作**: 在自对弈模式下，使用对手策略生成动作
- **环境分配**: 不同环境使用不同的对手策略

### **5. 数据插入 `insert()`**

```python
def insert(self, data: List[np.ndarray]):
    obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
    
    # 处理终止状态
    dones = dones.squeeze(axis=-1)
    dones_env = np.all(dones, axis=-1)
    
    # 重置终止环境的RNN状态
    rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
    rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)
    
    # 创建掩码
    masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
    
    active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
    active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
    
    # 自对弈: 分离我方和对手数据
    if self.use_selfplay:
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.opponent_masks = masks[:, self.num_agents // 2:, ...]
        
        obs = obs[:, :self.num_agents // 2, ...]
        share_obs = share_obs[:, :self.num_agents // 2, ...]
        actions = actions[:, :self.num_agents // 2, ...]
        rewards = rewards[:, :self.num_agents // 2, ...]
        masks = masks[:, :self.num_agents // 2, ...]
        active_masks = active_masks[:, :self.num_agents // 2, ...]
    
    # 插入缓冲区
    self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values, \
        rnn_states_actor, rnn_states_critic, active_masks = active_masks)
```

**功能解析:**
- **终止处理**: 处理episode结束时的状态重置
- **RNN重置**: 终止环境重置RNN隐藏状态
- **掩码创建**: 创建用于训练的掩码
- **数据分离**: 自对弈模式下分离我方和对手数据
- **缓冲区插入**: 将处理后的数据插入经验回放缓冲区

### **6. 评估方法 `eval()`**

```python
@torch.no_grad()
def eval(self, total_num_steps):
    logging.info("\nStart evaluation...")
    total_episodes, eval_episode_rewards = 0, []
    eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
    
    # 环境重置
    eval_obs, eval_share_obs = self.eval_envs.reset()
    eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
    eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
    
    # 自对弈: 选择评估对手
    if self.use_selfplay:
        eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.all_args.n_choose_opponents)]
        eval_each_episodes = self.eval_episodes // self.all_args.n_choose_opponents
        eval_cur_opponent_idx = 0
    
    # 评估循环
    while total_episodes < self.eval_episodes:
        # 加载对手策略
        if self.use_selfplay and total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
            policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
            self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
            eval_cur_opponent_idx += 1
        
        # 获取动作并执行
        eval_actions, eval_rnn_states = self.policy.act(...)
        if self.use_selfplay:
            eval_opponent_actions, eval_opponent_rnn_states = self.eval_opponent_policy.act(...)
            eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)
        
        # 环境步进
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
        
        # 处理奖励和终止
        if self.use_selfplay:
            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
        
        eval_cumulative_rewards += eval_rewards
        eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
        total_episodes += np.sum(eval_dones_env)
        
        # 重置终止环境
        eval_masks[eval_dones_env == True] = np.zeros(...)
        eval_rnn_states[eval_dones_env == True] = np.zeros(...)
    
    # 计算评估结果
    eval_infos = {}
    eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean()
    self.log_info(eval_infos, total_num_steps)
```

**功能解析:**
- **评估初始化**: 设置评估环境和初始状态
- **对手选择**: 从策略池中选择评估对手
- **策略执行**: 使用当前策略和对手策略进行对抗
- **性能统计**: 收集和计算评估指标
- **结果记录**: 记录评估结果到日志系统

### **7. 渲染方法 `render()`**

```python
@torch.no_grad()
def render(self):
    logging.info("\nStart render ...")
    self.render_opponent_index = self.all_args.render_opponent_index
    render_episode_rewards = 0
    
    # 环境重置
    render_obs, render_share_obs = self.envs.reset()
    render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
    render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
    
    # 开始渲染
    self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
    
    # 自对弈: 加载指定对手策略
    if self.use_selfplay:
        policy_idx = self.render_opponent_index
        self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
        # ... 设置对手状态
    
    # 渲染循环
    while True:
        # 获取动作
        render_actions, render_rnn_states = self.policy.act(...)
        if self.use_selfplay:
            render_opponent_actions, render_opponent_rnn_states = self.eval_opponent_policy.act(...)
            render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
        
        # 环境步进和渲染
        render_obs, render_share_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        
        if render_dones.all():
            break
```

**功能解析:**
- **渲染初始化**: 设置渲染环境和状态
- **对手加载**: 加载指定的对手策略进行对抗
- **实时渲染**: 将训练过程保存为Tacview格式文件
- **循环控制**: 控制渲染直到episode结束

### **8. 模型保存 `save()`**

```python
def save(self, episode):
    # 保存最新模型
    policy_actor_state_dict = self.policy.actor.state_dict()
    torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
    policy_critic_state_dict = self.policy.critic.state_dict()
    torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
    
    # 自对弈: 保存策略版本和性能
    if self.use_selfplay:
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        self.policy_pool[str(episode)] = self.all_args.init_elo
```

**功能解析:**
- **模型保存**: 保存Actor和Critic网络参数
- **版本管理**: 自对弈模式下保存每个episode的策略版本
- **性能记录**: 为新策略分配初始ELO评分

### **9. 对手重置 `reset_opponent()`**

```python
def reset_opponent(self):
    # 选择新的对手策略
    choose_opponents = []
    for policy in self.opponent_policy:
        choose_idx = self.selfplay_algo.choose(self.policy_pool)
        choose_opponents.append(choose_idx)
        policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt'))
        policy.prep_rollout()
    
    logging.info(f" Choose opponents {choose_opponents} for training")
    
    # 清空缓冲区
    self.buffer.clear()
    self.opponent_obs = np.zeros_like(self.opponent_obs)
    self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
    self.opponent_masks = np.ones_like(self.opponent_masks)
    
    # 重置环境
    obs, share_obs = self.envs.reset()
    if self.all_args.n_choose_opponents > 0:
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2:, ...]
        share_obs = share_obs[:, :self.num_agents // 2:, ...]
    
    self.buffer.obs[0] = obs.copy()
    self.buffer.share_obs[0] = share_obs.copy()
```

**功能解析:**
- **对手选择**: 使用自对弈算法选择新的对手策略
- **策略加载**: 将选中的策略加载到对手策略实例中
- **缓冲区清理**: 清空经验回放缓冲区
- **环境重置**: 重置训练环境并重新分配数据

## 🔑 **关键设计特点**

### **1. 多智能体支持**
- 支持共享观察空间
- 智能体间经验共享
- 团队奖励分配

### **2. 自对弈训练**
- FSP算法实现
- 策略池管理
- 动态对手选择

### **3. 并行训练**
- 多进程环境支持
- 批量数据收集
- 高效的经验回放

### **4. 模块化设计**
- 清晰的职责分离
- 可扩展的算法支持
- 灵活的配置选项

## 📊 **使用场景**

1. **2v2空战训练**: 4架战斗机协同作战
2. **多智能体协作**: 学习团队策略和协调
3. **自对弈提升**: 通过对抗不断提升策略质量
4. **实时评估**: 定期评估策略性能
5. **可视化分析**: 生成Tacview格式的作战记录

这个Runner是整个多智能体训练系统的核心，它协调了环境交互、策略训练、自对弈管理和性能评估等所有关键功能。 