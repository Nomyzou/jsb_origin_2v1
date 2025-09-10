# 导弹渲染代码详解

## 概述

在LAG项目中，导弹的渲染是通过ACMI（Air Combat Maneuvering Instrumentation）格式实现的。导弹作为临时模拟器（temporary simulator）被添加到环境中，并在每次渲染时输出其状态信息。

## 核心代码位置

### 1. 导弹模拟器类
**文件**: `envs/JSBSim/core/simulatior.py`

```python
class MissileSimulator(BaseSimulator):
    """导弹模拟器类"""
    
    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-9L"):
        """创建导弹实例"""
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile
    
    def log(self):
        """输出导弹状态到ACMI文件"""
        if self.is_alive:
            # 导弹飞行时的状态输出
            log_msg = super().log()  # 调用基类的log方法
        elif self.is_done and (not self.render_explosion):
            # 导弹爆炸时的效果输出
            self.render_explosion = True
            log_msg = f"-{self.uid}\n"  # 移除导弹模型
            # 添加爆炸效果
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg
```

### 2. 环境基类渲染方法
**文件**: `envs/JSBSim/envs/env_base.py`

```python
def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', tacview=None):
    """环境渲染方法"""
    if mode == "txt":
        # 写入ACMI文件头
        if not self._create_records:
            with open(filepath, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
            self._create_records = True
        
        # 写入当前时间步的所有模拟器状态
        with open(filepath, mode='a', encoding='utf-8-sig') as f:
            timestamp = self.current_step * self.time_interval
            f.write(f"#{timestamp:.2f}\n")
            
            # 输出飞机状态
            for sim in self._jsbsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    f.write(log_msg + "\n")
            
            # 输出临时模拟器状态（包括导弹）
            for sim in self._tempsims.values():
                log_msg = sim.log()
                if log_msg is not None:
                    f.write(log_msg + "\n")
```

### 3. 导弹创建和添加
**文件**: `envs/JSBSim/tasks/singlecombat_with_missle_task.py`

```python
# 在任务中创建导弹
new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
env.add_temp_simulator(
    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid)
)
self.remaining_missiles[agent_id] -= 1
```

## 导弹渲染流程

### 1. 导弹创建阶段
1. **继承参数**: 导弹从发射飞机继承初始位置、速度、姿态
2. **设置目标**: 指定目标飞机
3. **添加到环境**: 通过`add_temp_simulator()`添加到`_tempsims`字典

### 2. 导弹飞行阶段
- **状态**: `is_alive = True`
- **输出格式**: `uid,T=lon|lat|alt|roll|pitch|yaw,Name=AIM-9L,Color=颜色`
- **示例**: `MA1001,T=120.0|60.0|6000.0|0.0|0.0|45.0,Name=AIM-9L,Color=Red`

### 3. 导弹爆炸阶段
- **状态**: `is_done = True` 且 `is_success = True`
- **输出格式**: 
  ```
  -uid                    # 移除导弹模型
  uidF,T=lon|lat|alt|roll|pitch|yaw,Type=Misc+Explosion,Color=颜色,Radius=300
  ```
- **示例**: 
  ```
  -MA1001
  MA1001F,T=120.0|60.0|5000.0|0.0|0.0|0.0,Type=Misc+Explosion,Color=Red,Radius=300
  ```

### 4. 导弹未命中阶段
- **状态**: `is_done = True` 且 `is_success = False`
- **输出格式**: `-uid` (仅移除导弹模型)

## 导弹参数配置

### 物理参数
```python
# 导弹参数（AIM-9L）
self._g = 9.81          # 重力加速度
self._t_max = 60        # 最大飞行时间（秒）
self._t_thrust = 3      # 发动机工作时间（秒）
self._Isp = 120         # 平均比冲
self._Rc = 300          # 爆炸半径（米）
self._v_min = 150       # 最小速度（m/s）
self._K = 3             # 比例导引系数
```

### 配置文件中设置
```yaml
# envs/JSBSim/configs/1v1/ShootMissile/HierarchySelfplay.yaml
aircraft_configs: {
  A0100: {
    missile: 2  # 每架飞机携带2枚导弹
  }
}
```

## ACMI文件格式详解

### 文件头
```
FileType=text/acmi/tacview
FileVersion=2.1
0,ReferenceTime=2020-04-01T00:00:00Z
```

### 时间步格式
```
#时间戳
物体1状态
物体2状态
...
```

### 物体状态格式
```
uid,T=经度|纬度|高度|滚转|俯仰|偏航,Name=模型名,Color=颜色[,其他属性]
```

### 导弹状态示例
```
#0.20
A0100,T=120.0|60.0|6000.0|0.0|0.0|45.0,Name=F16,Color=Red
B0100,T=120.0|60.2|6000.0|0.0|0.0|225.0,Name=F16,Color=Blue
MA1001,T=120.0|60.0|6000.0|0.0|0.0|45.0,Name=AIM-9L,Color=Red
```

## 查看导弹轨迹

### 使用TacView软件
1. 下载并安装 [TacView](https://www.tacview.net/)
2. 打开生成的`.txt.acmi`文件
3. 可以看到3D导弹轨迹和爆炸效果

### 命令行分析
```bash
# 查看导弹相关行
grep "^M" your_file.txt.acmi

# 查看爆炸效果
grep "Explosion" your_file.txt.acmi

# 统计导弹数量
grep "^M" your_file.txt.acmi | wc -l
```

## 测试导弹渲染

运行测试脚本：
```bash
python renders/test_missile_render.py
```

这将：
1. 创建一个1v1导弹任务环境
2. 手动发射一枚导弹
3. 生成ACMI文件
4. 分析文件内容

## 注意事项

1. **导弹UID**: 导弹的UID通常以'M'开头，例如`MA1001`
2. **颜色继承**: 导弹颜色继承自发射飞机
3. **临时模拟器**: 导弹是临时模拟器，不会在环境重置时保留
4. **爆炸效果**: 只有命中的导弹才会显示爆炸效果
5. **文件编码**: ACMI文件使用UTF-8-BOM编码 