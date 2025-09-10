# ACMI 3D 可视化工具使用说明

## 概述
这个MATLAB工具可以将ACMI文件转换为3D动画可视化，支持多架飞机的飞行轨迹显示。

## 系统要求

- MATLAB R2016b 或更高版本
- 支持3D图形显示
- 建议使用较新版本的MATLAB以获得最佳性能

## 兼容性说明

- 支持TacView ACMI文件格式（推荐）
- 支持简单ACMI文件格式
- 已修复MATLAB版本兼容性问题（Alpha属性）
- 适用于Windows、macOS和Linux系统

## 文件说明

### 1. `acmi_3d_visualizer.m` - 完整版可视化工具
- 功能最全面的版本
- 支持复杂的ACMI文件格式
- 包含详细的3D飞机模型
- 提供完整的播放控制功能

### 2. `acmi_simple_3d_viewer.m` - 简化版可视化工具
- 轻量级版本，易于使用
- 支持TacView ACMI文件格式
- 自动解析飞机名称和颜色
- 简化的飞机标记显示
- 适合快速查看和调试

### 3. `test_acmi_viewer.m` - 测试脚本
- 用于测试ACMI文件解析和可视化
- 包含错误处理和调试信息

## 使用方法

### 基本使用
```matlab
% 使用简化版（推荐用于TacView ACMI文件）
acmi_simple_3d_viewer('your_file.acmi');

% 使用完整版
acmi_3d_visualizer('your_file.acmi');

% 运行测试脚本
test_acmi_viewer;
```

### ACMI文件格式要求

#### TacView ACMI格式（推荐）
```
FileType=text/acmi/tacview
FileVersion=2.1
0,ReferenceTime=2020-04-01T00:00:00Z
#0.00
A0100,T=120.1|60.02|6096.0|-1.96e-15|1.59e-15|360.0,Name=F16,Color=Red
A0200,T=119.9|60.01|6096.0|-7.92e-16|1.59e-15|2.75e-15,Name=F16,Color=Red
#0.20
A0100,T=120.1|60.02|6095.8|2.41|-0.39|0.06,Name=F16,Color=Red
```

其中：
- `FileType=text/acmi/tacview` - 文件类型标识
- `FileVersion=2.1` - 文件版本
- `ReferenceTime=...` - 参考时间
- `#时间戳` - 时间点标记
- `A0100,T=X|Y|Z|Heading|Pitch|Roll,Name=飞机名,Color=颜色` - 飞机数据

#### 简单ACMI格式
```
#0.0
1,1000,2000,3000,45,10,5
2,1500,2500,3500,90,15,0
#0.1
1,1100,2100,3100,50,12,6
2,1600,2600,3600,95,18,2
```

其中：
- `#时间戳` - 时间点标记
- `ID,X,Y,Z,Heading,Pitch,Roll` - 飞机数据
  - ID: 飞机编号
  - X,Y,Z: 3D坐标位置
  - Heading: 航向角（度）
  - Pitch: 俯仰角（度）
  - Roll: 滚转角（度）

## 功能特性

### 3D可视化
- 实时3D飞行轨迹显示
- 多架飞机同时显示
- 不同颜色区分不同飞机
- 显示起点和终点标记

### 播放控制
- **播放**: 开始动画播放
- **暂停**: 暂停动画
- **重置**: 回到起始位置
- **速度控制**: 调整播放速度（0.1x - 5x）

### 显示信息
- 实时时间显示
- 飞机位置和方向
- 完整飞行轨迹线

## 界面布局

```
+------------------------------------------+
|              3D主视图区域                |
|                                          |
|                                          |
+------------------------------------------+
| 播放 | 暂停 | 重置 | 速度滑块 | 时间显示  |
+------------------------------------------+
```

## 自定义选项

### 修改飞机标记
在 `create_aircraft_marker` 函数中可以修改飞机标记的样式：
```matlab
% 修改标记大小
size_scale = 10;  % 增大标记

% 修改标记颜色
handle.triangle = fill3(ax, triangle_x, triangle_y, triangle_z, 'red', 'Alpha', 0.8);
```

### 修改轨迹显示
在 `create_simple_3d_view` 函数中可以修改轨迹线的样式：
```matlab
% 修改轨迹线样式
plot3(ax, positions(:,1), positions(:,2), positions(:,3), ...
    'Color', colors(i,:), 'LineWidth', 2, 'LineStyle', '-', 'Alpha', 0.8);
```

## 故障排除

### 常见问题

1. **文件无法打开**
   - 检查文件路径是否正确
   - 确保文件存在且有读取权限

2. **数据解析错误**
   - 检查ACMI文件格式是否正确
   - 确保数据行包含足够的字段

3. **显示异常**
   - 检查数据范围是否合理
   - 确保所有坐标值都是数值类型

### 调试模式
在解析函数中添加调试信息：
```matlab
fprintf('解析第%d行: %s\n', line_num, line);
```

## 扩展功能

### 添加新的数据字段
如果需要显示更多信息（如速度、高度等），可以修改解析函数：
```matlab
% 在parse_acmi_simple函数中添加新字段
velocity = str2double(parts{8});  % 添加速度字段
aircraft_data.(id_str).velocities = ...
    [aircraft_data.(id_str).velocities; velocity];
```

### 添加新的显示元素
可以添加速度矢量、高度指示器等：
```matlab
% 添加速度矢量显示
quiver3(ax, x, y, z, vx, vy, vz, 'Color', color, 'LineWidth', 2);
```

## 性能优化

### 大数据集处理
对于大型ACMI文件，可以考虑：
- 数据采样（跳帧显示）
- 轨迹线简化
- 内存优化

### 实时显示优化
- 减少刷新频率
- 简化3D模型
- 使用更高效的绘图函数

## 示例代码

### 批量处理多个文件
```matlab
files = dir('*.acmi');
for i = 1:length(files)
    fprintf('处理文件: %s\n', files(i).name);
    acmi_simple_3d_viewer(files(i).name);
    pause(2);  % 等待用户查看
end
```

### 数据导出
```matlab
% 将解析的数据保存为MAT文件
save('flight_data.mat', 'time_data', 'aircraft_data');
```

## 技术支持

如果遇到问题，请检查：
1. MATLAB版本兼容性
2. 文件格式是否正确
3. 数据范围是否合理
4. 内存是否充足

## 更新日志

- v1.0: 初始版本，基本3D可视化功能
- v1.1: 添加播放控制功能
- v1.2: 优化性能和用户界面
- v1.3: 添加简化版本，提高易用性 