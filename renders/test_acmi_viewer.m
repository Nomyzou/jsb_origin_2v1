% ACMI 3D 可视化工具测试脚本
% 测试TacView ACMI文件解析和可视化

clear; clc; close all;

% 设置文件路径
acmi_file = '4v4_hierarchy_1v1_style_20250904_144503.txt.acmi';

fprintf('=== ACMI 3D 可视化工具测试 ===\n');
fprintf('测试文件: %s\n', acmi_file);

% 检查文件是否存在
if ~exist(acmi_file, 'file')
    error('测试文件不存在: %s', acmi_file);
end

% 运行可视化工具
try
    fprintf('\n开始解析和可视化...\n');
    acmi_simple_3d_viewer(acmi_file);
    fprintf('可视化工具启动成功！\n');
    fprintf('使用播放、暂停、重置按钮控制动画。\n');
    fprintf('使用速度滑块调整播放速度。\n');
catch ME
    fprintf('错误: %s\n', ME.message);
    fprintf('详细错误信息:\n');
    disp(getReport(ME, 'extended'));
end

fprintf('\n=== 测试完成 ===\n'); 