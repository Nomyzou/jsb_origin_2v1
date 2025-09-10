% 快速测试脚本
clear; clc;

% 测试文件
acmi_file = '4v4_hierarchy_1v1_blood_20250904_1626.txt.acmi';

fprintf('测试文件: %s\n', acmi_file);

try
    acmi_simple_3d_viewer(acmi_file);
    fprintf('测试成功！可视化工具已启动。\n');
catch ME
    fprintf('测试失败: %s\n', ME.message);
end 