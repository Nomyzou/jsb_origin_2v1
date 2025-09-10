function acmi_3d_visualizer(acmi_file_path)
% ACMI_3D_VISUALIZER - 将ACMI文件转换为3D动画可视化
% 
% 输入:
%   acmi_file_path - ACMI文件的路径
%
% 功能:
%   - 解析ACMI文件中的飞行数据
%   - 创建3D动画显示飞行轨迹
%   - 支持多架飞机的同时显示
%   - 显示时间轴和状态信息

    % 检查文件是否存在
    if ~exist(acmi_file_path, 'file')
        error('ACMI文件不存在: %s', acmi_file_path);
    end
    
    % 解析ACMI文件
    [time_data, aircraft_data] = parse_acmi_file(acmi_file_path);
    
    % 创建3D可视化
    create_3d_animation(time_data, aircraft_data);
end

function [time_data, aircraft_data] = parse_acmi_file(file_path)
% 解析ACMI文件，提取时间和飞机数据
    
    fprintf('正在解析ACMI文件: %s\n', file_path);
    
    % 读取文件内容
    fid = fopen(file_path, 'r');
    if fid == -1
        error('无法打开文件: %s', file_path);
    end
    
    % 初始化数据结构
    time_data = [];
    aircraft_data = struct();
    
    % ACMI文件格式解析
    line_num = 0;
    current_time = 0;
    
    while ~feof(fid)
        line = fgetl(fid);
        line_num = line_num + 1;
        
        if isempty(line) || line(1) == '#'
            continue; % 跳过注释和空行
        end
        
        % 解析时间戳行 (格式: #0.0)
        if line(1) == '#'
            try
                current_time = str2double(line(2:end));
                time_data = [time_data; current_time];
            catch
                warning('无法解析时间戳在第%d行: %s', line_num, line);
            end
            continue;
        end
        
        % 解析飞机数据行
        % 格式: ID,Type,Name,X,Y,Z,Heading,Pitch,Roll,Velocity
        parts = strsplit(line, ',');
        if length(parts) >= 10
            try
                aircraft_id = str2double(parts{1});
                aircraft_type = parts{2};
                aircraft_name = parts{3};
                x = str2double(parts{4});
                y = str2double(parts{5});
                z = str2double(parts{6});
                heading = str2double(parts{7});
                pitch = str2double(parts{8});
                roll = str2double(parts{9});
                velocity = str2double(parts{10});
                
                % 存储数据
                if ~isfield(aircraft_data, sprintf('id_%d', aircraft_id))
                    aircraft_data.(sprintf('id_%d', aircraft_id)) = struct();
                    aircraft_data.(sprintf('id_%d', aircraft_id)).type = aircraft_type;
                    aircraft_data.(sprintf('id_%d', aircraft_id)).name = aircraft_name;
                    aircraft_data.(sprintf('id_%d', aircraft_id)).positions = [];
                    aircraft_data.(sprintf('id_%d', aircraft_id)).orientations = [];
                    aircraft_data.(sprintf('id_%d', aircraft_id)).velocities = [];
                end
                
                aircraft_data.(sprintf('id_%d', aircraft_id)).positions = ...
                    [aircraft_data.(sprintf('id_%d', aircraft_id)).positions; x, y, z];
                aircraft_data.(sprintf('id_%d', aircraft_id)).orientations = ...
                    [aircraft_data.(sprintf('id_%d', aircraft_id)).orientations; heading, pitch, roll];
                aircraft_data.(sprintf('id_%d', aircraft_id)).velocities = ...
                    [aircraft_data.(sprintf('id_%d', aircraft_id)).velocities; velocity];
                
            catch ME
                warning('解析飞机数据失败在第%d行: %s\n错误: %s', line_num, line, ME.message);
            end
        end
    end
    
    fclose(fid);
    
    fprintf('解析完成。时间点数量: %d, 飞机数量: %d\n', ...
        length(time_data), length(fieldnames(aircraft_data)));
end

function create_3d_animation(time_data, aircraft_data)
% 创建3D动画可视化
    
    % 获取飞机ID列表
    aircraft_ids = fieldnames(aircraft_data);
    num_aircraft = length(aircraft_ids);
    
    if num_aircraft == 0
        error('没有找到有效的飞机数据');
    end
    
    % 创建图形窗口
    fig = figure('Name', 'ACMI 3D 可视化', 'Position', [100, 100, 1200, 800]);
    
    % 创建子图布局
    subplot('Position', [0.1, 0.2, 0.6, 0.7]); % 3D主视图
    ax_main = gca;
    hold(ax_main, 'on');
    grid(ax_main, 'on');
    
    % 创建时间轴子图
    subplot('Position', [0.75, 0.2, 0.2, 0.7]); % 时间轴
    ax_time = gca;
    hold(ax_time, 'on');
    
    % 创建控制面板
    subplot('Position', [0.1, 0.05, 0.8, 0.1]); % 控制面板
    ax_control = gca;
    axis(ax_control, 'off');
    
    % 定义颜色方案
    colors = lines(num_aircraft);
    
    % 计算数据范围
    all_positions = [];
    for i = 1:num_aircraft
        id = aircraft_ids{i};
        positions = aircraft_data.(id).positions;
        all_positions = [all_positions; positions];
    end
    
    % 设置坐标轴范围
    x_range = [min(all_positions(:,1)), max(all_positions(:,1))];
    y_range = [min(all_positions(:,2)), max(all_positions(:,2))];
    z_range = [min(all_positions(:,3)), max(all_positions(:,3))];
    
    % 添加一些边距
    margin = 0.1;
    x_range = x_range + diff(x_range) * [-margin, margin];
    y_range = y_range + diff(y_range) * [-margin, margin];
    z_range = z_range + diff(z_range) * [-margin, margin];
    
    % 设置主视图
    xlim(ax_main, x_range);
    ylim(ax_main, y_range);
    zlim(ax_main, z_range);
    xlabel(ax_main, 'X (m)');
    ylabel(ax_main, 'Y (m)');
    zlabel(ax_main, 'Z (m)');
    title(ax_main, 'ACMI 3D 飞行轨迹');
    view(ax_main, 45, 30);
    
    % 绘制完整轨迹
    for i = 1:num_aircraft
        id = aircraft_ids{i};
        positions = aircraft_data.(id).positions;
        
        % 绘制轨迹线
        plot3(ax_main, positions(:,1), positions(:,2), positions(:,3), ...
            'Color', colors(i,:), 'LineWidth', 1, 'LineStyle', '--');
        
        % 绘制起点和终点
        plot3(ax_main, positions(1,1), positions(1,2), positions(1,3), ...
            'o', 'Color', colors(i,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
        plot3(ax_main, positions(end,1), positions(end,2), positions(end,3), ...
            's', 'Color', colors(i,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(i,:));
    end
    
    % 创建动画对象
    aircraft_handles = zeros(num_aircraft, 1);
    for i = 1:num_aircraft
        id = aircraft_ids{i};
        positions = aircraft_data.(id).positions;
        
        % 创建飞机3D模型（简化版）
        aircraft_handles(i) = create_aircraft_model(ax_main, positions(1,:), colors(i,:), ...
            aircraft_data.(id).name);
    end
    
    % 创建时间轴
    plot(ax_time, time_data, 1:length(time_data), 'b-', 'LineWidth', 2);
    xlabel(ax_time, '时间 (s)');
    ylabel(ax_time, '时间点');
    title(ax_time, '时间轴');
    grid(ax_time, 'on');
    
    % 创建时间指示器
    time_indicator = plot(ax_time, time_data(1), 1, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % 创建控制按钮
    play_button = uicontrol('Style', 'pushbutton', 'String', '播放', ...
        'Position', [50, 20, 60, 30], 'Callback', @play_animation);
    pause_button = uicontrol('Style', 'pushbutton', 'String', '暂停', ...
        'Position', [120, 20, 60, 30], 'Callback', @pause_animation);
    reset_button = uicontrol('Style', 'pushbutton', 'String', '重置', ...
        'Position', [190, 20, 60, 30], 'Callback', @reset_animation);
    
    % 创建速度滑块
    speed_slider = uicontrol('Style', 'slider', 'Min', 0.1, 'Max', 5, 'Value', 1, ...
        'Position', [270, 20, 100, 30], 'Callback', @change_speed);
    speed_text = uicontrol('Style', 'text', 'String', '速度: 1x', ...
        'Position', [380, 20, 60, 30]);
    
    % 创建时间显示
    time_text = uicontrol('Style', 'text', 'String', sprintf('时间: %.2f s', time_data(1)), ...
        'Position', [450, 20, 100, 30]);
    
    % 存储动画数据
    animation_data = struct();
    animation_data.time_data = time_data;
    animation_data.aircraft_data = aircraft_data;
    animation_data.aircraft_handles = aircraft_handles;
    animation_data.time_indicator = time_indicator;
    animation_data.time_text = time_text;
    animation_data.speed = 1;
    animation_data.current_frame = 1;
    animation_data.is_playing = false;
    animation_data.timer = [];
    
    % 将数据存储到图形中
    setappdata(fig, 'animation_data', animation_data);
    
    % 初始化显示
    update_display(1);
    
    function play_animation(~, ~)
        animation_data = getappdata(fig, 'animation_data');
        if ~animation_data.is_playing
            animation_data.is_playing = true;
            setappdata(fig, 'animation_data', animation_data);
            
            % 创建定时器
            animation_data.timer = timer('ExecutionMode', 'fixedRate', ...
                'Period', 0.1 / animation_data.speed, ...
                'TimerFcn', @(~,~) update_frame());
            start(animation_data.timer);
        end
    end
    
    function pause_animation(~, ~)
        animation_data = getappdata(fig, 'animation_data');
        if animation_data.is_playing
            animation_data.is_playing = false;
            setappdata(fig, 'animation_data', animation_data);
            
            if ~isempty(animation_data.timer)
                stop(animation_data.timer);
                delete(animation_data.timer);
                animation_data.timer = [];
            end
        end
    end
    
    function reset_animation(~, ~)
        animation_data = getappdata(fig, 'animation_data');
        pause_animation();
        animation_data.current_frame = 1;
        setappdata(fig, 'animation_data', animation_data);
        update_display(1);
    end
    
    function change_speed(~, ~)
        animation_data = getappdata(fig, 'animation_data');
        animation_data.speed = get(speed_slider, 'Value');
        set(speed_text, 'String', sprintf('速度: %.1fx', animation_data.speed));
        setappdata(fig, 'animation_data', animation_data);
        
        % 如果正在播放，重新启动定时器
        if animation_data.is_playing
            pause_animation();
            play_animation();
        end
    end
    
    function update_frame()
        animation_data = getappdata(fig, 'animation_data');
        animation_data.current_frame = animation_data.current_frame + 1;
        
        if animation_data.current_frame > length(animation_data.time_data)
            animation_data.current_frame = 1; % 循环播放
        end
        
        setappdata(fig, 'animation_data', animation_data);
        update_display(animation_data.current_frame);
    end
    
    function update_display(frame_idx)
        animation_data = getappdata(fig, 'animation_data');
        
        % 更新飞机位置
        for i = 1:num_aircraft
            id = aircraft_ids{i};
            positions = aircraft_data.(id).positions;
            orientations = aircraft_data.(id).orientations;
            
            if frame_idx <= size(positions, 1)
                pos = positions(frame_idx, :);
                orient = orientations(frame_idx, :);
                
                % 更新飞机模型位置和方向
                update_aircraft_model(aircraft_handles(i), pos, orient);
            end
        end
        
        % 更新时间指示器
        set(animation_data.time_indicator, 'XData', time_data(frame_idx), 'YData', frame_idx);
        
        % 更新时间显示
        set(animation_data.time_text, 'String', sprintf('时间: %.2f s', time_data(frame_idx)));
        
        % 刷新显示
        drawnow;
    end
end

function handle = create_aircraft_model(ax, position, color, name)
% 创建简化的飞机3D模型
    
    % 飞机尺寸
    length_scale = 10;
    width_scale = 5;
    height_scale = 2;
    
    % 创建飞机主体（长方体）
    x = position(1);
    y = position(2);
    z = position(3);
    
    % 飞机主体
    body_vertices = [
        x-length_scale/2, y-width_scale/2, z-height_scale/2;
        x+length_scale/2, y-width_scale/2, z-height_scale/2;
        x+length_scale/2, y+width_scale/2, z-height_scale/2;
        x-length_scale/2, y+width_scale/2, z-height_scale/2;
        x-length_scale/2, y-width_scale/2, z+height_scale/2;
        x+length_scale/2, y-width_scale/2, z+height_scale/2;
        x+length_scale/2, y+width_scale/2, z+height_scale/2;
        x-length_scale/2, y+width_scale/2, z+height_scale/2;
    ];
    
    body_faces = [
        1 2 3 4;  % 底面
        5 6 7 8;  % 顶面
        1 2 6 5;  % 前面
        3 4 8 7;  % 后面
        1 4 8 5;  % 左面
        2 3 7 6;  % 右面
    ];
    
    % 绘制飞机主体
    handle.body = patch('Vertices', body_vertices, 'Faces', body_faces, ...
        'FaceColor', color, 'EdgeColor', 'black', 'Parent', ax);
    
    % 创建机翼
    wing_vertices = [
        x, y-width_scale, z;
        x+length_scale/3, y-width_scale, z;
        x+length_scale/3, y+width_scale, z;
        x, y+width_scale, z;
    ];
    
    wing_faces = [1 2 3 4];
    
    handle.wings = patch('Vertices', wing_vertices, 'Faces', wing_faces, ...
        'FaceColor', color, 'EdgeColor', 'black', 'Parent', ax);
    
    % 添加标签
    handle.label = text(x, y, z + height_scale + 2, name, ...
        'Color', color, 'FontSize', 8, 'HorizontalAlignment', 'center', 'Parent', ax);
end

function update_aircraft_model(handle, position, orientation)
% 更新飞机模型的位置和方向
    
    % 飞机尺寸
    length_scale = 10;
    width_scale = 5;
    height_scale = 2;
    
    x = position(1);
    y = position(2);
    z = position(3);
    heading = orientation(1);
    pitch = orientation(2);
    roll = orientation(3);
    
    % 创建旋转矩阵
    R_heading = [cosd(heading), -sind(heading), 0; ...
                 sind(heading), cosd(heading), 0; ...
                 0, 0, 1];
    
    R_pitch = [cosd(pitch), 0, sind(pitch); ...
               0, 1, 0; ...
               -sind(pitch), 0, cosd(pitch)];
    
    R_roll = [1, 0, 0; ...
              0, cosd(roll), -sind(roll); ...
              0, sind(roll), cosd(roll)];
    
    R = R_heading * R_pitch * R_roll;
    
    % 更新飞机主体
    body_vertices = [
        -length_scale/2, -width_scale/2, -height_scale/2;
        length_scale/2, -width_scale/2, -height_scale/2;
        length_scale/2, width_scale/2, -height_scale/2;
        -length_scale/2, width_scale/2, -height_scale/2;
        -length_scale/2, -width_scale/2, height_scale/2;
        length_scale/2, -width_scale/2, height_scale/2;
        length_scale/2, width_scale/2, height_scale/2;
        -length_scale/2, width_scale/2, height_scale/2;
    ];
    
    % 应用旋转和平移
    body_vertices = (R * body_vertices')' + repmat([x, y, z], size(body_vertices, 1), 1);
    
    set(handle.body, 'Vertices', body_vertices);
    
    % 更新机翼
    wing_vertices = [
        0, -width_scale, 0;
        length_scale/3, -width_scale, 0;
        length_scale/3, width_scale, 0;
        0, width_scale, 0;
    ];
    
    wing_vertices = (R * wing_vertices')' + repmat([x, y, z], size(wing_vertices, 1), 1);
    
    set(handle.wings, 'Vertices', wing_vertices);
    
    % 更新标签位置
    label_pos = (R * [0, 0, height_scale + 2]')' + [x, y, z];
    set(handle.label, 'Position', label_pos);
end

% 示例使用函数
function example_usage()
% 示例：如何使用这个可视化工具
    
    % 假设ACMI文件路径
    acmi_file = 'path/to/your/file.acmi';
    
    % 调用可视化函数
    acmi_3d_visualizer(acmi_file);
    
    fprintf('ACMI 3D可视化已启动。\n');
    fprintf('使用播放、暂停、重置按钮控制动画。\n');
    fprintf('使用速度滑块调整播放速度。\n');
end 