function acmi_simple_3d_viewer(acmi_file)
% 简化的ACMI 3D查看器 - 只显示轨迹点
% 输入: acmi_file - ACMI文件路径

    % 解析ACMI文件
    [aircraft_data, time_data] = parse_acmi_file(acmi_file);
    
    if isempty(aircraft_data)
        error('无法解析ACMI文件或文件为空');
    end
    
    % 获取飞机ID列表
    aircraft_ids = fieldnames(aircraft_data);
    num_aircraft = length(aircraft_ids);
    
    % 创建图形窗口
    fig = figure('Name', 'ACMI 3D轨迹查看器', 'Position', [100, 100, 1200, 800]);
    
    % 创建3D坐标轴
    ax = axes('Parent', fig);
    hold(ax, 'on');
    grid(ax, 'on');
    xlabel(ax, 'X (m)');
    ylabel(ax, 'Y (m)');
    zlabel(ax, 'Z (m)');
    title(ax, 'ACMI 3D轨迹');
    
    % 计算所有轨迹的边界
    all_positions = [];
    for i = 1:num_aircraft
        id = aircraft_ids{i};
        positions = aircraft_data.(id).positions;
        all_positions = [all_positions; positions];
    end
    
    % 检查是否有位置数据
    if isempty(all_positions)
        error('没有找到任何飞机位置数据，请检查ACMI文件格式');
    end
    
    % 设置坐标轴范围
    x_range = [min(all_positions(:,1)), max(all_positions(:,1))];
    y_range = [min(all_positions(:,2)), max(all_positions(:,2))];
    z_range = [min(all_positions(:,3)), max(all_positions(:,3))];
    
    % 添加一些边距
    margin = 100;
    xlim(ax, [x_range(1) - margin, x_range(2) + margin]);
    ylim(ax, [y_range(1) - margin, y_range(2) + margin]);
    zlim(ax, [z_range(1) - margin, z_range(2) + margin]);
    
    % 设置视角
    view(ax, 45, 30);
    
    % 创建颜色映射
    colors = lines(num_aircraft);
    
    % 初始化轨迹线对象
    trajectory_lines = cell(num_aircraft, 1);
    trajectory_points = cell(num_aircraft, 1);
    
    % 为每个飞机创建轨迹线
    for i = 1:num_aircraft
        id = aircraft_ids{i};
        color = colors(i, :);
        
        % 创建空的轨迹线
        trajectory_lines{i} = plot3(ax, NaN, NaN, NaN, 'Color', color, 'LineWidth', 2, 'DisplayName', id);
        
        % 创建轨迹点
        trajectory_points{i} = scatter3(ax, NaN, NaN, NaN, 50, color, 'filled', 'DisplayName', [id ' 当前位置']);
    end
    
    % 添加图例
    legend(ax, 'show', 'Location', 'northeast');
    
    % 创建控制面板
    control_panel = uipanel('Parent', fig, 'Title', '控制面板', ...
        'Position', [0.8, 0.1, 0.18, 0.8]);
    
    % 播放/暂停按钮
    play_button = uicontrol('Parent', control_panel, 'Style', 'pushbutton', ...
        'String', '播放', 'Position', [10, 300, 80, 30], ...
        'Callback', @toggle_play);
    
    % 重置按钮
    reset_button = uicontrol('Parent', control_panel, 'Style', 'pushbutton', ...
        'String', '重置', 'Position', [10, 260, 80, 30], ...
        'Callback', @reset_animation);
    
    % 速度滑块
    speed_slider = uicontrol('Parent', control_panel, 'Style', 'slider', ...
        'Min', 0.1, 'Max', 5, 'Value', 1, ...
        'Position', [10, 220, 100, 20], ...
        'Callback', @update_speed);
    
    speed_text = uicontrol('Parent', control_panel, 'Style', 'text', ...
        'String', '速度: 1x', 'Position', [10, 200, 100, 20]);
    
    % 时间显示
    time_text = uicontrol('Parent', control_panel, 'Style', 'text', ...
        'String', '时间: 0.00 s', 'Position', [10, 160, 100, 20]);
    
    % 帧数显示
    frame_text = uicontrol('Parent', control_panel, 'Style', 'text', ...
        'String', '帧: 0 / 0', 'Position', [10, 120, 100, 20]);
    
    % 动画数据
    anim_data = struct();
    anim_data.aircraft_data = aircraft_data;
    anim_data.time_data = time_data;
    anim_data.trajectory_lines = trajectory_lines;
    anim_data.trajectory_points = trajectory_points;
    anim_data.current_frame = 0;
    anim_data.is_playing = false;
    anim_data.speed = 1;
    anim_data.timer = [];
    anim_data.play_button = play_button;
    anim_data.speed_text = speed_text;
    anim_data.time_text = time_text;
    anim_data.frame_text = frame_text;
    
    % 存储动画数据
    setappdata(fig, 'anim_data', anim_data);
    
    % 更新帧数显示
    update_frame_display();
    
    % 开始动画
    start_animation();
    
    % 回调函数
    function toggle_play(~, ~)
        anim_data = getappdata(fig, 'anim_data');
        if anim_data.is_playing
            stop_animation();
            set(anim_data.play_button, 'String', '播放');
        else
            start_animation();
            set(anim_data.play_button, 'String', '暂停');
        end
    end
    
    function reset_animation(~, ~)
        anim_data = getappdata(fig, 'anim_data');
        anim_data.current_frame = 0;
        setappdata(fig, 'anim_data', anim_data);
        
        % 清除所有轨迹
        for i = 1:num_aircraft
            set(anim_data.trajectory_lines{i}, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
            set(anim_data.trajectory_points{i}, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
        end
        
        update_display(0);
        update_frame_display();
    end
    
    function update_speed(~, ~)
        anim_data = getappdata(fig, 'anim_data');
        anim_data.speed = get(speed_slider, 'Value');
        setappdata(fig, 'anim_data', anim_data);
        set(anim_data.speed_text, 'String', sprintf('速度: %.1fx', anim_data.speed));
        
        if anim_data.is_playing
            stop_animation();
            start_animation();
        end
    end
    
    function start_animation()
        anim_data = getappdata(fig, 'anim_data');
        anim_data.is_playing = true;
        setappdata(fig, 'anim_data', anim_data);
        
        % 创建定时器
        interval = 0.1 / anim_data.speed; % 基础间隔100ms
        anim_data.timer = timer('ExecutionMode', 'fixedRate', ...
            'Period', interval, 'TimerFcn', @update_frame);
        setappdata(fig, 'anim_data', anim_data);
        
        start(anim_data.timer);
    end
    
    function stop_animation()
        anim_data = getappdata(fig, 'anim_data');
        anim_data.is_playing = false;
        setappdata(fig, 'anim_data', anim_data);
        
        if ~isempty(anim_data.timer)
            stop(anim_data.timer);
            delete(anim_data.timer);
            anim_data.timer = [];
            setappdata(fig, 'anim_data', anim_data);
        end
    end
    
    function update_frame()
        anim_data = getappdata(fig, 'anim_data');
        anim_data.current_frame = anim_data.current_frame + 1;
        
        if anim_data.current_frame > length(anim_data.time_data)
            anim_data.current_frame = 1;
        end
        
        setappdata(fig, 'anim_data', anim_data);
        update_display(anim_data.current_frame);
        update_frame_display();
    end
    
    function update_display(frame_idx)
        anim_data = getappdata(fig, 'anim_data');
        
        % 更新每个飞机的轨迹
        for i = 1:num_aircraft
            id = aircraft_ids{i};
            positions = anim_data.aircraft_data.(id).positions;
            
            if frame_idx > 0 && frame_idx <= size(positions, 1)
                % 获取当前帧及之前的所有位置
                current_positions = positions(1:frame_idx, :);
                
                % 更新轨迹线
                set(anim_data.trajectory_lines{i}, ...
                    'XData', current_positions(:,1), ...
                    'YData', current_positions(:,2), ...
                    'ZData', current_positions(:,3));
                
                % 更新当前位置点
                current_pos = positions(frame_idx, :);
                set(anim_data.trajectory_points{i}, ...
                    'XData', current_pos(1), ...
                    'YData', current_pos(2), ...
                    'ZData', current_pos(3));
            end
        end
        
        % 更新时间显示
        if frame_idx > 0 && frame_idx <= length(anim_data.time_data)
            set(anim_data.time_text, 'String', sprintf('时间: %.2f s', anim_data.time_data(frame_idx)));
        end
        
        drawnow;
    end
    
    function update_frame_display()
        anim_data = getappdata(fig, 'anim_data');
        set(anim_data.frame_text, 'String', sprintf('帧: %d / %d', ...
            anim_data.current_frame, length(anim_data.time_data)));
    end
end

function [aircraft_data, time_data] = parse_acmi_file(acmi_file)
% 解析ACMI文件，提取飞机轨迹数据
% 输入: acmi_file - ACMI文件路径
% 输出: aircraft_data - 飞机数据结构
%       time_data - 时间数据

    aircraft_data = struct();
    time_data = [];
    
    try
        % 读取文件
        fid = fopen(acmi_file, 'r');
        if fid == -1
            error('无法打开文件: %s', acmi_file);
        end
        
        % 读取所有行
        lines = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        lines = lines{1};
        
        % 解析数据
        current_time = 0;
        aircraft_positions = containers.Map();
        aircraft_orientations = containers.Map();
        
        for i = 1:length(lines)
            line = strtrim(lines{i});
            
            % 跳过空行和注释
            if isempty(line) || line(1) == '#'
                continue;
            end
            
            % 解析以0:开头的行
            if startsWith(line, '0:')
                parts = strsplit(line, ' ');
                
                % 如果只有2个部分，这是时间戳行
                if length(parts) == 2
                    time_str = parts{2};
                    current_time = str2double(time_str);
                    time_data = [time_data; current_time];
                % 如果有4个或更多部分，这是飞机数据行
                elseif length(parts) >= 4
                    aircraft_id = parts{1};
                    x = str2double(parts{2});
                    y = str2double(parts{3});
                    z = str2double(parts{4});
                    
                    % 存储位置
                    if ~isKey(aircraft_positions, aircraft_id)
                        aircraft_positions(aircraft_id) = [];
                    end
                    aircraft_positions(aircraft_id) = [aircraft_positions(aircraft_id); x, y, z];
                    
                    % 存储方向（如果有）
                    if length(parts) >= 5
                        heading = str2double(parts{5});
                        if ~isKey(aircraft_orientations, aircraft_id)
                            aircraft_orientations(aircraft_id) = [];
                        end
                        aircraft_orientations(aircraft_id) = [aircraft_orientations(aircraft_id); heading, 0, 0];
                    end
                end
            end
        end
        
        % 转换为结构体
        aircraft_ids = keys(aircraft_positions);
        for i = 1:length(aircraft_ids)
            id = aircraft_ids{i};
            aircraft_data.(id).positions = aircraft_positions(id);
            
            if isKey(aircraft_orientations, id)
                aircraft_data.(id).orientations = aircraft_orientations(id);
            else
                % 如果没有方向数据，创建默认值
                num_positions = size(aircraft_data.(id).positions, 1);
                aircraft_data.(id).orientations = zeros(num_positions, 3);
            end
        end
        
        % 添加调试信息
        fprintf('解析完成: 找到 %d 个时间点, %d 架飞机\n', length(time_data), length(aircraft_ids));
        for i = 1:length(aircraft_ids)
            id = aircraft_ids{i};
            num_pos = size(aircraft_data.(id).positions, 1);
            fprintf('飞机 %s: %d 个位置点\n', id, num_pos);
        end
        
    catch ME
        error('解析ACMI文件时出错: %s', ME.message);
    end
end

% 示例使用
function example()
    % 使用示例
    acmi_simple_3d_viewer('your_acmi_file.acmi');
end 