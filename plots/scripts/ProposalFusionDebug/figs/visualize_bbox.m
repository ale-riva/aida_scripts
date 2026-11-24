% Prepare the shared CSV timeline and per-camera debug lookups.
camera_fields = "camera_" + cam_idx;
n_frames = numel(timestamps.(camera_fields(1)));
if n_frames == 0
    error('visualize_bbox:EmptyTimeline', 'timestamps.csv contains no frames.');
end

lidar_stamp = NaN(n_frames, 1);
if isfield(timestamps, 'lidar_center')
    lidar_stamp = parseLidarStamp(timestamps.lidar_center);
end

debug_frames = zeros(n_frames, numel(cam_names));
for i = 1:numel(cam_names)
    debug_frames(:, i) = matchDebugFrames(lidar_stamp, dbg_matched.(cam_names(i)).lidar_stamp);
end

% Create the camera tiles.
fig = figure("Name", "Proposal fusion", 'NumberTitle', 'off');
layout = tiledlayout(1, numel(cam_names), 'Padding', 'compact', 'TileSpacing', 'compact');
camera_axes = gobjects(1, numel(cam_names));
for i = 1:numel(cam_names)
    camera_axes(i) = nexttile(layout);
end

fig.UserData = struct( ...
    'frame', 1, ...
    'n_frames', n_frames, ...
    'cam_names', cam_names, ...
    'camera_fields', camera_fields, ...
    'images_dir', images_dir, ...
    'timestamps', timestamps, ...
    'dbg_matched', dbg_matched, ...
    'dbg_unmatched', dbg_unmatched, ...
    'dbg_cam_only', dbg_cam_only, ...
    'MATCHED', MATCHED, ...
    'LID_UNMATCHED', LID_UNMATCHED, ...
    'CAM_UNMATCHED', CAM_UNMATCHED, ...
    'debug_frames', debug_frames, ...
    'camera_axes', camera_axes, ...
    'layout', layout ...
);
fig.WindowKeyPressFcn = @stepFrame;
drawFrame(fig.UserData);


function stamps = parseLidarStamp(stems)
stems = strip(string(stems(:)));
stamps = NaN(size(stems));
rows = find(~ismissing(stems));
matches = regexp(cellstr(stems(rows)), '^\d{8}_\d{6}_\d{9}$', 'once');
rows = rows(~cellfun(@isempty, matches));
if isempty(rows)
    return;
end

whole_seconds = datetime(extractBefore(stems(rows), 16), ...
    'InputFormat', 'yyyyMMdd_HHmmss', 'TimeZone', 'UTC');
nanoseconds = str2double(extractAfter(stems(rows), 16));
stamps(rows) = posixtime(whole_seconds) + nanoseconds / 1e9;
end


function frames = matchDebugFrames(lidar_stamp, debug_stamp)
frames = zeros(size(lidar_stamp));
debug_rows = find(isfinite(debug_stamp));
if isempty(debug_rows)
    return;
end

[stamps, ~, group] = unique(debug_stamp(debug_rows));
counts = accumarray(group, 1);
first_row = accumarray(group, debug_rows, [], @min);
[matched, location] = ismember(lidar_stamp, stamps);
rows = find(matched);
rows = rows(counts(location(rows)) == 1);
frames(rows) = first_row(location(rows));
end


function stepFrame(fig, event)
state = fig.UserData;
switch event.Key
    case 'leftarrow'
        frame = max(state.frame - 1, 1);
    case 'rightarrow'
        frame = min(state.frame + 1, state.n_frames);
    otherwise
        return;
end
if frame == state.frame
    return;
end

state.frame = frame;
fig.UserData = state;
drawFrame(state);
end


function drawFrame(state)
frame = state.frame;
title(state.layout, sprintf('Frame %d / %d', frame, state.n_frames));
for i = 1:numel(state.cam_names)
    ax = state.camera_axes(i);
    camera_name = state.cam_names(i);
    stems = state.timestamps.(state.camera_fields(i));
    stem = strip(string(stems(frame)));
    cla(ax, 'reset');
    axis(ax, 'off');
    title(ax, camera_name, 'Interpreter', 'none');

    if ismissing(stem) || strlength(stem) == 0
        continue;
    end
    filename = fullfile(state.images_dir(i), stem + ".png");
    if ~isfile(filename)
        continue;
    end

    [pixels, color_map] = imread(filename);
    if isempty(color_map)
        imshow(pixels, 'Parent', ax);
    else
        imshow(pixels, color_map, 'Parent', ax);
    end
    title(ax, camera_name, 'Interpreter', 'none');

    debug_frame = state.debug_frames(frame, i);
    if debug_frame == 0
        continue;
    end

    % Matched camera and LiDAR boxes.
    if state.MATCHED
        debug = state.dbg_matched.(camera_name);
        cols = 1:debug.count(debug_frame);

        drawBoxes(ax, selectBoxes(debug.camera_box, debug_frame, cols), [0 0.8 0.2]);
        drawBoxes(ax, selectBoxes(debug.proj_lid_box, debug_frame, cols), [1 0.3 0]);
    end

    % Unmatched LiDAR boxes.
    if state.LID_UNMATCHED
        debug = state.dbg_unmatched.(camera_name);
        cols = 1:debug.count(debug_frame);

        drawBoxes(ax, selectBoxes(debug.proj_lid_box, debug_frame, cols), [1 0.3 0]);
    end

    % Camera-only boxes.
    if state.CAM_UNMATCHED
        debug = state.dbg_cam_only.(camera_name);
        cols = 1:debug.count(debug_frame);

        drawBoxes(ax, selectBoxes(debug.camera_box, debug_frame, cols), [0 0.8 0.2]);
    end
end
end


function drawBoxes(ax, boxes, color)
for i = 1:size(boxes, 1)
    position = boxes(i, :) + [1 1 0 0];
    rectangle(ax, 'Position', position, 'EdgeColor', color, 'LineWidth', 1.5);
end
end


function boxes = selectBoxes(bbox, frame, cols)
boxes = [bbox.ul_corner_x(frame, cols).', bbox.ul_corner_y(frame, cols).', ...
    bbox.width(frame, cols).', bbox.height(frame, cols).'];
valid = all(isfinite(boxes), 2) & all(boxes(:, 3:4) > 0, 2);
boxes = boxes(valid, :);
end