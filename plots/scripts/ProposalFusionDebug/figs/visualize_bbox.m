% Prepare a separate frame lookup for each box layer and camera.
camera_fields = "camera_" + cam_idx;
n_frames = numel(timestamps.(camera_fields(1)));
if n_frames == 0
    error('visualize_bbox:EmptyTimeline', 'timestamps.csv contains no frames.');
end

csv_fields = string(T_timestamps.Properties.VariableNames);
sensor_fields = csv_fields(startsWith(csv_fields, ["camera_", "lidar_"]));
timeline_stamps = NaN(n_frames, numel(sensor_fields));
for k = 1:numel(sensor_fields)
    timeline_stamps(:, k) = parse_timestamp_column(T_timestamps.(sensor_fields(k)));
end

box_frames = zeros(n_frames, numel(cam_names), numel(bbox_vis));
for j = 1:numel(bbox_vis)
    box_run = bbox_vis{j};
    for i = 1:numel(cam_names)
        box_frames(:, i, j) = matchBoxFrames(timeline_stamps, box_run.s.(cam_names(i)).stamp);
    end
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
    'bbox_vis', {bbox_vis}, ...
    'box_frames', box_frames, ...
    'camera_axes', camera_axes, ...
    'layout', layout ...
);
fig.WindowKeyPressFcn = @stepFrame;
drawFrame(fig.UserData);


function frames = matchBoxFrames(timeline_stamps, box_stamp)
frames = zeros(size(timeline_stamps, 1), 1);
box_stamp = box_stamp(:);
box_rows = find(isfinite(box_stamp));
if isempty(box_rows) || isempty(timeline_stamps)
    return;
end

[stamps, ~, group] = unique(box_stamp(box_rows));
counts = accumarray(group, 1);
first_row = accumarray(group, box_rows, [], @min);
[matched, location] = ismember(timeline_stamps, stamps);
locations = location(matched);
row_lookup = zeros(size(timeline_stamps));
row_lookup(matched) = first_row(locations);
ambiguous = false(size(timeline_stamps));
ambiguous(matched) = counts(locations) > 1;

% Repeated sensor stamps can point to the same box row.
% Skip CSV rows that point to multiple distinct box rows.
frames = max(row_lookup, [], 2);
conflict = any(row_lookup > 0 & row_lookup ~= frames, 2);
frames(conflict | any(ambiguous, 2)) = 0;
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

    for j = 1:numel(state.bbox_vis)
        box_frame = state.box_frames(frame, i, j);
        if box_frame == 0
            continue;
        end
        box_run = state.bbox_vis{j};
        boxes = box_run.s.(camera_name).bbox;
        drawBoxes(ax, selectBoxes(boxes, box_frame), box_run.col);
    end
end
end


function drawBoxes(ax, boxes, color)
for i = 1:size(boxes, 1)
    position = boxes(i, :) + [1 1 0 0];
    rectangle(ax, 'Position', position, 'EdgeColor', color, 'LineWidth', 1.5);
end
end


function boxes = selectBoxes(bbox, frame)
boxes = [bbox.ul_corner_x(frame, :).', bbox.ul_corner_y(frame, :).', ...
    bbox.width(frame, :).', bbox.height(frame, :).'];
valid = all(isfinite(boxes), 2) & all(boxes(:, 3:4) > 0, 2);
boxes = boxes(valid, :);
end
