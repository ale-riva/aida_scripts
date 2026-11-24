function [matched, unmatched, dbg_matched_camera, dbg_matched_lidar, dbg_unmatched, dbg_cam_only] = ...
    get_match_split(prop_debug, det, cam_det)
% GET_MATCH_SPLIT  Split fused detections and proposal fusion debug.
%   Matched detections have both LiDAR and camera bounding boxes in at
%   least one camera.
%   Unmatched detections have a LiDAR bounding box but no camera match.
%   Detections are associated by lidar_stamp and fused_box_id.
%
%   Proposal debug is split per camera into matched, unmatched with
%   a valid fused box, and camera-only proposals. Each debug output has
%   the cam_det schema, with stamp set to the LiDAR overlay timestamp.
%   Matched camera boxes and LiDAR projections share the same row and slot.
%
%   Camera-only proposals are retained only if their corresponding
%   detection exists in cam_det, matching camera_stamp and camera_box_id.

matched_keep = false(size(det.id));
unmatched_keep = false(size(det.id));

dbg_matched_camera = struct();
dbg_matched_lidar = struct();
dbg_unmatched = struct();
dbg_cam_only = struct();

camera_names = fieldnames(prop_debug);
for i = 1:numel(camera_names)
    camera_name = camera_names{i};
    debug = prop_debug.(camera_name);

    [frame_matched, det_frame] = ismember(debug.lidar_stamp(:), det.stamp(:));
    frame_matched = frame_matched & isfinite(debug.lidar_stamp(:));

    valid = (1:size(debug.fused_box_id, 2)) <= debug.count(:);

    has_lidar = debug.lid_box_id >= 0;
    has_camera = debug.camera_box_id >= 0;
    has_fused = debug.fused_box_id >= 0;

    valid_matched = valid & has_lidar & has_camera & has_fused;
    valid_unmatched = valid & has_lidar & ~has_camera & has_fused;
    valid_cam_only = valid & has_camera & debug.lid_box_id < 0;

    dbg_matched_keep = false(size(valid));
    dbg_unmatched_keep = false(size(valid));

    for frame = find(frame_matched).'
        row = det_frame(frame);
        ids = det.id(row, :);

        % Retain proposals associated with existing fused detections.
        retained = ismember(debug.fused_box_id(frame, :), ...
            ids(isfinite(ids)));

        dbg_matched_keep(frame, :) = ...
            valid_matched(frame, :) & retained;

        dbg_unmatched_keep(frame, :) = ...
            valid_unmatched(frame, :) & retained;

        % Extract the corresponding fused IDs.
        matched_ids = debug.fused_box_id(frame, dbg_matched_keep(frame, :));
        unmatched_ids = debug.fused_box_id(frame, dbg_unmatched_keep(frame, :));

        % Accumulate fused detections across cameras.
        matched_keep(row, :) = matched_keep(row, :) | ...
            (isfinite(ids) & ismember(ids, matched_ids));

        unmatched_keep(row, :) = unmatched_keep(row, :) | ...
            (isfinite(ids) & ismember(ids, unmatched_ids));
    end

    % Retain camera-only proposals associated with filtered camera detections.
    [debug_cam_only, dbg_cam_only_keep] = ...
        filterCameraOnly(debug, cam_det.(camera_name), valid_cam_only);

    % Preserve common frame indexing across the debug splits.
    frames = frame_matched | any(dbg_cam_only_keep, 2);

    dbg_matched_camera.(camera_name) = compactCamera( ...
        debug.lidar_stamp, debug.camera_box, debug.camera_box_id, ...
        frames, dbg_matched_keep(frames, :));

    dbg_matched_lidar.(camera_name) = compactCamera( ...
        debug.lidar_stamp, debug.proj_lid_box, debug.lid_box_id, ...
        frames, dbg_matched_keep(frames, :));

    dbg_unmatched.(camera_name) = compactCamera( ...
        debug.lidar_stamp, debug.proj_lid_box, debug.lid_box_id, ...
        frames, dbg_unmatched_keep(frames, :));

    dbg_cam_only.(camera_name) = compactCamera( ...
        debug.lidar_stamp, debug_cam_only.camera_box, debug.camera_box_id, ...
        frames, dbg_cam_only_keep(frames, :));
end

% Give priority to matched detections across cameras.
unmatched_keep = unmatched_keep & ~matched_keep;

matched = maskDetections(det, matched_keep);
unmatched = maskDetections(det, unmatched_keep);
end


function [out, keep] = filterCameraOnly(debug, cam_det, valid)
% Retain camera-only proposals associated with existing camera detections.

out = debug;
keep = false(size(valid));
source_idx = zeros(size(valid));

% Match camera frame timestamps.
[frame_matched, cam_frame] = ...
    ismember(debug.camera_stamp(:), cam_det.stamp(:));

frame_matched = frame_matched & isfinite(debug.camera_stamp(:));

% Process only frames containing camera-only proposals.
for frame = find(frame_matched & any(valid, 2)).'
    row = cam_frame(frame);
    ids = cam_det.id(row, :);

    cols = find(valid(frame, :));

    % Match camera bounding box IDs within the corresponding frame.
    [retained, cam_cols] = ...
        ismember(debug.camera_box_id(frame, cols), ids);

    cols = cols(retained);
    cam_cols = cam_cols(retained);

    keep(frame, cols) = true;

    % Store linear indices of the corresponding camera detections.
    source_idx(frame, cols) = ...
        row + (cam_cols - 1) * size(cam_det.id, 1);
end

% Recover bounding box coordinates from the source camera detections.
box_fields = fieldnames(debug.camera_box);
for i = 1:numel(box_fields)
    field = box_fields{i};

    if ~isfield(cam_det.bbox, field)
        continue;
    end

    out.camera_box.(field)(keep) = ...
        cam_det.bbox.(field)(source_idx(keep));
end
end


function out = maskDetections(det, keep)
out = det;

fields = {'id', 'meas_class', 'class', ...
    'x_cog', 'y_cog', 'z_cog', 'range', ...
    'x_map', 'y_map', 'z_map', ...
    'orientation', 'orientation_map', 'roll_map', 'pitch_map', ...
    'score', 'height', 'length', 'width', ...
    'pt_count', 'sensor_type', 'smallest_box'};

remove = ~keep;
for i = 1:numel(fields)
    field = fields{i};
    if isfield(out, field)
        out.(field)(remove) = NaN;
    end
end

box_fields = fieldnames(out.bbox);
for i = 1:numel(box_fields)
    field = box_fields{i};
    out.bbox.(field)(remove) = NaN;
end

out.count = sum(keep, 2);
out.max_det = max([1; out.count]);
end


function out = compactCamera(stamps, bbox, ids, frames, keep)
count = sum(keep, 2);
max_det = max([1; count]);
shape = [numel(count), max_det];

[row, ~] = find(keep);
cols = cumsum(keep, 2);
destination = row + (cols(keep) - 1) * shape(1);

out = struct( ...
    'stamp', stamps(frames), ...
    'frame_id', "", ...
    'bbox', struct(), ...
    'id', compactField(ids, frames, keep, destination, shape), ...
    'class', NaN(shape), ...
    'class_name', strings(shape), ...
    'score', NaN(shape), ...
    'sensor_type', NaN(shape), ...
    'img_width', NaN(shape(1), 1), ...
    'img_height', NaN(shape(1), 1), ...
    'is_valid', NaN(shape(1), 1), ...
    'count', count, ...
    'max_det', max_det);

box_fields = fieldnames(bbox);
for i = 1:numel(box_fields)
    field = box_fields{i};
    out.bbox.(field) = compactField(bbox.(field), ...
        frames, keep, destination, shape);
end
end


function out = compactField(values, frames, keep, destination, shape)
out = NaN(shape);
values = values(frames, :);
out(destination) = values(keep);
end