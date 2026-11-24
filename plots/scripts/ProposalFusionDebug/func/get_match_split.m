function [matched, unmatched, dbg_matched, dbg_unmatched, dbg_cam_only] = ...
    get_match_split(prop_debug, det)
% GET_MATCH_SPLIT  Split fused detections and proposal fusion debug.
%   Matched detections have both LiDAR and camera bounding boxes in at
%   least one camera.
%   Unmatched detections have a LiDAR bounding box but no camera match.
%   Detections are associated by lidar_stamp and fused_box_id.
%
%   Proposal debug is split per camera into matched, unmatched with
%   a valid fused box, and camera-only proposals.

matched_keep = false(size(det.id));
unmatched_keep = false(size(det.id));

dbg_matched = struct();
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

    % Preserve common frame indexing across the debug splits.
    frames = frame_matched | any(valid_cam_only, 2);

    dbg_matched.(camera_name) = ...
        compactCamera(debug, frames, dbg_matched_keep(frames, :));

    dbg_unmatched.(camera_name) = ...
        compactCamera(debug, frames, dbg_unmatched_keep(frames, :));

    dbg_cam_only.(camera_name) = ...
        compactCamera(debug, frames, valid_cam_only(frames, :));
end

% Give priority to matched detections across cameras.
unmatched_keep = unmatched_keep & ~matched_keep;

matched = maskDetections(det, matched_keep);
unmatched = maskDetections(det, unmatched_keep);
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


function out = compactCamera(debug, frames, keep)
out = debug;

out.camera_stamp = debug.camera_stamp(frames);
out.lidar_stamp = debug.lidar_stamp(frames);

if ~isscalar(debug.cam_id) && numel(debug.cam_id) == numel(frames)
    out.cam_id = debug.cam_id(frames);
end

out.count = sum(keep, 2);
out.max_det = max([1; out.count]);

shape = [numel(out.count), out.max_det];

[row, ~] = find(keep);
cols = cumsum(keep, 2);
destination = row + (cols(keep) - 1) * shape(1);

fields = {'camera_considered', 'camera_box_id', ...
    'camera_index', 'lid_box_id', 'fused_box_id', 'lid_index'};

for i = 1:numel(fields)
    field = fields{i};
    out.(field) = compactField(debug.(field), ...
        frames, keep, destination, shape);
end

box_fields = fieldnames(debug.camera_box);
for i = 1:numel(box_fields)
    field = box_fields{i};

    out.camera_box.(field) = compactField(debug.camera_box.(field), ...
        frames, keep, destination, shape);
end

box_fields = fieldnames(debug.proj_lid_box);
for i = 1:numel(box_fields)
    field = box_fields{i};

    out.proj_lid_box.(field) = compactField(debug.proj_lid_box.(field), ...
        frames, keep, destination, shape);
end
end


function out = compactField(values, frames, keep, destination, shape)
out = NaN(shape);
values = values(frames, :);
out(destination) = values(keep);
end