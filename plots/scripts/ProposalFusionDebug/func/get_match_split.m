function [matched, unmatched] = get_match_split(prop_debug, det)
% get_match_split  Split fused detections into matched and unmatched.
%   Matched detections have both LiDAR and camera bounding boxes in at
%   least one camera.
%   Unmatched detections have a LiDAR bounding box but no camera match.
%   Detections are associated by lidar_stamp and fused_box_id.

matched_keep = false(size(det.id));
unmatched_keep = false(size(det.id));

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

    for frame = find(frame_matched).'
        row = det_frame(frame);
        ids = det.id(row, :);

        matched_ids = debug.fused_box_id(frame, valid_matched(frame, :));
        unmatched_ids = debug.fused_box_id(frame, valid_unmatched(frame, :));

        matched_keep(row, :) = matched_keep(row, :) | ...
            (isfinite(ids) & ismember(ids, matched_ids));

        unmatched_keep(row, :) = unmatched_keep(row, :) | ...
            (isfinite(ids) & ismember(ids, unmatched_ids));
    end
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