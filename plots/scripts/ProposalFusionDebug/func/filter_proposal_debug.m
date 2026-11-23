function filtered = filter_proposal_debug(prop_debug, det)
% FILTER_PROPOSAL_DEBUG  Keep overlays for retained LiDAR IDs and camera-only rows.
%   det.stamp and det.id must come from the original LiDAR detections.
%   Filtered IDs may have NaN gaps. IDs are matched within the LiDAR frame.

filtered = prop_debug;
camera_names = fieldnames(prop_debug);
for i = 1:numel(camera_names)
    camera_name = camera_names{i};
    debug = prop_debug.(camera_name);
    [frame_matched, det_frame] = ismember(debug.lidar_stamp(:), det.stamp(:));
    frame_matched = frame_matched & isfinite(debug.lidar_stamp(:));

    valid = (1:size(debug.lid_box_id, 2)) <= debug.count(:);
    camera_only = valid & debug.has_camera_box == 1 & debug.has_lid_box == 0;
    keep = camera_only;
    for frame = find(frame_matched).'
        ids = det.id(det_frame(frame), :);
        keep(frame, :) = keep(frame, :) | (valid(frame, :) & ...
            debug.has_lid_box(frame, :) == 1 & ...
            ismember(debug.lid_box_id(frame, :), ids(isfinite(ids))));
    end

    frames = frame_matched | any(camera_only, 2);
    filtered.(camera_name) = compactCamera(debug, frames, keep(frames, :));
end
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

fields = {'has_camera_box', 'camera_considered', 'camera_box_id', ...
    'camera_index', 'has_lid_box', 'lid_box_id', 'lid_index'};
for i = 1:numel(fields)
    field = fields{i};
    out.(field) = compactField(debug.(field), frames, keep, destination, shape);
end

box_fields = fieldnames(debug.camera_box);
for i = 1:numel(box_fields)
    field = box_fields{i};
    out.camera_box.(field) = compactField(debug.camera_box.(field), ...
        frames, keep, destination, shape);
    out.proj_lid_box.(field) = compactField(debug.proj_lid_box.(field), ...
        frames, keep, destination, shape);
end
end

function out = compactField(values, frames, keep, destination, shape)
out = NaN(shape);
values = values(frames, :);
out(destination) = values(keep);
end
