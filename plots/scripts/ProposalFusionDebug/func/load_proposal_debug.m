function [out] = load_proposal_debug(log, camera_name)
% LOAD_PROPOSAL_DEBUG  Parse MatchedOverlayList for front, left, or right.
%
%   Stamps are in seconds, and source indices retain their ROS zero base.
%   Overlay fields are NaN-padded [N_messages x max_det] matrices.
%   camera_box and proj_lid_box contain corner, width, and height matrices.
%   Negative IDs mark absent boxes. Presence flags are derived for callers.
%   camera_stamp is the front image stamp.

emptyBBoxStruct = struct( ...
    'height', NaN, ...
    'width', NaN, ...
    'lr_corner_x', NaN, ...
    'lr_corner_y', NaN, ...
    'ul_corner_x', NaN, ...
    'ul_corner_y', NaN ...
);

emptyProposalDebugStruct = struct( ...
    'camera_stamp', NaN, ...
    'lidar_stamp', NaN, ...
    'cam_id', NaN, ...
    'has_camera_box', NaN, ...
    'camera_considered', NaN, ...
    'camera_box_id', NaN, ...
    'camera_index', NaN, ...
    'camera_box', emptyBBoxStruct, ...
    'has_lid_box', NaN, ...
    'lid_box_id', NaN, ...
    'fused_box_id', NaN, ...
    'lid_index', NaN, ...
    'proj_lid_box', emptyBBoxStruct, ...
    'count', 0, ...
    'max_det', 1 ...
);

topic = ['perception_proposal_fusion_detection__overlay_debug__', char(camera_name)];
sens_stamp_field = 'stamp';
out = emptyProposalDebugStruct;

if isfield(log, topic)
    m = log.(topic);

    % Number of messages and detections
    n_msg = numel(m.lidar_stamp__sec);
    if isfield(m, 'count')
        out.count = double(m.count(:));
    elseif isfield(m, 'overlays__has_camera_box')
        % Older logs may omit count.
        flags = m.overlays__has_camera_box;
        if iscell(flags)
            out.count = cellfun(@numel, flags(:));
        else
            out.count = repmat(size(flags, 2), n_msg, 1);
        end
    else
        out.count = zeros(n_msg, 1);
    end
    
    out.max_det = max([1; out.count]);
    n_det = out.max_det;

    % out.stamp = m.(sens_stamp_field)(:);
    out.camera_stamp = double(m.camera_stamp__sec) + double(m.camera_stamp__nanosec) / 1e9;
    out.lidar_stamp = double(m.lidar_stamp__sec) + double(m.lidar_stamp__nanosec) / 1e9;
    % out.frame_id_rel = d.frame_id{1,1};
    % out.frame_id_map = " ";

    out.camera_considered = unpackOverlayField(m, 'overlays__camera_considered', out.count, n_msg, n_det);
    out.camera_box_id = unpackOverlayField(m, 'overlays__camera_box_id', out.count, n_msg, n_det);
    out.camera_index = unpackOverlayField(m, 'overlays__camera_index', out.count, n_msg, n_det);
    out.lid_box_id = unpackOverlayField(m, 'overlays__lidar_box_id', out.count, n_msg, n_det);
    out.fused_box_id = unpackOverlayField(m, 'overlays__fused_box_id', out.count, n_msg, n_det);
    out.lid_index = unpackOverlayField(m, 'overlays__lidar_index', out.count, n_msg, n_det);
    out.has_camera_box = boxPresence(m, 'overlays__has_camera_box', out.camera_box_id, out.count, n_msg, n_det);
    out.has_lid_box = boxPresence(m, 'overlays__has_projected_lidar_box', out.lid_box_id, out.count, n_msg, n_det);
    out.camera_box = unpackBBox(m, 'overlays__camera_bbox', out.count, n_msg, n_det);
    out.proj_lid_box = unpackBBox(m, 'overlays__projected_lidar_bbox', out.count, n_msg, n_det);
    
end

end % FUNCTION END

function out = unpackOverlayField(m, field, count, n_msg, n_det)
if isfield(m, field) && ~iscell(m.(field))
    m.(field) = num2cell(m.(field), 2);
end
out = unpackField(m, field, count, n_msg, n_det);
end

function present = boxPresence(m, legacy_field, ids, count, n_msg, n_det)
if isfield(m, legacy_field)
    present = unpackOverlayField(m, legacy_field, count, n_msg, n_det);
else
    present = double(ids >= 0);
    present(isnan(ids)) = NaN;
end
end

function bbox = unpackBBox(m, field, count, n_msg, n_det)
xyxy = NaN(n_msg, n_det, 4);
if isfield(m, field)
    values = m.(field);
    if ~iscell(values)
        values = num2cell(values, [2 3]);
    end
    for i = 1:n_msg
        boxes = reshape(double(values{i}), [], 4);
        n = min(count(i), size(boxes, 1));
        if n > 0
            xyxy(i, 1:n, :) = reshape(boxes(1:n, :), 1, n, 4);
        end
    end
end

bbox.ul_corner_x = xyxy(:, :, 1);
bbox.ul_corner_y = xyxy(:, :, 2);
bbox.lr_corner_x = xyxy(:, :, 3);
bbox.lr_corner_y = xyxy(:, :, 4);
bbox.width = bbox.lr_corner_x - bbox.ul_corner_x;
bbox.height = bbox.lr_corner_y - bbox.ul_corner_y;
end
