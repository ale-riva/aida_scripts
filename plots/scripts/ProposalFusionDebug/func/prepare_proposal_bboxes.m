function [selected] = prepare_proposal_bboxes(prop_debug, MATCHED, LID_UNMATCHED, CAM_UNMATCHED)
% PREPARE_PROPOSAL_BBOXES  Select camera and LiDAR rectangles for each frame.
%   Each frame contains separate camera and LiDAR [x, y, width, height] arrays.

selected = struct();
camera_names = fieldnames(prop_debug);
for i = 1:numel(camera_names)
    camera_name = camera_names{i};
    debug = prop_debug.(camera_name);

    valid = (1:size(debug.camera_box_id, 2)) <= debug.count(:);
    has_camera = debug.camera_box_id >= 0;
    has_lidar = debug.lid_box_id >= 0;
    matched = has_camera & has_lidar;
    keep_camera = valid & ((MATCHED & matched) | (CAM_UNMATCHED & has_camera & ~has_lidar));
    keep_lidar = valid & ((MATCHED & matched) | (LID_UNMATCHED & has_lidar & ~has_camera));

    n_frames = numel(debug.lidar_stamp);
    camera_boxes = cell(n_frames, 1);
    lidar_boxes = cell(n_frames, 1);
    for frame = 1:n_frames
        camera_boxes{frame} = selectBoxes(debug.camera_box, frame, keep_camera(frame, :));
        lidar_boxes{frame} = selectBoxes(debug.proj_lid_box, frame, keep_lidar(frame, :));
    end

    selected.(camera_name).lidar_stamp = debug.lidar_stamp(:);
    selected.(camera_name).camera_boxes = camera_boxes;
    selected.(camera_name).lidar_boxes = lidar_boxes;
end

end

function boxes = selectBoxes(bbox, frame, keep)
boxes = [bbox.ul_corner_x(frame, keep).', bbox.ul_corner_y(frame, keep).', ...
    bbox.width(frame, keep).', bbox.height(frame, keep).'];
valid = all(isfinite(boxes), 2) & all(boxes(:, 3:4) > 0, 2);
boxes = boxes(valid, :);
end
