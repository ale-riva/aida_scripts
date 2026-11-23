function [cam_det] = load_camera_detections(log, VIS_CAM)
% LOAD_CAMERA_DETECTIONS  Parse 2D detections for the requested cameras.
%
%   Returns one struct per camera, with NaN-padded detection matrices.
%   count identifies valid detection slots. stamp is the front image stamp
%   shared by the CameraMeasList message.


% Helper empty structs

emptyBBoxStruct = struct( ...
    'height', NaN, ...
    'width', NaN, ...
    'lr_corner_x', NaN, ...
    'lr_corner_y', NaN, ...
    'ul_corner_x', NaN, ...
    'ul_corner_y', NaN ...
);

emptyCameraDetectionStruct = struct( ...
    'stamp', NaN, ...
    'frame_id', "", ...
    'bbox', emptyBBoxStruct, ...
    'id', NaN, ...
    'class', NaN, ...
    'class_name', "", ...
    'score', NaN, ...
    'sensor_type', NaN, ...
    'img_width', NaN, ...
    'img_height', NaN, ...
    'is_valid', NaN, ...
    'count', 0, ...
    'max_det', 1 ...
);


% Configuration

topic = 'perception__camera_detections';
sens_stamp_field = 'stamp';
camera_names = cellstr(VIS_CAM);


% Initialise output

cam_det = struct();

for i = 1:numel(camera_names)
    camera_name = camera_names{i};
    out = emptyCameraDetectionStruct;
    prefix = ['detections_', camera_name, '__'];

    if isfield(log, topic)

        d = log.(topic);

        % Number of messages and detections
        n_msg = numel(d.(sens_stamp_field));
        id_field = [prefix, 'box_id'];
        out.count = zeros(n_msg, 1);
        if isfield(d, id_field)
            ids = d.(id_field);
            if iscell(ids)
                out.count = cellfun(@numel, ids(:));
            else
                out.count = repmat(size(ids, 2), n_msg, 1);
            end
        end
        out.max_det = max([1; out.count]);

        n_det = out.max_det;

        out.stamp = d.(sens_stamp_field)(:);
        if n_msg > 0
            out.frame_id = d.frame_id{1,1};
        end
        out.img_width = double(d.img_width(:));
        out.img_height = double(d.img_height(:));
        out.is_valid = double(d.is_valid(:));

        % Normalise fixed-length detection arrays for unpackField.
        fields = {'box_id', 'class_id', 'score', 'sensor__type', ...
            'bbox__height', 'bbox__width', 'bbox__lower_right_corner_x', ...
            'bbox__lower_right_corner_y', 'bbox__upper_left_corner_x', ...
            'bbox__upper_left_corner_y'};
        for j = 1:numel(fields)
            field = [prefix, fields{j}];
            if isfield(d, field) && ~iscell(d.(field))
                d.(field) = num2cell(d.(field), 2);
            end
        end

        out.id = unpackField(d, [prefix, 'box_id'], out.count, n_msg, n_det);
        out.class = unpackField(d, [prefix, 'class_id'], out.count, n_msg, n_det);
        out.score = unpackField(d, [prefix, 'score'], out.count, n_msg, n_det);
        out.sensor_type = unpackField(d, [prefix, 'sensor__type'], out.count, n_msg, n_det);

        out.bbox.height = unpackField(d, [prefix, 'bbox__height'], out.count, n_msg, n_det);
        out.bbox.width = unpackField(d, [prefix, 'bbox__width'], out.count, n_msg, n_det);
        out.bbox.lr_corner_x = unpackField(d, [prefix, 'bbox__lower_right_corner_x'], out.count, n_msg, n_det);
        out.bbox.lr_corner_y = unpackField(d, [prefix, 'bbox__lower_right_corner_y'], out.count, n_msg, n_det);
        out.bbox.ul_corner_x = unpackField(d, [prefix, 'bbox__upper_left_corner_x'], out.count, n_msg, n_det);
        out.bbox.ul_corner_y = unpackField(d, [prefix, 'bbox__upper_left_corner_y'], out.count, n_msg, n_det);

        out.class_name = strings(n_msg, n_det);
        name_field = [prefix, 'class_name'];
        if isfield(d, name_field)
            for j = 1:n_msg
                names = string(d.(name_field){j});
                n = min(numel(names), out.count(j));
                if n > 0
                    out.class_name(j, 1:n) = reshape(names(1:n), 1, n);
                end
            end
        end
    end

    cam_det.(camera_name) = out;
end

end % FUNCTION END
