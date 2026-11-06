function [lid_det] = load_perception(log)
% LOAD_PERCEPTION  Parse perception detections.
%
%   Converts variable-length detection cell arrays into NaN-padded
%   matrices of size [N_messages x max_detections].
%
%   All detection fields are guaranteed to exist. Missing fields
%   are replaced with NaN arrays.
%
%   Detection validity is determined by the message count field.


% Helper empty structs

emptyBBoxStruct = struct( ...
    'height', NaN, ...
    'width', NaN, ...
    'lr_corner_x', NaN, ...
    'lr_corner_y', NaN, ...
    'ul_corner_x', NaN, ...
    'ul_corner_y', NaN ...
);

emptyDetectionStruct = struct( ...
    'stamp', NaN, ...
    'frame_id_rel', "", ...
    'frame_id_map', "", ...
    'bbox', emptyBBoxStruct, ...
    'id', NaN, ...
    'meas_class', NaN, ...
    'class', NaN, ...
    'x_cog', NaN, ...
    'y_cog', NaN, ...
    'z_cog', NaN, ...
    'x_map', NaN, ...
    'y_map', NaN, ...
    'orientation', NaN, ...
    'orientation_map', NaN, ...
    'score', NaN, ...
    'height', NaN, ...
    'length', NaN, ...
    'width', NaN, ...
    'pt_count', NaN, ...
    'sensor_type', NaN, ...
    'smallest_box', NaN, ...
    'count', 0, ...
    'max_det', 1 ...
);


% Configuration

topic = 'perception__multimodal_3d_detections';
sens_stamp_field = 'stamp';


% Initialise output

lid_det = emptyDetectionStruct;

if isfield(log, topic)

    d = log.(topic);

    % Number of messages and detections
    n_msg = numel(d.(sens_stamp_field));

    lid_det.count = double(d.count(:));
    lid_det.max_det = max([1; lid_det.count]);

    n_det = lid_det.max_det;

    lid_det.stamp = d.(sens_stamp_field)(:);
    lid_det.frame_id_rel = d.frame_id{1,1};
    lid_det.frame_id_map = " ";

    lid_det.id = unpackField(d, 'meas_array__box_id', lid_det.count, n_msg, n_det);
    lid_det.x_cog = unpackField(d, 'meas_array__x_cog', lid_det.count, n_msg, n_det);
    lid_det.y_cog = unpackField(d, 'meas_array__y_cog', lid_det.count, n_msg, n_det);
    lid_det.z_cog = unpackField(d, 'meas_array__z_cog', lid_det.count, n_msg, n_det);
    lid_det.orientation = rad2deg(unpackField(d, 'meas_array__orientation', lid_det.count, n_msg, n_det));
    lid_det.height = unpackField(d, 'meas_array__height', lid_det.count, n_msg, n_det);
    lid_det.length = unpackField(d, 'meas_array__length', lid_det.count, n_msg, n_det);
    lid_det.width = unpackField(d, 'meas_array__width', lid_det.count, n_msg, n_det);
    lid_det.score = unpackField(d, 'meas_array__score', lid_det.count, n_msg, n_det);
    lid_det.class = unpackField(d, 'meas_array__meas_class', lid_det.count, n_msg, n_det);
    lid_det.meas_class = lid_det.class;
    lid_det.sensor_type = unpackField(d, 'meas_array__sensor_type', lid_det.count, n_msg, n_det);
    lid_det.smallest_box = unpackField(d, 'meas_array__smallest_box', lid_det.count, n_msg, n_det);

    lid_det.bbox.height = unpackField(d, 'meas_array__bbox__height', lid_det.count, n_msg, n_det);
    lid_det.bbox.width = unpackField(d, 'meas_array__bbox__width', lid_det.count, n_msg, n_det);
    lid_det.bbox.lr_corner_x = unpackField(d, 'meas_array__bbox__lower_right_corner_x',lid_det.count, n_msg, n_det);
    lid_det.bbox.lr_corner_y = unpackField(d, 'meas_array__bbox__lower_right_corner_y', lid_det.count, n_msg, n_det);
    lid_det.bbox.ul_corner_x = unpackField(d, 'meas_array__bbox__upper_left_corner_x', lid_det.count, n_msg, n_det);
    lid_det.bbox.ul_corner_y = unpackField(d, 'meas_array__bbox__upper_left_corner_y', lid_det.count, n_msg, n_det);

    % initialize empty
    lid_det.x_map = NaN(n_msg, n_det);
    lid_det.y_map = NaN(n_msg, n_det);
    lid_det.orientation_map = NaN(n_msg, n_det);
    lid_det.pt_count = NaN(n_msg, n_det);
end

end % FUNCTION END
