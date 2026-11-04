function [lid_det] = load_perception(log)
% LOAD_PERCEPTION  Parse perception detections.
%
%   All detection structs are guaranteed to exist. Missing sources
%   are replaced with empty structs containing NaN fields.


%  Helper empty structs

emptyBBoxStruct = struct( ...
    "height", NaN, ...
    "width", NaN, ...
    'lr_corner_x', NaN, ...
    'lr_corner_y', NaN, ...
    'ul_corner_x', NaN, ...
    'ul_corner_y', NaN ...
);

emptyDetectionStruct = struct( ...
    'stamp',NaN,...
    'frame_id',NaN, ...
    'bbox', emptyBBoxStruct, ...
    'id', NaN, ...
    'meas_class', NaN, ...
    'x_cog', NaN, ...
    'y_cog', NaN, ...
    'z_cog', NaN, ...
    'orientation', NaN, ...
    'score', NaN, ...
    'height', NaN, ...
    'length', NaN, ...
    'width', NaN ...
);


sens_stamp_field = "stamp";
% sens_stamp_field = "sensor_stamp__tot";

%  LIDAR CLUSTERING DETECTIONS
topic = 'perception__multimodal_3d_detections';
if isfield(log,topic)
    d = log.(topic);
    lid_det.frame_id_rel = d.frame_id{1,1};
    lid_det.frame_id_map = " ";

    lid_det.x_map   = NaN;
    lid_det.y_map   = NaN;
    lid_det.z_cog   = NaN;
    lid_det.orientation_map = NaN;
  
    lid_det.x_cog   = d.meas_array__x_cog;
    lid_det.y_cog   = d.meas_array__y_cog;
    lid_det.z_cog   = d.meas_array__z_cog;
    lid_det.orientation = rad2deg(d.meas_array__orientation);
    lid_det.height = replaceZeroWithNaN(d.meas_array__height);
    lid_det.width = replaceZeroWithNaN(d.meas_array__width);
    lid_det.length = replaceZeroWithNaN(d.meas_array__length);
    
    lid_det.score   = replaceZeroWithNaN(d.meas_array__score);
    lid_det.class   = double(d.meas_array__meas_class);
    % Padding has zero dimensions; zero coordinates and class IDs are valid.
    lid_det.x_cog(isnan(lid_det.height) | isnan(lid_det.width) | isnan(lid_det.length)) = NaN;
    lid_det.count    = getDetectionCount(d, lid_det.x_cog);
    lid_det.max_det = max(sum(~isnan(lid_det.x_cog), 2));

    % Rows are messages; each detection inherits its message timestamp.
    lid_det.stamp = repmat(d.(sens_stamp_field)(:), 1, size(lid_det.x_cog, 2));
    not_valid_idx = isnan(lid_det.x_cog);
    lid_det.stamp(not_valid_idx)=nan;
else
    lid_det = emptyDetectionStruct;
    lid_det.max_det = 1;
end


end % FUNCTION END

