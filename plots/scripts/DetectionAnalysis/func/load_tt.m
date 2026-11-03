function [tt] = load_tt(log)
% LOAD_PERCEPTION  Parse perception detections.
%
%   All detection structs are guaranteed to exist. Missing sources
%   are replaced with empty structs containing NaN fields.


%  Helper empty structs


emptyTrackStruct = struct( ...
    'stamp',NaN,...
    'det_stamp',NaN,...
    'frame_id',NaN, ...
    'id', NaN, ...
    'class', NaN, ...
    'x_cog', NaN, ...
    'y_cog', NaN, ...
    'z_cog', NaN, ...
    'x_map', NaN, ...
    'y_map', NaN, ...
    'z_map', NaN, ...
    'roll', NaN, ...
    'pitch', NaN, ...
    'yaw', NaN, ...
    'yaw_map', NaN, ...
    'vx', NaN, ...
    'vy', NaN, ...
    'vx_map', NaN, ...
    'vy_map', NaN, ...
    'height', NaN, ...
    'length', NaN, ...
    'width', NaN ...
);


sens_stamp_field = "stamp";
% sens_stamp_field = "sensor_stamp__tot";

%  Target TRACKING DETECTIONS
topic = 'perception__target_tracking__obstacles_list';
tt = emptyTrackStruct;

if isfield(log,topic)
    t = log.(topic);
    tt.frame_id_rel = " ";
    tt.frame_id_map = t.frame_id{1,1};
    
    tt.height = replaceZeroWithNaN(t.track_array__height);
    tt.width = replaceZeroWithNaN(t.track_array__width);
    tt.length = replaceZeroWithNaN(t.track_array__length);
    
    tt.class   = t.track_array__label;
    
    tt.x_map   = t.track_array__dyn_state__pos__x;
    tt.y_map   = t.track_array__dyn_state__pos__y;
    tt.z_map   = t.track_array__dyn_state__pos__z;
    tt.yaw_map = t.track_array__dyn_state__rot__yaw;
    tt.id      = t.track_array__id;


    % Padding has zero dimensions; zero coordinates and class IDs are valid.
    tt.x_map(isnan(tt.height) | isnan(tt.width) | isnan(tt.length)) = NaN;
    tt.count    = getDetectionCount(t, tt.x_map);
    tt.max_det = max(sum(~isnan(tt.x_map), 2));

    % Rows are messages; each detection inherits its message timestamp.
    tt.stamp = repmat(t.(sens_stamp_field)(:), 1, size(tt.x_map, 2));
    tt.stamp = (tt.stamp - log.t0);
    tt.det_stamp = t.detection_cloud_stamp__nanosec + t.detection_cloud_stamp__sec*1e9;
    tt.det_stamp = (tt.det_stamp - log.t0);
    not_valid_idx = isnan(tt.x_map);
    tt.stamp(not_valid_idx)=nan;
    tt.det_stamp(not_valid_idx)=nan;
else
    tt.max_det = 1;
end


end % FUNCTION END

