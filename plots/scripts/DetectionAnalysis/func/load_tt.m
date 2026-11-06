function [tt] = load_tt(log)
% LOAD_TT  Parse target-tracking detections.
%
%   Converts variable-length track cell arrays into NaN-padded
%   matrices of size [N_messages x max_tracks].
%
%   All track fields are guaranteed to exist. Missing fields
%   are replaced with NaN arrays.
%
%   Track validity is determined by the message count field.
%   Timestamps are per-message columns, as in load_perception.


% Helper empty structs

emptyTrackStruct = struct( ...
    'stamp', NaN, ...
    'det_stamp', NaN, ...
    'frame_id', NaN, ...
    'frame_id_rel', "", ...
    'frame_id_map', "", ...
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
    'vx_cog', NaN, ...
    'vy_cog', NaN, ...
    'vx_map', NaN, ...
    'vy_map', NaN, ...
    'height', NaN, ...
    'length', NaN, ...
    'width', NaN, ...
    'count', 0, ...
    'max_tracks', 1 ...
);


% Configuration

topic = 'perception__target_tracking__obstacles_list';
sens_stamp_field = 'stamp';


% Initialise output

tt = emptyTrackStruct;

if isfield(log, topic)

    t = log.(topic);

    % Number of messages and tracks
    n_msg = numel(t.(sens_stamp_field));

    tt.count = double(t.count(:));
    tt.max_tracks = max([1; tt.count]);

    n_det = tt.max_tracks;

    tt.stamp = t.(sens_stamp_field)(:);
    tt.frame_id = t.frame_id{1,1};
    tt.frame_id_rel = " ";
    tt.frame_id_map = tt.frame_id;

    tt.id = unpackField(t, 'track_array__id', tt.count, n_msg, n_det);
    tt.class = unpackField(t, 'track_array__label', tt.count, n_msg, n_det);
    tt.x_map = unpackField(t, 'track_array__dyn_state__pos__x', tt.count, n_msg, n_det);
    tt.y_map = unpackField(t, 'track_array__dyn_state__pos__y', tt.count, n_msg, n_det);
    tt.z_map = unpackField(t, 'track_array__dyn_state__pos__z', tt.count, n_msg, n_det);
    tt.yaw_map = unpackField(t, 'track_array__dyn_state__rot__yaw', tt.count, n_msg, n_det);
    tt.vx_cog = unpackField(t, 'track_array__dyn_state__vel__x', tt.count, n_msg, n_det);
    tt.vy_cog = unpackField(t, 'track_array__dyn_state__vel__y', tt.count, n_msg, n_det);
    tt.height = unpackField(t, 'track_array__height', tt.count, n_msg, n_det);
    tt.length = unpackField(t, 'track_array__length', tt.count, n_msg, n_det);
    tt.width = unpackField(t, 'track_array__width', tt.count, n_msg, n_det);

    tt.det_stamp = NaN(n_msg, 1);
    if isfield(t, 'detection_cloud_stamp__sec') && isfield(t, 'detection_cloud_stamp__nanosec')
        tt.det_stamp = double(t.detection_cloud_stamp__sec(:))*1e9 + ...
            double(t.detection_cloud_stamp__nanosec(:));
    end

    % Initialise fields without source data
    tt.x_cog = NaN(n_msg, n_det);
    tt.y_cog = NaN(n_msg, n_det);
    tt.z_cog = NaN(n_msg, n_det);
    tt.roll = NaN(n_msg, n_det);
    tt.pitch = NaN(n_msg, n_det);
    tt.yaw = NaN(n_msg, n_det);
    tt.vx = NaN(n_msg, n_det);
    tt.vy = NaN(n_msg, n_det);
    tt.vx_map = NaN(n_msg, n_det);
    tt.vy_map = NaN(n_msg, n_det);
end


end % FUNCTION END

