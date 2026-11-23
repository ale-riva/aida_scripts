function [ego_pose] = load_vehicle_state(log)
% LOAD_VEHICLE_STATE  Parse vehicle poses.
%
%   All pose fields are guaranteed to exist. Missing fields are
%   replaced with NaN columns of size [N_messages x 1].
%
%   Zero positions and angles are valid. Angles are returned in degrees.


% Helper empty structs

emptyEgoPose = struct( ...
    'stamp', NaN, ...
    'x', NaN, ...
    'y', NaN, ...
    'z', NaN, ...
    'roll', NaN, ...
    'pitch', NaN, ...
    'yaw', NaN, ...
    'frame_id', NaN ...
);


% Configuration

topic = 'estimation__odom__vehicle_state';
sens_stamp_field = 'stamp';


% Initialise output

ego_pose = emptyEgoPose;

if isfield(log, topic)

    vs = log.(topic);

    n_msg = numel(vs.(sens_stamp_field));

    ego_pose.stamp = vs.(sens_stamp_field)(:);
    ego_pose.frame_id = vs.frame_id{1,1};

    % Vehicle-state fields contain one scalar per message
    source_fields = {'pos__x', 'pos__y', 'pos__z', 'rot__roll', 'rot__pitch', 'rot__yaw'};
    pose_fields = {'x', 'y', 'z', 'roll', 'pitch', 'yaw'};

    for i = 1:numel(pose_fields)
        field = pose_fields{i};
        source = source_fields{i};

        ego_pose.(field) = NaN(n_msg, 1);
        if isfield(vs, source)
            ego_pose.(field) = double(vs.(source)(:));
        end
    end

    ego_pose.roll = rad2deg(ego_pose.roll);
    ego_pose.pitch = rad2deg(ego_pose.pitch);
    ego_pose.yaw = rad2deg(ego_pose.yaw);
end

end % FUNCTION END
