function ego_pose = load_vehicle_state(log)

    emptyEgoPose = struct( ...
        'stamp',NaN,...
        'x',NaN, ...
        'y', NaN, ...
        'z', NaN, ...
        'roll', NaN, ...
        'pitch', NaN, ...
        'yaw', NaN, ...
        'frame_id',NaN ...
    );

    stamp_field = "stamp";
    % sens_stamp_field = "sensor_stamp__tot";
   
    topic = 'estimation__ekf__vehicle_state';
    if isfield(log,topic)
        vs = log.(topic);
                    
        ego_pose.x   = replaceZeroWithNaN(vs.pos__x);
        ego_pose.y   = replaceZeroWithNaN(vs.pos__y);
        ego_pose.z   = replaceZeroWithNaN(vs.pos__z);
        ego_pose.yaw = rad2deg(replaceZeroWithNaN(vs.rot__yaw));
        ego_pose.roll = rad2deg(replaceZeroWithNaN(vs.rot__roll));
        ego_pose.pitch = rad2deg(replaceZeroWithNaN(vs.rot__pitch));

        ego_pose.stamp = vs.(stamp_field) - log.t0;
        not_valid_idx = isnan(ego_pose.x);
        ego_pose.stamp(not_valid_idx)=nan;
        ego_pose.frame_id = vs.frame_id{1,1};
    else
        ego_pose = emptyEgoPose;
    end


end % FUNCTION END
